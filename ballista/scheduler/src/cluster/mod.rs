// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

pub mod event;
pub mod memory;

#[cfg(test)]
#[allow(clippy::uninlined_format_args)]
pub mod test_util;

use crate::cluster::memory::{InMemoryClusterState, InMemoryJobState};
#[cfg(feature = "etcd")]
use crate::cluster::storage::etcd::EtcdClient;

use crate::config::{SchedulerConfig, TaskDistribution};
use crate::scheduler_server::SessionBuilder;
use crate::state::execution_graph::{create_task_info, ExecutionGraph, TaskDescription};
use crate::state::task_manager::JobInfoCache;
use ballista_core::config::BallistaConfig;
use ballista_core::error::Result;
use ballista_core::serde::protobuf::{
    job_status, AvailableTaskSlots, ExecutorHeartbeat, JobStatus,
};
use ballista_core::serde::scheduler::{ExecutorData, ExecutorMetadata, PartitionId};
use ballista_core::utils::default_session_builder;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::prelude::SessionContext;
use futures::Stream;
use log::{debug, info, warn};
use std::collections::{HashMap, HashSet};
use std::pin::Pin;
use std::sync::Arc;

#[derive(Clone)]
pub struct BallistaCluster {
    cluster_state: Arc<InMemoryClusterState>,
    job_state: Arc<InMemoryJobState>,
}

impl BallistaCluster {
    pub fn new(
        cluster_state: Arc<InMemoryClusterState>,
        job_state: Arc<InMemoryJobState>,
    ) -> Self {
        Self {
            cluster_state,
            job_state,
        }
    }

    pub fn new_memory(
        scheduler: impl Into<String>,
        session_builder: SessionBuilder,
    ) -> Self {
        Self {
            cluster_state: Arc::new(InMemoryClusterState::default()),
            job_state: Arc::new(InMemoryJobState::new(scheduler, session_builder)),
        }
    }

    pub async fn new_from_config(config: &SchedulerConfig) -> Result<Self> {
        let scheduler = config.scheduler_name();
        Ok(BallistaCluster::new_memory(
            scheduler,
            default_session_builder,
        ))
    }

    pub fn cluster_state(&self) -> Arc<InMemoryClusterState> {
        self.cluster_state.clone()
    }

    pub fn job_state(&self) -> Arc<InMemoryJobState> {
        self.job_state.clone()
    }
}

/// Stream of `ExecutorHeartbeat`. This stream should contain all `ExecutorHeartbeats` received
/// by any schedulers with a shared `ClusterState`
pub type ExecutorHeartbeatStream = Pin<Box<dyn Stream<Item = ExecutorHeartbeat> + Send>>;

/// A task bound with an executor to execute.
/// BoundTask.0 is the executor id; While BoundTask.1 is the task description.
pub type BoundTask = (String, TaskDescription);

/// ExecutorSlot.0 is the executor id; While ExecutorSlot.1 is for slot number.
pub type ExecutorSlot = (String, u32);

/// A trait that contains the necessary method to maintain a globally consistent view of cluster resources
pub trait ClusterState: Send + Sync + 'static {
    /// Initialize when it's necessary, especially for state with backend storage
    fn init(&self) -> Result<()> {
        Ok(())
    }

    /// Bind the ready to running tasks from [`active_jobs`] with available executors.
    ///
    /// If `executors` is provided, only bind slots from the specified executor IDs
    fn bind_schedulable_tasks(
        &self,
        distribution: TaskDistribution,
        active_jobs: Arc<HashMap<String, JobInfoCache>>,
        executors: Option<HashSet<String>>,
    ) -> impl std::future::Future<Output = Result<Vec<BoundTask>>> + Send;

    /// Unbind executor and task when a task finishes or fails. It will increase the executor
    /// available task slots.
    ///
    /// This operations should be atomic. Either all reservations are cancelled or none are
    fn unbind_tasks(
        &self,
        executor_slots: Vec<ExecutorSlot>,
    ) -> impl std::future::Future<Output = Result<()>> + Send;

    /// Register a new executor in the cluster.
    fn register_executor(
        &self,
        metadata: ExecutorMetadata,
        spec: ExecutorData,
    ) -> impl std::future::Future<Output = Result<()>> + Send;

    /// Save the executor metadata. This will overwrite existing metadata for the executor ID
    fn save_executor_metadata(&self, metadata: ExecutorMetadata);

    /// Get executor metadata for the provided executor ID. Returns an error if the executor does not exist
    fn get_executor_metadata(&self, executor_id: &str) -> Result<ExecutorMetadata>;

    /// Save the executor heartbeat
    fn save_executor_heartbeat(&self, heartbeat: ExecutorHeartbeat);

    /// Remove the executor from the cluster
    fn remove_executor(
        &self,
        executor_id: &str,
    ) -> impl std::future::Future<Output = Result<()>> + Send;

    /// Return a map of the last seen heartbeat for all active executors
    fn executor_heartbeats(&self) -> HashMap<String, ExecutorHeartbeat>;

    /// Get executor heartbeat for the provided executor ID. Return None if the executor does not exist
    fn get_executor_heartbeat(&self, executor_id: &str) -> Option<ExecutorHeartbeat>;
}

/// Events related to the state of jobs. Implementations may or may not support all event types.
#[derive(Debug, Clone, PartialEq)]
pub enum JobStateEvent {
    /// Event when a job status has been updated
    JobUpdated {
        /// Job ID of updated job
        job_id: String,
        /// New job status
        status: JobStatus,
    },
    /// Event when a scheduler acquires ownership of the job. This happens
    /// either when a scheduler submits a job (in which case ownership is implied)
    /// or when a scheduler acquires ownership of a running job release by a
    /// different scheduler
    JobAcquired {
        /// Job ID of the acquired job
        job_id: String,
        /// The scheduler which acquired ownership of the job
        owner: String,
    },
    /// Event when a scheduler releases ownership of a still active job
    JobReleased {
        /// Job ID of the released job
        job_id: String,
    },
    /// Event when a new session has been created
    SessionCreated {
        session_id: String,
        config: BallistaConfig,
    },
    /// Event when a session configuration has been updated
    SessionUpdated {
        session_id: String,
        config: BallistaConfig,
    },
}

/// Stream of `JobStateEvent`. This stream should contain all `JobStateEvent`s received
/// by any schedulers with a shared `ClusterState`
pub type JobStateEventStream = Pin<Box<dyn Stream<Item = JobStateEvent> + Send>>;

/// A trait that contains the necessary methods for persisting state related to executing jobs
pub trait JobState: Send + Sync {
    /// Accept job into  a scheduler's job queue. This should be called when a job is
    /// received by the scheduler but before it is planned and may or may not be saved
    /// in global state
    fn accept_job(&self, job_id: &str, job_name: &str, queued_at: u64) -> Result<()>;

    /// Get the number of queued jobs. If it's big, then it means the scheduler is too busy.
    /// In normal case, it's better to be 0.
    fn pending_job_number(&self) -> usize;

    /// Submit a new job to the `JobState`. It is assumed that the submitter owns the job.
    /// In local state the job should be save as `JobStatus::Active` and in shared state
    /// it should be saved as `JobStatus::Running` with `scheduler` set to the current scheduler
    fn submit_job(&self, job_id: String, graph: &ExecutionGraph) -> Result<()>;

    /// Return a `Vec` of all active job IDs in the `JobState`
    fn get_jobs(&self) -> Result<HashSet<String>>;

    /// Return the SQL query of the job if it exists
    fn get_sql(&self, job_id: &str) -> Result<Option<String>>;

    fn register_sql(&self, job_id: &str, sql: String);

    /// Fetch the job status
    fn get_job_status(&self, job_id: &str) -> Result<Option<JobStatus>>;

    /// Get the `ExecutionGraph` for job. The job may or may not belong to the caller
    /// and should return the `ExecutionGraph` for the given job (if it exists) at the
    /// time this method is called with no guarantees that the graph has not been
    /// subsequently updated by another scheduler.
    fn get_execution_graph(&self, job_id: &str) -> Result<Option<ExecutionGraph>>;

    /// Persist the current state of an owned job to global state. This should fail
    /// if the job is not owned by the caller.
    fn save_job(&self, job_id: &str, graph: &ExecutionGraph) -> Result<()>;

    /// Mark a job which has not been submitted as failed. This should be called if a job fails
    /// during planning (and does not yet have an `ExecutionGraph`)
    fn fail_unscheduled_job(&self, job_id: &str, reason: String) -> Result<()>;

    /// Delete a job from the global state
    fn remove_job(&self, job_id: &str) -> Result<()>;

    /// Attempt to acquire ownership of the given job. If the job is still in a running state
    /// and is successfully acquired by the caller, return the current `ExecutionGraph`,
    /// otherwise return `None`
    fn try_acquire_job(&self, job_id: &str) -> Result<Option<ExecutionGraph>>;

    /// Get a stream of all `JobState` events. An event should be published any time that status
    /// of a job changes in state
    fn job_state_events(&self) -> Result<JobStateEventStream>;

    /// Get the `SessionContext` associated with `session_id`. Returns an error if the
    /// session does not exist
    fn get_session(&self, session_id: &str) -> Result<Arc<SessionContext>>;

    /// Create a new saved session
    fn create_session(&self, config: &BallistaConfig) -> Result<Arc<SessionContext>>;

    // Update a new saved session. If the session does not exist, a new one will be created
    fn update_session(
        &self,
        session_id: &str,
        config: &BallistaConfig,
    ) -> Result<Arc<SessionContext>>;
}

pub(crate) async fn bind_task_bias(
    mut slots: Vec<&mut AvailableTaskSlots>,
    active_jobs: Arc<HashMap<String, JobInfoCache>>,
    if_skip: fn(Arc<dyn ExecutionPlan>) -> bool,
) -> Vec<BoundTask> {
    let mut schedulable_tasks: Vec<BoundTask> = vec![];

    let total_slots = slots.iter().fold(0, |acc, s| acc + s.slots);
    if total_slots == 0 {
        warn!("Not enough available executor slots for task running!!!");
        return schedulable_tasks;
    }

    // Sort the slots by descending order
    slots.sort_by(|a, b| Ord::cmp(&b.slots, &a.slots));

    let mut idx_slot = 0usize;
    let mut slot = &mut slots[idx_slot];
    for (job_id, job_info) in active_jobs.iter() {
        if !matches!(job_info.status, Some(job_status::Status::Running(_))) {
            debug!(
                "Job {} is not in running status and will be skipped",
                job_id
            );
            continue;
        }
        let mut graph = job_info.execution_graph.write().await;
        let session_id = graph.session_id().to_string();
        let mut black_list = vec![];
        while let Some((running_stage, task_id_gen)) =
            graph.fetch_running_stage(&black_list)
        {
            if if_skip(running_stage.plan.clone()) {
                info!(
                    "Will skip stage {}/{} for bias task binding",
                    job_id, running_stage.stage_id
                );
                black_list.push(running_stage.stage_id);
                continue;
            }
            // We are sure that it will at least bind one task by going through the following logic.
            // It will not go into a dead loop.
            let runnable_tasks = running_stage
                .task_infos
                .iter_mut()
                .enumerate()
                .filter(|(_partition, info)| info.is_none())
                .take(total_slots as usize)
                .collect::<Vec<_>>();
            for (partition_id, task_info) in runnable_tasks {
                // Assign [`slot`] with a slot available slot number larger than 0
                while slot.slots == 0 {
                    idx_slot += 1;
                    if idx_slot >= slots.len() {
                        return schedulable_tasks;
                    }
                    slot = &mut slots[idx_slot];
                }
                let executor_id = slot.executor_id.clone();
                let task_id = *task_id_gen;
                *task_id_gen += 1;
                *task_info = Some(create_task_info(executor_id.clone(), task_id));

                let partition = PartitionId {
                    job_id: job_id.clone(),
                    stage_id: running_stage.stage_id,
                    partition_id,
                };
                let task_desc = TaskDescription {
                    session_id: session_id.clone(),
                    partition,
                    stage_attempt_num: running_stage.stage_attempt_num,
                    task_id,
                    task_attempt: running_stage.task_failure_numbers[partition_id],
                    plan: running_stage.plan.clone(),
                };
                schedulable_tasks.push((executor_id, task_desc));

                slot.slots -= 1;
            }
        }
    }

    schedulable_tasks
}

pub(crate) async fn bind_task_round_robin(
    mut slots: Vec<&mut AvailableTaskSlots>,
    active_jobs: Arc<HashMap<String, JobInfoCache>>,
    if_skip: fn(Arc<dyn ExecutionPlan>) -> bool,
) -> Vec<BoundTask> {
    let mut schedulable_tasks: Vec<BoundTask> = vec![];

    let mut total_slots = slots.iter().fold(0, |acc, s| acc + s.slots);
    if total_slots == 0 {
        warn!("Not enough available executor slots for task running!!!");
        return schedulable_tasks;
    }
    info!("Total slot number is {}", total_slots);

    // Sort the slots by descending order
    slots.sort_by(|a, b| Ord::cmp(&b.slots, &a.slots));

    let mut idx_slot = 0usize;
    for (job_id, job_info) in active_jobs.iter() {
        if !matches!(job_info.status, Some(job_status::Status::Running(_))) {
            debug!(
                "Job {} is not in running status and will be skipped",
                job_id
            );
            continue;
        }
        let mut graph = job_info.execution_graph.write().await;
        let session_id = graph.session_id().to_string();
        let mut black_list = vec![];
        while let Some((running_stage, task_id_gen)) =
            graph.fetch_running_stage(&black_list)
        {
            if if_skip(running_stage.plan.clone()) {
                info!(
                    "Will skip stage {}/{} for round robin task binding",
                    job_id, running_stage.stage_id
                );
                black_list.push(running_stage.stage_id);
                continue;
            }
            // We are sure that it will at least bind one task by going through the following logic.
            // It will not go into a dead loop.
            let runnable_tasks = running_stage
                .task_infos
                .iter_mut()
                .enumerate()
                .filter(|(_partition, info)| info.is_none())
                .take(total_slots as usize)
                .collect::<Vec<_>>();
            for (partition_id, task_info) in runnable_tasks {
                // Move to the index which has available slots
                if idx_slot >= slots.len() {
                    idx_slot = 0;
                }
                if slots[idx_slot].slots == 0 {
                    idx_slot = 0;
                }

                // Since the slots is a vector with descending order, and the total available slots is larger than 0,
                // we are sure the available slot number at idx_slot is larger than 1
                let slot = &mut slots[idx_slot];
                let executor_id = slot.executor_id.clone();
                let task_id = *task_id_gen;
                *task_id_gen += 1;
                *task_info = Some(create_task_info(executor_id.clone(), task_id));

                let partition = PartitionId {
                    job_id: job_id.clone(),
                    stage_id: running_stage.stage_id,
                    partition_id,
                };
                let task_desc = TaskDescription {
                    session_id: session_id.clone(),
                    partition,
                    stage_attempt_num: running_stage.stage_attempt_num,
                    task_id,
                    task_attempt: running_stage.task_failure_numbers[partition_id],
                    plan: running_stage.plan.clone(),
                };
                schedulable_tasks.push((executor_id, task_desc));

                idx_slot += 1;
                slot.slots -= 1;
                total_slots -= 1;
                if total_slots == 0 {
                    return schedulable_tasks;
                }
            }
        }
    }

    schedulable_tasks
}
