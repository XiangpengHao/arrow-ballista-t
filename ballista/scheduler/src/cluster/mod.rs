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
pub mod test;

use crate::cluster::memory::{InMemoryClusterState, InMemoryJobState};
#[cfg(feature = "etcd")]
use crate::cluster::storage::etcd::EtcdClient;

use crate::config::{SchedulerConfig, TaskDistribution};
use crate::scheduler_server::SessionBuilder;
use crate::state::execution_graph::ExecutionGraph;
use crate::state::executor_manager::ExecutorReservation;
use ballista_core::config::BallistaConfig;
use ballista_core::error::Result;
use ballista_core::serde::protobuf::{AvailableTaskSlots, ExecutorHeartbeat, JobStatus};
use ballista_core::serde::scheduler::{ExecutorData, ExecutorMetadata};
use ballista_core::utils::default_session_builder;
use datafusion::prelude::SessionContext;
use futures::Stream;
use std::collections::{HashMap, HashSet};
use std::pin::Pin;
use std::sync::Arc;

#[derive(Clone)]
pub struct BallistaCluster {
    cluster_state: Arc<InMemoryClusterState>,
    job_state: Arc<dyn JobState>,
}

impl BallistaCluster {
    pub fn new(
        cluster_state: Arc<InMemoryClusterState>,
        job_state: Arc<dyn JobState>,
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

    pub fn job_state(&self) -> Arc<dyn JobState> {
        self.job_state.clone()
    }
}

/// Stream of `ExecutorHeartbeat`. This stream should contain all `ExecutorHeartbeats` received
/// by any schedulers with a shared `ClusterState`
pub type ExecutorHeartbeatStream = Pin<Box<dyn Stream<Item = ExecutorHeartbeat> + Send>>;

/// A trait that contains the necessary method to maintain a globally consistent view of cluster resources
#[tonic::async_trait]
pub trait ClusterState: Send + Sync + 'static {
    /// Initialize when it's necessary, especially for state with backend storage
    async fn init(&self) -> Result<()> {
        Ok(())
    }

    /// Reserve up to `num_slots` executor task slots. If not enough task slots are available, reserve
    /// as many as possible.
    ///
    /// If `executors` is provided, only reserve slots of the specified executor IDs
    async fn reserve_slots(
        &self,
        num_slots: u32,
        distribution: TaskDistribution,
        executors: Option<HashSet<String>>,
    ) -> Result<Vec<ExecutorReservation>>;

    /// Reserve exactly `num_slots` executor task slots. If not enough task slots are available,
    /// returns an empty vec
    ///
    /// If `executors` is provided, only reserve slots of the specified executor IDs
    async fn reserve_slots_exact(
        &self,
        num_slots: u32,
        distribution: TaskDistribution,
        executors: Option<HashSet<String>>,
    ) -> Result<Vec<ExecutorReservation>>;

    /// Cancel the specified reservations. This will make reserved executor slots available to other
    /// tasks.
    /// This operations should be atomic. Either all reservations are cancelled or none are
    async fn cancel_reservations(
        &self,
        reservations: Vec<ExecutorReservation>,
    ) -> Result<()>;

    /// Register a new executor in the cluster. If `reserve` is true, then the executors task slots
    /// will be reserved and returned in the response and none of the new executors task slots will be
    /// available to other tasks.
    fn register_executor(
        &self,
        metadata: ExecutorMetadata,
        spec: ExecutorData,
        reserve: bool,
    ) -> Result<Vec<ExecutorReservation>>;

    /// Save the executor metadata. This will overwrite existing metadata for the executor ID
    fn save_executor_metadata(&self, metadata: ExecutorMetadata);

    /// Get executor metadata for the provided executor ID. Returns an error if the executor does not exist
    fn get_executor_metadata(&self, executor_id: &str) -> Result<ExecutorMetadata>;

    /// Save the executor heartbeat
    fn save_executor_heartbeat(&self, heartbeat: ExecutorHeartbeat);

    /// Remove the executor from the cluster
    fn remove_executor(&self, executor_id: &str);

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
#[tonic::async_trait]
pub trait JobState: Send + Sync {
    /// Accept job into  a scheduler's job queue. This should be called when a job is
    /// received by the scheduler but before it is planned and may or may not be saved
    /// in global state
    async fn accept_job(
        &self,
        job_id: &str,
        job_name: &str,
        queued_at: u64,
    ) -> Result<()>;

    /// Submit a new job to the `JobState`. It is assumed that the submitter owns the job.
    /// In local state the job should be save as `JobStatus::Active` and in shared state
    /// it should be saved as `JobStatus::Running` with `scheduler` set to the current scheduler
    async fn submit_job(&self, job_id: String, graph: &ExecutionGraph) -> Result<()>;

    /// Return a `Vec` of all active job IDs in the `JobState`
    async fn get_jobs(&self) -> Result<HashSet<String>>;

    /// Fetch the job status
    async fn get_job_status(&self, job_id: &str) -> Result<Option<JobStatus>>;

    /// Get the `ExecutionGraph` for job. The job may or may not belong to the caller
    /// and should return the `ExecutionGraph` for the given job (if it exists) at the
    /// time this method is called with no guarantees that the graph has not been
    /// subsequently updated by another scheduler.
    async fn get_execution_graph(&self, job_id: &str) -> Result<Option<ExecutionGraph>>;

    /// Persist the current state of an owned job to global state. This should fail
    /// if the job is not owned by the caller.
    async fn save_job(&self, job_id: &str, graph: &ExecutionGraph) -> Result<()>;

    /// Mark a job which has not been submitted as failed. This should be called if a job fails
    /// during planning (and does not yet have an `ExecutionGraph`)
    async fn fail_unscheduled_job(&self, job_id: &str, reason: String) -> Result<()>;

    /// Delete a job from the global state
    async fn remove_job(&self, job_id: &str) -> Result<()>;

    /// Attempt to acquire ownership of the given job. If the job is still in a running state
    /// and is successfully acquired by the caller, return the current `ExecutionGraph`,
    /// otherwise return `None`
    async fn try_acquire_job(&self, job_id: &str) -> Result<Option<ExecutionGraph>>;

    /// Get a stream of all `JobState` events. An event should be published any time that status
    /// of a job changes in state
    async fn job_state_events(&self) -> Result<JobStateEventStream>;

    /// Get the `SessionContext` associated with `session_id`. Returns an error if the
    /// session does not exist
    async fn get_session(&self, session_id: &str) -> Result<Arc<SessionContext>>;

    /// Create a new saved session
    async fn create_session(
        &self,
        config: &BallistaConfig,
    ) -> Result<Arc<SessionContext>>;

    // Update a new saved session. If the session does not exist, a new one will be created
    async fn update_session(
        &self,
        session_id: &str,
        config: &BallistaConfig,
    ) -> Result<Arc<SessionContext>>;
}

pub(crate) fn reserve_slots_bias(
    mut slots: Vec<&mut AvailableTaskSlots>,
    mut n: u32,
) -> Vec<ExecutorReservation> {
    let mut reservations = Vec::with_capacity(n as usize);

    let mut iter = slots.iter_mut();

    while n > 0 {
        if let Some(executor) = iter.next() {
            let take = executor.slots.min(n);
            for _ in 0..take {
                reservations
                    .push(ExecutorReservation::new_free(executor.executor_id.clone()));
            }

            executor.slots -= take;
            n -= take;
        } else {
            break;
        }
    }

    reservations
}

pub(crate) fn reserve_slots_round_robin(
    mut slots: Vec<&mut AvailableTaskSlots>,
    mut n: u32,
) -> Vec<ExecutorReservation> {
    let mut reservations = Vec::with_capacity(n as usize);

    let mut last_updated_idx = 0usize;

    loop {
        let n_before = n;
        for (idx, data) in slots.iter_mut().enumerate() {
            if n == 0 {
                break;
            }

            // Since the vector is sorted in descending order,
            // if finding one executor has not enough slots, the following will have not enough, either
            if data.slots == 0 {
                break;
            }

            reservations.push(ExecutorReservation::new_free(data.executor_id.clone()));
            data.slots -= 1;
            n -= 1;

            if idx >= last_updated_idx {
                last_updated_idx = idx + 1;
            }
        }

        if n_before == n {
            break;
        }
    }

    reservations
}
