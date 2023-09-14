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

use std::time::Duration;

use ballista_core::error::{BallistaError, Result};
use ballista_core::serde::protobuf::{self, MultiTaskDefinition, StopExecutorParams};

use crate::cluster::memory::InMemoryClusterState;
use crate::cluster::{BoundTask, ClusterState, ExecutorSlot};
use crate::config::SchedulerConfig;

use crate::state::execution_graph::RunningTaskInfo;
use ballista_core::serde::protobuf::executor_grpc_client::ExecutorGrpcClient;
use ballista_core::serde::protobuf::{
    executor_status, CancelTasksParams, ExecutorHeartbeat, RemoveJobDataParams,
};
use ballista_core::serde::scheduler::{ExecutorData, ExecutorMetadata};
use ballista_core::utils::{create_grpc_client_connection, get_time_before};
use dashmap::DashMap;
use log::{error, info, warn};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tonic::transport::Channel;

use super::task_manager::JobInfoCache;

type ExecutorClients = Arc<DashMap<String, ExecutorGrpcClient<Channel>>>;

#[derive(Clone)]
pub struct ExecutorManager {
    config: Arc<SchedulerConfig>,
    cluster_state: Arc<InMemoryClusterState>,
    clients: ExecutorClients,
}

impl ExecutorManager {
    pub(crate) fn new(
        cluster_state: Arc<InMemoryClusterState>,
        config: Arc<SchedulerConfig>,
    ) -> Self {
        Self {
            cluster_state,
            config,
            clients: Default::default(),
        }
    }

    pub fn init(&self) -> Result<()> {
        self.cluster_state.init()?;

        Ok(())
    }

    /// Bind the ready to running tasks from [`active_jobs`] with available executors.
    ///
    /// If `executors` is provided, only bind slots from the specified executor IDs
    pub async fn bind_schedulable_tasks(
        &self,
        active_jobs: Arc<HashMap<String, JobInfoCache>>,
    ) -> Result<Vec<BoundTask>> {
        if active_jobs.is_empty() {
            warn!("There's no active jobs for binding tasks");
            return Ok(vec![]);
        }
        let alive_executors = self.get_alive_executors();
        if alive_executors.is_empty() {
            warn!("There's no alive executors for binding tasks");
            return Ok(vec![]);
        }
        self.cluster_state
            .bind_schedulable_tasks(
                self.config.task_distribution,
                active_jobs,
                Some(alive_executors),
            )
            .await
    }

    /// Returned reserved task slots to the pool of available slots. This operation is atomic
    /// so either the entire pool of reserved task slots it returned or none are.
    pub async fn unbind_tasks(&self, executor_slots: Vec<ExecutorSlot>) -> Result<()> {
        self.cluster_state.unbind_tasks(executor_slots).await
    }

    /// Send rpc to Executors to cancel the running tasks
    pub async fn cancel_running_tasks(&self, tasks: Vec<RunningTaskInfo>) -> Result<()> {
        let mut tasks_to_cancel: HashMap<&str, Vec<protobuf::RunningTaskInfo>> =
            Default::default();

        for task_info in &tasks {
            if let Some(infos) = tasks_to_cancel.get_mut(task_info.executor_id.as_str()) {
                infos.push(protobuf::RunningTaskInfo {
                    task_id: task_info.task_id as u32,
                    job_id: task_info.job_id.clone(),
                    stage_id: task_info.stage_id as u32,
                    partition_id: task_info.partition_id as u32,
                })
            } else {
                tasks_to_cancel.insert(
                    task_info.executor_id.as_str(),
                    vec![protobuf::RunningTaskInfo {
                        task_id: task_info.task_id as u32,
                        job_id: task_info.job_id.clone(),
                        stage_id: task_info.stage_id as u32,
                        partition_id: task_info.partition_id as u32,
                    }],
                );
            }
        }

        for (executor_id, infos) in tasks_to_cancel {
            match self.get_client(executor_id).await {
                Ok(mut client) => {
                    client
                        .cancel_tasks(CancelTasksParams { task_infos: infos })
                        .await?;
                }
                Err(e) => {
                    error!(
                        "Failed to get client for executor ID {} to cancel tasks, err: {}",
                        executor_id, e
                    )
                }
            }
        }
        Ok(())
    }

    /// Send rpc to Executors to clean up the job data by delayed clean_up_interval seconds
    pub(crate) fn clean_up_job_data_delayed(
        &self,
        job_id: String,
        clean_up_interval: u64,
    ) {
        if clean_up_interval == 0 {
            info!(
                "The interval is 0 and the clean up for job data {} will not triggered",
                job_id
            );
            return;
        }

        let executor_manager = self.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_secs(clean_up_interval)).await;
            executor_manager.clean_up_job_data_inner(job_id).await;
        });
    }

    /// Send rpc to Executors to clean up the job data in a spawn thread
    pub fn clean_up_job_data(&self, job_id: String) {
        let executor_manager = self.clone();
        tokio::spawn(async move {
            executor_manager.clean_up_job_data_inner(job_id).await;
        });
    }

    /// Send rpc to Executors to clean up the job data
    async fn clean_up_job_data_inner(&self, job_id: String) {
        let alive_executors = self.get_alive_executors();
        for executor in alive_executors {
            let job_id_clone = job_id.to_owned();
            match self.get_client(&executor).await {
                Ok(mut client) => {
                    tokio::spawn(async move {
                        if let Err(err) = client
                            .remove_job_data(RemoveJobDataParams {
                                job_id: job_id_clone,
                            })
                            .await
                        {
                            warn!(
                                "Failed to call remove_job_data on Executor {} due to {:?}",
                                executor, err
                            )
                        }
                    });
                }
                Err(e) => {
                    warn!("Failed to get client for Executor {}: {}", executor, e);
                }
            }
        }
    }

    pub async fn get_client(
        &self,
        executor_id: &str,
    ) -> Result<ExecutorGrpcClient<Channel>> {
        let client = self.clients.get(executor_id).map(|value| value.clone());

        if let Some(client) = client {
            Ok(client)
        } else {
            let executor_metadata = self.get_executor_metadata(executor_id)?;
            let executor_url = format!(
                "http://{}:{}",
                executor_metadata.host, executor_metadata.grpc_port
            );
            let connection = create_grpc_client_connection(executor_url).await?;
            let client = ExecutorGrpcClient::new(connection);

            {
                self.clients.insert(executor_id.to_owned(), client.clone());
            }
            Ok(client)
        }
    }

    /// Get a list of all executors along with the timestamp of their last recorded heartbeat
    pub fn get_executor_state(&self) -> Result<Vec<(ExecutorMetadata, Duration)>> {
        let heartbeat_timestamps: Vec<(String, u64)> = self
            .cluster_state
            .executor_heartbeats()
            .into_iter()
            .map(|(executor_id, heartbeat)| (executor_id, heartbeat.timestamp))
            .collect();

        let mut state: Vec<(ExecutorMetadata, Duration)> = vec![];
        for (executor_id, ts) in heartbeat_timestamps {
            let duration = Duration::from_secs(ts);

            let metadata = self.get_executor_metadata(&executor_id)?;

            state.push((metadata, duration));
        }

        Ok(state)
    }

    pub fn get_executor_metadata(&self, executor_id: &str) -> Result<ExecutorMetadata> {
        self.cluster_state.get_executor_metadata(executor_id)
    }

    pub fn save_executor_metadata(&self, metadata: ExecutorMetadata) -> Result<()> {
        self.cluster_state.save_executor_metadata(metadata);
        Ok(())
    }

    /// Register the executor with the scheduler.
    ///
    /// This will save the executor metadata and the executor data to persistent state.
    ///
    /// It's only used for push-based task scheduling
    pub async fn register_executor(
        &self,
        metadata: ExecutorMetadata,
        specification: ExecutorData,
    ) -> Result<()> {
        info!(
            "registering executor {} with {} task slots",
            metadata.id, specification.total_task_slots
        );

        ExecutorManager::test_connectivity(&metadata).await?;

        self.cluster_state
            .register_executor(metadata, specification)
            .await?;

        Ok(())
    }

    /// Remove the executor within the scheduler.
    pub async fn remove_executor(
        &self,
        executor_id: &str,
        reason: Option<String>,
    ) -> Result<()> {
        info!("Removing executor {}: {:?}", executor_id, reason);
        self.cluster_state.remove_executor(executor_id).await
    }

    pub async fn stop_executor(&self, executor_id: &str, stop_reason: String) {
        let executor_id = executor_id.to_string();
        match self.get_client(&executor_id).await {
            Ok(mut client) => {
                tokio::task::spawn(async move {
                    match client
                        .stop_executor(StopExecutorParams {
                            executor_id: executor_id.to_string(),
                            reason: stop_reason,
                            force: true,
                        })
                        .await
                    {
                        Err(error) => {
                            warn!("Failed to send stop_executor rpc due to, {}", error);
                        }
                        Ok(_value) => {}
                    }
                });
            }
            Err(_) => {
                warn!(
                    "Executor is already dead, failed to connect to Executor {}",
                    executor_id
                );
            }
        }
    }

    pub async fn launch_multi_task(
        &self,
        executor_id: &str,
        multi_tasks: Vec<MultiTaskDefinition>,
        scheduler_id: String,
    ) -> Result<()> {
        let mut client = self.get_client(executor_id).await?;
        client
            .launch_multi_task(protobuf::LaunchMultiTaskParams {
                multi_tasks,
                scheduler_id,
            })
            .await
            .map_err(|e| {
                BallistaError::Internal(format!(
                    "Failed to connect to executor {}: {:?}",
                    executor_id, e
                ))
            })?;

        Ok(())
    }

    pub(crate) fn save_executor_heartbeat(
        &self,
        heartbeat: ExecutorHeartbeat,
    ) -> Result<()> {
        self.cluster_state.save_executor_heartbeat(heartbeat);

        Ok(())
    }

    pub(crate) fn is_dead_executor(&self, executor_id: &str) -> bool {
        self.cluster_state
            .get_executor_heartbeat(executor_id)
            .map_or(true, |heartbeat| {
                matches!(
                    heartbeat.status,
                    Some(ballista_core::serde::generated::ballista::ExecutorStatus {
                        status: Some(executor_status::Status::Dead(_))
                    })
                )
            })
    }

    /// Retrieve the set of all executor IDs where the executor has been observed in the last
    /// `last_seen_ts_threshold` seconds.
    pub(crate) fn get_alive_executors(&self) -> HashSet<String> {
        let last_seen_ts_threshold =
            get_time_before(self.config.executor_timeout_seconds);
        self.cluster_state
            .executor_heartbeats()
            .iter()
            .filter_map(|(exec, heartbeat)| {
                let active = matches!(
                    heartbeat
                        .status
                        .as_ref()
                        .and_then(|status| status.status.as_ref()),
                    Some(executor_status::Status::Active(_))
                );
                let live = heartbeat.timestamp > last_seen_ts_threshold;

                (active && live).then(|| exec.clone())
            })
            .collect()
    }

    /// Return a list of expired executors
    pub(crate) fn get_expired_executors(&self) -> Vec<ExecutorHeartbeat> {
        // Threshold for last heartbeat from Active executor before marking dead
        let last_seen_threshold = get_time_before(self.config.executor_timeout_seconds);

        // Threshold for last heartbeat for Fenced executor before marking dead
        let termination_wait_threshold =
            get_time_before(self.config.executor_termination_grace_period);

        self.cluster_state
            .executor_heartbeats()
            .iter()
            .filter_map(|(_exec, heartbeat)| {
                let terminating = matches!(
                    heartbeat
                        .status
                        .as_ref()
                        .and_then(|status| status.status.as_ref()),
                    Some(executor_status::Status::Terminating(_))
                );

                let grace_period_expired =
                    heartbeat.timestamp <= termination_wait_threshold;

                let expired = heartbeat.timestamp <= last_seen_threshold;

                ((terminating && grace_period_expired) || expired)
                    .then(|| heartbeat.clone())
            })
            .collect::<Vec<_>>()
    }

    #[cfg(not(test))]
    async fn test_connectivity(metadata: &ExecutorMetadata) -> Result<()> {
        let executor_url = format!("http://{}:{}", metadata.host, metadata.grpc_port);
        log::debug!("Connecting to executor {:?}", executor_url);
        let _ = protobuf::executor_grpc_client::ExecutorGrpcClient::connect(executor_url)
            .await
            .map_err(|e| {
                BallistaError::Internal(format!(
                    "Failed to register executor at {}:{}, could not connect: {:?}",
                    metadata.host, metadata.grpc_port, e
                ))
            })?;
        Ok(())
    }

    #[cfg(test)]
    async fn test_connectivity(_metadata: &ExecutorMetadata) -> Result<()> {
        Ok(())
    }
}
