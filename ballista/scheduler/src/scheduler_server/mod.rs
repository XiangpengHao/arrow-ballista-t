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

use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crate::event_loop::{EventLoop, EventSender};
use ballista_core::error::Result;
use ballista_core::serde::protobuf::TaskStatus;
use ballista_core::serde::BallistaCodec;

use ballista_core::utils::RemoteMemoryMode;
use datafusion::execution::context::SessionState;
use datafusion::logical_expr::LogicalPlan;
use datafusion::prelude::{SessionConfig, SessionContext};
use datafusion_proto::logical_plan::AsLogicalPlan;
use datafusion_proto::physical_plan::AsExecutionPlan;

use crate::cluster::BallistaCluster;
use crate::config::SchedulerConfig;
use crate::metrics::SchedulerMetricsCollector;
use ballista_core::serde::scheduler::{ExecutorData, ExecutorMetadata};
use log::{error, warn};

use crate::scheduler_server::event::QueryStageSchedulerEvent;
use crate::scheduler_server::query_stage_scheduler::QueryStageScheduler;

use crate::state::executor_manager::ExecutorManager;

#[cfg(test)]
use crate::state::task_manager::TaskLauncher;

use crate::state::SchedulerState;

pub mod event;
mod grpc;
pub(crate) mod query_stage_scheduler;

pub(crate) type SessionBuilder = fn(SessionConfig) -> SessionState;

#[derive(Clone)]
pub struct SchedulerServer<T: 'static + AsLogicalPlan, U: 'static + AsExecutionPlan> {
    pub scheduler_name: String,
    pub start_time: u128,
    pub state: Arc<SchedulerState<T, U>>,
    pub(crate) query_stage_event_loop: EventLoop<T, U>,
    query_stage_scheduler: Arc<QueryStageScheduler<T, U>>,
    executor_termination_grace_period: u64,
}

impl<T: 'static + AsLogicalPlan, U: 'static + AsExecutionPlan> SchedulerServer<T, U> {
    pub fn new(
        scheduler_name: String,
        cluster: BallistaCluster,
        codec: BallistaCodec<T, U>,
        config: Arc<SchedulerConfig>,
        metrics_collector: Arc<dyn SchedulerMetricsCollector>,
    ) -> Self {
        let state = Arc::new(SchedulerState::new(
            cluster,
            codec,
            scheduler_name.clone(),
            config.clone(),
        ));
        let query_stage_scheduler = Arc::new(QueryStageScheduler::new(
            state.clone(),
            metrics_collector,
            config.clone(),
        ));
        let query_stage_event_loop = EventLoop::new(
            "query_stage".to_owned(),
            config.event_loop_buffer_size as usize,
            query_stage_scheduler.clone(),
        );

        Self {
            scheduler_name,
            start_time: timestamp_millis() as u128,
            state,
            query_stage_event_loop,
            query_stage_scheduler,
            executor_termination_grace_period: config.executor_termination_grace_period,
        }
    }

    #[cfg(test)]
    pub fn new_with_task_launcher(
        scheduler_name: String,
        cluster: BallistaCluster,
        codec: BallistaCodec<T, U>,
        config: Arc<SchedulerConfig>,
        metrics_collector: Arc<dyn SchedulerMetricsCollector>,
        task_launcher: Arc<dyn TaskLauncher>,
    ) -> Self {
        let state = Arc::new(SchedulerState::new_with_task_launcher(
            cluster,
            codec,
            scheduler_name.clone(),
            config.clone(),
            task_launcher,
        ));
        let query_stage_scheduler = Arc::new(QueryStageScheduler::new(
            state.clone(),
            metrics_collector,
            config.clone(),
        ));
        let query_stage_event_loop = EventLoop::new(
            "query_stage".to_owned(),
            config.event_loop_buffer_size as usize,
            query_stage_scheduler.clone(),
        );

        Self {
            scheduler_name,
            start_time: timestamp_millis() as u128,
            state,
            query_stage_event_loop,
            query_stage_scheduler,
            executor_termination_grace_period: config.executor_termination_grace_period,
        }
    }

    pub fn init(&mut self) -> Result<()> {
        self.state.init()?;
        self.query_stage_event_loop.start()?;
        self.expire_dead_executors()?;

        Ok(())
    }

    pub fn pending_job_number(&self) -> usize {
        self.state.task_manager.pending_job_number()
    }

    pub fn running_job_number(&self) -> usize {
        self.state.task_manager.running_job_number()
    }

    pub(crate) fn metrics_collector(&self) -> &dyn SchedulerMetricsCollector {
        self.query_stage_scheduler.metrics_collector()
    }

    pub(crate) async fn submit_job(
        &self,
        job_id: &str,
        job_name: &str,
        ctx: Arc<SessionContext>,
        plan: &LogicalPlan,
        sql: String,
        mode: RemoteMemoryMode,
    ) -> Result<()> {
        self.state.task_manager.register_sql(job_id, sql);
        self.query_stage_event_loop
            .get_sender()?
            .post_event(QueryStageSchedulerEvent::JobQueued {
                job_id: job_id.to_owned(),
                job_name: job_name.to_owned(),
                session_ctx: ctx,
                plan: Box::new(plan.clone()),
                queued_at: timestamp_millis(),
                mode,
            })
            .await
    }

    /// It just send task status update event to the channel,
    /// and will not guarantee the event processing completed after return
    pub(crate) async fn update_task_status(
        &self,
        executor_id: &str,
        tasks_status: Vec<TaskStatus>,
    ) -> Result<()> {
        // We might receive buggy task updates from dead executors.
        if self.state.executor_manager.is_dead_executor(executor_id) {
            let error_msg = format!(
                "Receive buggy tasks status from dead Executor {executor_id}, task status update ignored."
            );
            warn!("{}", error_msg);
            return Ok(());
        }
        self.query_stage_event_loop
            .get_sender()?
            .post_event(QueryStageSchedulerEvent::TaskUpdating(
                executor_id.to_owned(),
                tasks_status,
            ))
            .await
    }

    pub(crate) async fn revive_offers(&self) -> Result<()> {
        self.query_stage_event_loop
            .get_sender()?
            .post_event(QueryStageSchedulerEvent::ReviveOffers)
            .await
    }

    /// Spawn an async task which periodically check the active executors' status and
    /// expire the dead executors
    fn expire_dead_executors(&self) -> Result<()> {
        let state = self.state.clone();
        let event_sender = self.query_stage_event_loop.get_sender()?;
        tokio::task::spawn(async move {
            loop {
                let expired_executors = state.executor_manager.get_expired_executors();
                for expired in expired_executors {
                    let executor_id = expired.executor_id.clone();

                    let sender_clone = event_sender.clone();

                    let terminating = matches!(
                        expired
                            .status
                            .as_ref()
                            .and_then(|status| status.status.as_ref()),
                        Some(ballista_core::serde::protobuf::executor_status::Status::Terminating(_))
                    );

                    let stop_reason = if terminating {
                        format!(
                        "TERMINATING executor {executor_id} heartbeat timed out after {}s", state.config.executor_termination_grace_period,
                    )
                    } else {
                        format!(
                            "ACTIVE executor {executor_id} heartbeat timed out after {}s",
                            state.config.executor_timeout_seconds,
                        )
                    };

                    warn!("{stop_reason}");

                    // If executor is expired, remove it immediately
                    Self::remove_executor(
                        state.executor_manager.clone(),
                        sender_clone,
                        &executor_id,
                        Some(stop_reason.clone()),
                        0,
                    );

                    // If executor is not already terminating then stop it. If it is terminating then it should already be shutting
                    // down and we do not need to do anything here.
                    if !terminating {
                        state
                            .executor_manager
                            .stop_executor(&executor_id, stop_reason)
                            .await;
                    }
                }
                tokio::time::sleep(Duration::from_secs(
                    state.config.expire_dead_executor_interval_seconds,
                ))
                .await;
            }
        });
        Ok(())
    }

    pub(crate) fn remove_executor(
        executor_manager: ExecutorManager,
        event_sender: EventSender<QueryStageSchedulerEvent>,
        executor_id: &str,
        reason: Option<String>,
        wait_secs: u64,
    ) {
        let executor_id = executor_id.to_owned();
        tokio::spawn(async move {
            // Wait for `wait_secs` before removing executor
            tokio::time::sleep(Duration::from_secs(wait_secs)).await;

            // Update the executor manager immediately here
            if let Err(e) = executor_manager
                .remove_executor(&executor_id, reason.clone())
                .await
            {
                error!("error removing executor {executor_id}: {e:?}");
            }

            if let Err(e) = event_sender
                .post_event(QueryStageSchedulerEvent::ExecutorLost(executor_id, reason))
                .await
            {
                error!("error sending ExecutorLost event: {e:?}");
            }
        });
    }

    async fn do_register_executor(&self, metadata: ExecutorMetadata) -> Result<()> {
        let executor_data = ExecutorData {
            executor_id: metadata.id.clone(),
            total_task_slots: metadata.specification.task_slots,
            available_task_slots: metadata.specification.task_slots,
        };

        // Save the executor to state
        self.state
            .executor_manager
            .register_executor(metadata, executor_data)
            .await?;

        // If we are using push-based scheduling then reserve this executors slots and send
        // them for scheduling tasks.
        self.revive_offers().await?;

        Ok(())
    }
}

pub fn timestamp_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("Time went backwards")
        .as_secs()
}

pub fn timestamp_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("Time went backwards")
        .as_millis() as u64
}
