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

use ballista_core::config::{BallistaConfig, BALLISTA_JOB_NAME};
use std::convert::TryInto;

use ballista_core::serde::protobuf::scheduler_grpc_server::SchedulerGrpc;
use ballista_core::serde::protobuf::{
    CancelJobParams, CancelJobResult, CleanJobDataParams, CleanJobDataResult,
    CreateSessionParams, CreateSessionResult, ExecuteQueryParams, ExecuteQueryResult,
    ExecutorHeartbeat, ExecutorStoppedParams, ExecutorStoppedResult,
    GetFileMetadataParams, GetFileMetadataResult, GetJobStatusParams, GetJobStatusResult,
    HeartBeatParams, HeartBeatResult, RegisterExecutorParams, RegisterExecutorResult,
    UpdateTaskStatusParams, UpdateTaskStatusResult,
};
use ballista_core::serde::scheduler::ExecutorMetadata;

use datafusion::datasource::file_format::parquet::ParquetFormat;
use datafusion::datasource::file_format::FileFormat;
use datafusion_proto::logical_plan::AsLogicalPlan;
use datafusion_proto::physical_plan::AsExecutionPlan;
use futures::TryStreamExt;
use log::{debug, error, info, trace, warn};
use object_store::{local::LocalFileSystem, path::Path, ObjectStore};

use std::ops::Deref;
use std::sync::Arc;

use crate::scheduler_server::event::QueryStageSchedulerEvent;
use datafusion::prelude::SessionContext;
use std::time::{SystemTime, UNIX_EPOCH};
use tonic::{Request, Response, Status};

use crate::scheduler_server::SchedulerServer;

#[tonic::async_trait]
impl<T: 'static + AsLogicalPlan, U: 'static + AsExecutionPlan> SchedulerGrpc
    for SchedulerServer<T, U>
{
    async fn register_executor(
        &self,
        request: Request<RegisterExecutorParams>,
    ) -> Result<Response<RegisterExecutorResult>, Status> {
        let remote_addr = request.remote_addr();
        if let RegisterExecutorParams {
            metadata: Some(metadata),
        } = request.into_inner()
        {
            info!("Received register executor request for {:?}", metadata);
            let metadata = ExecutorMetadata {
                id: metadata.id,
                host: metadata
                    .host
                    .unwrap_or_else(|| remote_addr.unwrap().ip().to_string()),
                port: metadata.port as u16,
                grpc_port: metadata.grpc_port as u16,
                specification: metadata.specification.unwrap().into(),
            };

            self.do_register_executor(metadata).await.map_err(|e| {
                let msg = format!("Fail to do executor registration due to: {e}");
                error!("{}", msg);
                Status::internal(msg)
            })?;

            Ok(Response::new(RegisterExecutorResult { success: true }))
        } else {
            warn!("Received invalid register executor request");
            Err(Status::invalid_argument("Missing metadata in request"))
        }
    }

    async fn heart_beat_from_executor(
        &self,
        request: Request<HeartBeatParams>,
    ) -> Result<Response<HeartBeatResult>, Status> {
        let remote_addr = request.remote_addr();
        let HeartBeatParams {
            executor_id,
            metrics,
            status,
            metadata,
        } = request.into_inner();
        debug!("Received heart beat request for {:?}", executor_id);

        // If not registered, do registration first before saving heart beat
        if let Err(e) = self
            .state
            .executor_manager
            .get_executor_metadata(&executor_id)
        {
            warn!("Fail to get executor metadata: {}", e);
            if let Some(metadata) = metadata {
                let metadata = ExecutorMetadata {
                    id: metadata.id,
                    host: metadata
                        .host
                        .unwrap_or_else(|| remote_addr.unwrap().ip().to_string()),
                    port: metadata.port as u16,
                    grpc_port: metadata.grpc_port as u16,
                    specification: metadata.specification.unwrap().into(),
                };

                self.do_register_executor(metadata).await.map_err(|e| {
                    let msg = format!("Fail to do executor registration due to: {e}");
                    error!("{}", msg);
                    Status::internal(msg)
                })?;
            } else {
                return Err(Status::invalid_argument(format!(
                    "The registration spec for executor {executor_id} is not included"
                )));
            }
        }

        let executor_heartbeat = ExecutorHeartbeat {
            executor_id,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("Time went backwards")
                .as_secs(),
            metrics,
            status,
        };

        self.state
            .executor_manager
            .save_executor_heartbeat(executor_heartbeat)
            .map_err(|e| {
                let msg = format!("Could not save executor heartbeat: {e}");
                error!("{}", msg);
                Status::internal(msg)
            })?;
        Ok(Response::new(HeartBeatResult { reregister: false }))
    }

    async fn update_task_status(
        &self,
        request: Request<UpdateTaskStatusParams>,
    ) -> Result<Response<UpdateTaskStatusResult>, Status> {
        let UpdateTaskStatusParams {
            executor_id,
            task_status,
        } = request.into_inner();

        debug!(
            "Received task status update request for executor {:?}",
            executor_id
        );

        self.update_task_status(&executor_id, task_status)
            .await
            .map_err(|e| {
                let msg = format!(
                    "Fail to update tasks status from executor {:?} due to {:?}",
                    &executor_id, e
                );
                error!("{}", msg);
                Status::internal(msg)
            })?;

        Ok(Response::new(UpdateTaskStatusResult { success: true }))
    }

    async fn get_file_metadata(
        &self,
        request: Request<GetFileMetadataParams>,
    ) -> Result<Response<GetFileMetadataResult>, Status> {
        // Here, we use the default config, since we don't know the session id
        let session_ctx = SessionContext::new();
        let state = session_ctx.state();

        // TODO support multiple object stores
        let obj_store: Arc<dyn ObjectStore> = Arc::new(LocalFileSystem::new());
        // TODO shouldn't this take a ListingOption object as input?

        let GetFileMetadataParams { path, file_type } = request.into_inner();
        let file_format: Arc<dyn FileFormat> = match file_type.as_str() {
            "parquet" => Ok(Arc::new(ParquetFormat::default())),
            // TODO implement for CSV
            _ => Err(tonic::Status::unimplemented(
                "get_file_metadata unsupported file type",
            )),
        }?;

        let path = Path::from(path.as_str());
        let file_metas: Vec<_> = obj_store
            .list(Some(&path))
            .await
            .map_err(|e| {
                let msg = format!("Error listing files: {e}");
                error!("{}", msg);
                tonic::Status::internal(msg)
            })?
            .try_collect()
            .await
            .map_err(|e| {
                let msg = format!("Error listing files: {e}");
                error!("{}", msg);
                tonic::Status::internal(msg)
            })?;

        let schema = file_format
            .infer_schema(&state, &obj_store, &file_metas)
            .await
            .map_err(|e| {
                let msg = format!("Error inferring schema: {e}");
                error!("{}", msg);
                tonic::Status::internal(msg)
            })?;

        Ok(Response::new(GetFileMetadataResult {
            schema: Some(schema.as_ref().try_into().map_err(|e| {
                let msg = format!("Error inferring schema: {e}");
                error!("{}", msg);
                tonic::Status::internal(msg)
            })?),
        }))
    }

    async fn create_session(
        &self,
        request: Request<CreateSessionParams>,
    ) -> Result<Response<CreateSessionResult>, Status> {
        let settings = request.into_inner();
        // parse config for new session
        let mut config_builder = BallistaConfig::builder();
        for kv_pair in &settings.settings {
            config_builder = config_builder.set(&kv_pair.key, &kv_pair.value);
        }
        let config = config_builder.build().map_err(|e| {
            let msg = format!("Could not parse configs: {e}");
            error!("{}", msg);
            Status::internal(msg)
        })?;
        let session =
            self.state
                .session_manager
                .create_session(&config)
                .map_err(|e| {
                    Status::internal(format!(
                        "Failed to create new SessionContext: {e:?}"
                    ))
                })?;

        Ok(Response::new(CreateSessionResult {
            session_id: session.session_id(),
        }))
    }

    async fn execute_query(
        &self,
        request: Request<ExecuteQueryParams>,
    ) -> Result<Response<ExecuteQueryResult>, Status> {
        let query_params = request.into_inner();
        let ExecuteQueryParams {
            logical_plan: query,
            settings,
            session_id,
        } = query_params;

        // parse config
        let mut config_builder = BallistaConfig::builder();
        for kv_pair in &settings {
            config_builder = config_builder.set(&kv_pair.key, &kv_pair.value);
        }
        let config = config_builder.build().map_err(|e| {
            let msg = format!("Could not parse configs: {e}");
            error!("{}", msg);
            Status::internal(msg)
        })?;

        let session_ctx = self
            .state
            .session_manager
            .get_session(&session_id)
            .map_err(|e| {
                Status::internal(format!(
                    "Failed to load SessionContext for session ID {session_id}: {e:?}"
                ))
            })?;

        let plan = {
            T::try_decode(query.as_slice())
                .and_then(|m| {
                    m.try_into_logical_plan(
                        session_ctx.deref(),
                        self.state.codec.logical_extension_codec(),
                    )
                })
                .map_err(|e| {
                    let msg = format!("Could not parse logical plan protobuf: {e}");
                    error!("{}", msg);
                    Status::internal(msg)
                })?
        };

        debug!("Received plan for execution: {:?}", plan);

        let job_id = self.state.task_manager.generate_job_id();
        let job_name = config
            .settings()
            .get(BALLISTA_JOB_NAME)
            .cloned()
            .unwrap_or_default();

        self.submit_job(&job_id, &job_name, session_ctx, &plan)
            .await
            .map_err(|e| {
                let msg = format!("Failed to send JobQueued event for {job_id}: {e:?}");
                error!("{}", msg);

                Status::internal(msg)
            })?;

        Ok(Response::new(ExecuteQueryResult { job_id, session_id }))
    }

    async fn get_job_status(
        &self,
        request: Request<GetJobStatusParams>,
    ) -> Result<Response<GetJobStatusResult>, Status> {
        let job_id = request.into_inner().job_id;
        trace!("Received get_job_status request for job {}", job_id);
        match self.state.task_manager.get_job_status(&job_id).await {
            Ok(status) => Ok(Response::new(GetJobStatusResult { status })),
            Err(e) => {
                let msg = format!("Error getting status for job {job_id}: {e:?}");
                error!("{}", msg);
                Err(Status::internal(msg))
            }
        }
    }

    async fn executor_stopped(
        &self,
        request: Request<ExecutorStoppedParams>,
    ) -> Result<Response<ExecutorStoppedResult>, Status> {
        let ExecutorStoppedParams {
            executor_id,
            reason,
        } = request.into_inner();
        info!(
            "Received executor stopped request from Executor {} with reason '{}'",
            executor_id, reason
        );

        let executor_manager = self.state.executor_manager.clone();
        let event_sender = self.query_stage_event_loop.get_sender().map_err(|e| {
            let msg = format!("Get query stage event loop error due to {e:?}");
            error!("{}", msg);
            Status::internal(msg)
        })?;

        Self::remove_executor(
            executor_manager,
            event_sender,
            &executor_id,
            Some(reason),
            self.executor_termination_grace_period,
        );

        Ok(Response::new(ExecutorStoppedResult {}))
    }

    async fn cancel_job(
        &self,
        request: Request<CancelJobParams>,
    ) -> Result<Response<CancelJobResult>, Status> {
        let job_id = request.into_inner().job_id;
        info!("Received cancellation request for job {}", job_id);

        self.query_stage_event_loop
            .get_sender()
            .map_err(|e| {
                let msg = format!("Get query stage event loop error due to {e:?}");
                error!("{}", msg);
                Status::internal(msg)
            })?
            .post_event(QueryStageSchedulerEvent::JobCancel(job_id))
            .await
            .map_err(|e| {
                let msg = format!("Post to query stage event loop error due to {e:?}");
                error!("{}", msg);
                Status::internal(msg)
            })?;
        Ok(Response::new(CancelJobResult { cancelled: true }))
    }

    async fn clean_job_data(
        &self,
        request: Request<CleanJobDataParams>,
    ) -> Result<Response<CleanJobDataResult>, Status> {
        let job_id = request.into_inner().job_id;
        info!("Received clean data request for job {}", job_id);

        self.query_stage_event_loop
            .get_sender()
            .map_err(|e| {
                let msg = format!("Get query stage event loop error due to {e:?}");
                error!("{}", msg);
                Status::internal(msg)
            })?
            .post_event(QueryStageSchedulerEvent::JobDataClean(job_id))
            .await
            .map_err(|e| {
                let msg = format!("Post to query stage event loop error due to {e:?}");
                error!("{}", msg);
                Status::internal(msg)
            })?;
        Ok(Response::new(CleanJobDataResult {}))
    }
}
