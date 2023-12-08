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

use std::any::Any;
use std::collections::HashMap;
use std::ffi::CString;
use std::fmt::Debug;
use std::fs::File;
use std::pin::Pin;
use std::result;
use std::sync::Arc;
use std::task::{Context, Poll};

use crate::client::BallistaClient;
use crate::serde::scheduler::{PartitionLocation, PartitionStats};
use crate::utils::RemoteMemoryMode;

use datafusion::arrow::datatypes::SchemaRef;
use datafusion::arrow::error::ArrowError;
use datafusion::arrow::ipc::reader::FileReader;
use datafusion::arrow::record_batch::{RecordBatch, RecordBatchReader};
use datafusion::common::stats::Precision;

use datafusion::error::Result;
use datafusion::physical_plan::expressions::PhysicalSortExpr;
use datafusion::physical_plan::metrics::{ExecutionPlanMetricsSet, MetricsSet};
use datafusion::physical_plan::{
    ColumnStatistics, DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning,
    RecordBatchStream, SendableRecordBatchStream, Statistics,
};
use futures::{Stream, StreamExt, TryStreamExt};

use crate::error::BallistaError;
use datafusion::execution::context::TaskContext;
use datafusion::physical_plan::common::AbortOnDropMany;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use itertools::Itertools;
use log::{error, info};
use rand::prelude::SliceRandom;
use rand::thread_rng;
use tokio::sync::{mpsc, Semaphore};
use tokio::task::JoinHandle;
use tokio_stream::wrappers::ReceiverStream;

use super::rm_writer::{MemoryReader, NoBufReader};

static JOIN_STATE: once_cell::sync::Lazy<std::sync::Mutex<HashMap<String, String>>> =
    once_cell::sync::Lazy::new(|| std::sync::Mutex::new(HashMap::new()));

/// ShuffleReaderExec reads partitions that have already been materialized by a ShuffleWriterExec
/// being executed by an executor
#[derive(Debug, Clone)]
pub struct ShuffleReaderExec {
    /// The query stage id to read from
    pub stage_id: usize,
    /// Each partition of a shuffle can read data from multiple locations
    pub partition: Vec<Vec<PartitionLocation>>,
    pub schema: SchemaRef,
    /// Execution metrics
    metrics: ExecutionPlanMetricsSet,

    remote_mode: RemoteMemoryMode,
}

impl ShuffleReaderExec {
    /// Create a new ShuffleReaderExec
    pub fn try_new(
        stage_id: usize,
        partition: Vec<Vec<PartitionLocation>>,
        schema: SchemaRef,
        remote_mode: RemoteMemoryMode,
    ) -> Result<Self> {
        Ok(Self {
            stage_id,
            schema,
            partition,
            metrics: ExecutionPlanMetricsSet::new(),
            remote_mode,
        })
    }

    pub fn remote_memory_mode(&self) -> RemoteMemoryMode {
        self.remote_mode
    }
}

impl ExecutionPlan for ShuffleReaderExec {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn output_partitioning(&self) -> Partitioning {
        // TODO partitioning may be known and could be populated here
        // see https://github.com/apache/arrow-datafusion/issues/758
        Partitioning::UnknownPartitioning(self.partition.len())
    }

    fn output_ordering(&self) -> Option<&[PhysicalSortExpr]> {
        None
    }

    fn children(&self) -> Vec<Arc<dyn ExecutionPlan>> {
        vec![]
    }

    fn with_new_children(
        self: Arc<Self>,
        _children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(ShuffleReaderExec::try_new(
            self.stage_id,
            self.partition.clone(),
            self.schema.clone(),
            self.remote_mode,
        )?))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let task_id = context.task_id().unwrap_or_else(|| partition.to_string());
        info!("ShuffleReaderExec::execute({})", task_id);

        log::warn!("ShuffleReader: partition:{:?}", self.partition);

        // TODO make the maximum size configurable, or make it depends on global memory control
        let max_request_num = 50usize;
        let mut partition_locations = HashMap::new();
        for p in &self.partition[partition] {
            partition_locations
                .entry(p.executor_meta.id.clone())
                .or_insert_with(Vec::new)
                .push(p.clone());
        }
        // Sort partitions for evenly send fetching partition requests to avoid hot executors within one task
        let mut partition_locations: Vec<PartitionLocation> = partition_locations
            .into_values()
            .flat_map(|ps| ps.into_iter().enumerate())
            .sorted_by(|(p1_idx, _), (p2_idx, _)| Ord::cmp(p1_idx, p2_idx))
            .map(|(_, p)| p)
            .collect();
        // Shuffle partitions for evenly send fetching partition requests to avoid hot executors within multiple tasks
        partition_locations.shuffle(&mut thread_rng());

        if partition_locations.len() > 0
            && matches!(self.remote_mode, RemoteMemoryMode::JoinOnRemote)
        {
            assert_eq!(partition_locations.len(), 1);
            let mut join_state = JOIN_STATE.lock().unwrap();
            let loc = partition_locations.first().unwrap();
            let has_ht = loc.partition_stats.ht_items.unwrap() > 0;
            if has_ht {
                let rt = join_state
                    .insert("shuffle_partition_path".to_string(), loc.path.clone());
                log::info!("JoinOnRemote partition path: {}", loc.path);

                if rt.is_some() {
                    panic!("JoinOnRemote partition path is not empty {}", rt.unwrap());
                }
            }
        }

        let response_receiver =
            send_fetch_partitions(partition_locations, max_request_num, self.remote_mode);

        let result = RecordBatchStreamAdapter::new(
            Arc::new(self.schema.as_ref().clone()),
            response_receiver.try_flatten(),
        );
        Ok(Box::pin(result))
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }

    fn statistics(&self) -> Result<Statistics> {
        Ok(stats_for_partitions(
            self.schema.fields().len(),
            self.partition
                .iter()
                .flatten()
                .map(|loc| loc.partition_stats),
        ))
    }
}

impl DisplayAs for ShuffleReaderExec {
    fn fmt_as(
        &self,
        t: DisplayFormatType,
        f: &mut std::fmt::Formatter,
    ) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(
                    f,
                    "ShuffleReaderExec: part={}, mode={}",
                    self.partition.len(),
                    self.remote_memory_mode()
                )
            }
        }
    }
}

fn stats_for_partitions(
    num_fields: usize,
    partition_stats: impl Iterator<Item = PartitionStats>,
) -> Statistics {
    // TODO stats: add column statistics to PartitionStats
    let (num_rows, total_byte_size) =
        partition_stats.fold((Some(0), Some(0)), |(num_rows, total_byte_size), part| {
            // if any statistic is unkown it makes the entire statistic unkown
            let num_rows = num_rows.zip(part.num_rows).map(|(a, b)| a + b as usize);
            let total_byte_size = total_byte_size
                .zip(part.num_bytes)
                .map(|(a, b)| a + b as usize);
            (num_rows, total_byte_size)
        });

    Statistics {
        num_rows: num_rows.map(Precision::Exact).unwrap_or(Precision::Absent),
        total_byte_size: total_byte_size
            .map(Precision::Exact)
            .unwrap_or(Precision::Absent),
        column_statistics: vec![ColumnStatistics::new_unknown(); num_fields],
    }
}

pub(super) struct LocalShuffleStream<
    I: Iterator<Item = Result<RecordBatch, ArrowError>> + RecordBatchReader,
> {
    reader: I,
}

impl<I: Iterator<Item = Result<RecordBatch, ArrowError>> + RecordBatchReader>
    LocalShuffleStream<I>
{
    pub fn new(reader: I) -> Self {
        LocalShuffleStream { reader }
    }
}

impl<I: Iterator<Item = Result<RecordBatch, ArrowError>> + RecordBatchReader + Unpin>
    Stream for LocalShuffleStream<I>
{
    type Item = Result<RecordBatch>;

    fn poll_next(
        mut self: Pin<&mut Self>,
        _: &mut Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        let mut this = self.as_mut();
        if let Some(batch) = this.reader.next() {
            return Poll::Ready(Some(batch.map_err(|e| e.into())));
        }
        Poll::Ready(None)
    }
}
impl<I: Iterator<Item = Result<RecordBatch, ArrowError>> + RecordBatchReader + Unpin>
    RecordBatchStream for LocalShuffleStream<I>
{
    fn schema(&self) -> SchemaRef {
        self.reader.schema().clone()
    }
}

/// Adapter for a tokio ReceiverStream that implements the SendableRecordBatchStream interface
pub(super) struct AbortableReceiverStream {
    inner: ReceiverStream<result::Result<SendableRecordBatchStream, BallistaError>>,

    #[allow(dead_code)]
    drop_helper: AbortOnDropMany<()>,
}

impl AbortableReceiverStream {
    /// Construct a new SendableRecordBatchReceiverStream which will send batches of the specified schema from inner
    pub fn create(
        rx: tokio::sync::mpsc::Receiver<
            result::Result<SendableRecordBatchStream, BallistaError>,
        >,
        join_handles: Vec<JoinHandle<()>>,
    ) -> AbortableReceiverStream {
        let inner = ReceiverStream::new(rx);
        Self {
            inner,
            drop_helper: AbortOnDropMany(join_handles),
        }
    }
}

impl Stream for AbortableReceiverStream {
    type Item = result::Result<SendableRecordBatchStream, ArrowError>;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        self.inner
            .poll_next_unpin(cx)
            .map_err(|e| ArrowError::ExternalError(Box::new(e)))
    }
}
fn send_fetch_partitions(
    partition_locations: Vec<PartitionLocation>,
    max_request_num: usize,
    mode: RemoteMemoryMode,
) -> AbortableReceiverStream {
    match mode {
        RemoteMemoryMode::DoNotUse => {
            send_fetch_partitions_from_disk(partition_locations, max_request_num)
        }
        RemoteMemoryMode::FileBasedShuffle => {
            send_fetch_partitions_from_remote_memory_file(
                partition_locations,
                max_request_num,
            )
        }
        RemoteMemoryMode::MemoryBasedShuffle | RemoteMemoryMode::JoinOnRemote => {
            send_fetch_partitions_from_remote_memory_byte(
                partition_locations,
                max_request_num,
            )
        }
    }
}

fn send_fetch_partitions_from_remote_memory_file(
    partition_locations: Vec<PartitionLocation>,
    max_request_num: usize,
) -> AbortableReceiverStream {
    let (response_sender, response_receiver) = mpsc::channel(max_request_num);
    let mut join_handles = vec![];

    info!(
        "fetching {} partitions from remote memory",
        partition_locations.len()
    );

    // keep local shuffle files reading in serial order for memory control.
    let response_sender_c = response_sender.clone();
    let join_handle = tokio::spawn(async move {
        for p in partition_locations {
            let r = fetch_partition_remote_memory_file(&p).await;
            if let Err(e) = response_sender_c.send(r).await {
                error!("Fail to send response event to the channel due to {}", e);
            }
        }
    });
    join_handles.push(join_handle);

    AbortableReceiverStream::create(response_receiver, join_handles)
}

fn send_fetch_partitions_from_remote_memory_byte(
    partition_locations: Vec<PartitionLocation>,
    max_request_num: usize,
) -> AbortableReceiverStream {
    let (response_sender, response_receiver) = mpsc::channel(max_request_num);
    let mut join_handles = vec![];

    info!(
        "fetching {} partitions from remote memory",
        partition_locations.len()
    );

    // keep local shuffle files reading in serial order for memory control.
    let response_sender_c = response_sender.clone();
    let join_handle = tokio::spawn(async move {
        for p in partition_locations {
            let r = fetch_partition_remote_memory_byte(&p).await;
            if let Err(e) = response_sender_c.send(r).await {
                error!("Fail to send response event to the channel due to {}", e);
            }
        }
    });
    join_handles.push(join_handle);

    AbortableReceiverStream::create(response_receiver, join_handles)
}

fn send_fetch_partitions_from_disk(
    partition_locations: Vec<PartitionLocation>,
    max_request_num: usize,
) -> AbortableReceiverStream {
    let (response_sender, response_receiver) = mpsc::channel(max_request_num);
    let semaphore = Arc::new(Semaphore::new(max_request_num));
    let mut join_handles = vec![];
    let (local_locations, remote_locations): (Vec<_>, Vec<_>) = partition_locations
        .into_iter()
        .partition(check_is_local_location);

    info!(
        "local shuffle file counts:{}, remote shuffle file count:{}.",
        local_locations.len(),
        remote_locations.len()
    );

    // keep local shuffle files reading in serial order for memory control.
    let response_sender_c = response_sender.clone();
    let join_handle = tokio::spawn(async move {
        for p in local_locations {
            let r = PartitionReaderEnum::Local.fetch_partition(&p).await;
            if let Err(e) = response_sender_c.send(r).await {
                error!("Fail to send response event to the channel due to {}", e);
            }
        }
    });
    join_handles.push(join_handle);

    for p in remote_locations.into_iter() {
        let semaphore = semaphore.clone();
        let response_sender = response_sender.clone();
        let join_handle = tokio::spawn(async move {
            // Block if exceeds max request number
            let permit = semaphore.acquire_owned().await.unwrap();
            let r = PartitionReaderEnum::FlightRemoteDisk
                .fetch_partition(&p)
                .await;
            // Block if the channel buffer is ful
            if let Err(e) = response_sender.send(r).await {
                error!("Fail to send response event to the channel due to {}", e);
            }
            // Increase semaphore by dropping existing permits.
            drop(permit);
        });
        join_handles.push(join_handle);
    }

    AbortableReceiverStream::create(response_receiver, join_handles)
}

fn check_is_local_location(location: &PartitionLocation) -> bool {
    std::path::Path::new(location.path.as_str()).exists()
}

#[derive(Clone)]
enum PartitionReaderEnum {
    Local,
    FlightRemoteDisk,
}

impl PartitionReaderEnum {
    // Notice return `BallistaError::FetchFailed` will let scheduler re-schedule the task.
    async fn fetch_partition(
        &self,
        location: &PartitionLocation,
    ) -> result::Result<SendableRecordBatchStream, BallistaError> {
        match self {
            PartitionReaderEnum::FlightRemoteDisk => {
                fetch_partition_remote_disk(location).await
            }
            PartitionReaderEnum::Local => fetch_partition_local_disk(location).await,
        }
    }
}

async fn fetch_partition_remote_disk(
    location: &PartitionLocation,
) -> result::Result<SendableRecordBatchStream, BallistaError> {
    let metadata = &location.executor_meta;
    let partition_id = &location.partition_id;
    // TODO for shuffle client connections, we should avoid creating new connections again and again.
    // And we should also avoid to keep alive too many connections for long time.
    let host = metadata.host.as_str();
    let port = metadata.port;
    let mut ballista_client =
        BallistaClient::try_new(host, port)
            .await
            .map_err(|error| match error {
                // map grpc connection error to partition fetch error.
                BallistaError::GrpcConnectionError(msg) => BallistaError::FetchFailed(
                    metadata.id.clone(),
                    partition_id.stage_id,
                    partition_id.partition_id,
                    msg,
                ),
                other => other,
            })?;

    ballista_client
        .fetch_partition(&metadata.id, partition_id, &location.path, host, port)
        .await
}

async fn fetch_partition_local_disk(
    location: &PartitionLocation,
) -> result::Result<SendableRecordBatchStream, BallistaError> {
    let path = &location.path;
    let metadata = &location.executor_meta;
    let partition_id = &location.partition_id;

    let reader = fetch_partition_local_disk_inner(path).map_err(|e| {
        // return BallistaError::FetchFailed may let scheduler retry this task.
        BallistaError::FetchFailed(
            metadata.id.clone(),
            partition_id.stage_id,
            partition_id.partition_id,
            e.to_string(),
        )
    })?;
    Ok(Box::pin(LocalShuffleStream::new(reader)))
}

fn fetch_partition_local_disk_inner(
    path: &str,
) -> result::Result<FileReader<File>, BallistaError> {
    let file = File::open(path).map_err(|e| {
        BallistaError::General(format!("Failed to open partition file at {path}: {e:?}"))
    })?;
    FileReader::try_new(file, None).map_err(|e| {
        BallistaError::General(format!("Failed to new arrow FileReader at {path}: {e:?}"))
    })
}

#[allow(dead_code)]
fn get_shm_size(fd: i32) -> Result<usize, std::io::Error> {
    let mut stat_buf = std::mem::MaybeUninit::<libc::stat>::uninit();
    let ret = unsafe { libc::fstat(fd, stat_buf.as_mut_ptr()) };
    if ret < 0 {
        return Err(std::io::Error::last_os_error());
    }
    let stat_buf = unsafe { stat_buf.assume_init() };
    Ok(stat_buf.st_size as usize)
}

async fn fetch_partition_remote_memory_byte(
    location: &PartitionLocation,
) -> result::Result<SendableRecordBatchStream, BallistaError> {
    let path = &location.path;
    let metadata = &location.executor_meta;
    let partition_id = &location.partition_id;

    let reader = {
        let shm_name = CString::new(path.to_owned()).unwrap();

        let raw_fd =
            unsafe { libc::shm_open(shm_name.as_ptr(), libc::O_RDONLY, libc::S_IRUSR) };

        if raw_fd < 0 {
            return Err(BallistaError::General(format!(
                "Failed to open shared memory at {}, err {}",
                path,
                std::io::Error::last_os_error()
            )));
        }
        let size = location
            .partition_stats
            .physical_bytes
            .expect("physical_bytes is None") as usize;

        log::info!(
            "fetch_partition_remote_memory_byte: path:{}, physical size:{}, record batch size:{}",
            path,
            size,
            location.partition_stats.num_bytes.unwrap()
        );

        let shm_ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                size,
                libc::PROT_READ,
                libc::MAP_SHARED,
                raw_fd,
                0,
            )
        } as *mut u8;

        let reader = MemoryReader::new(shm_ptr, size);

        NoBufReader::try_new(reader, None).map_err(|e| {
            BallistaError::General(format!(
                "Failed to new arrow FileReader at {path}: {e:?}"
            ))
        })
    };

    let reader = reader.map_err(|e| {
        // return BallistaError::FetchFailed may let scheduler retry this task.
        BallistaError::FetchFailed(
            metadata.id.clone(),
            partition_id.stage_id,
            partition_id.partition_id,
            e.to_string(),
        )
    })?;
    Ok(Box::pin(LocalShuffleStream::new(reader)))
}

async fn fetch_partition_remote_memory_file(
    location: &PartitionLocation,
) -> result::Result<SendableRecordBatchStream, BallistaError> {
    let path = &location.path;
    let metadata = &location.executor_meta;
    let partition_id = &location.partition_id;

    let reader = {
        let shm_name = CString::new(path.to_owned()).unwrap();

        let raw_fd = unsafe {
            libc::shm_open(
                shm_name.as_ptr(),
                libc::O_RDONLY,
                libc::S_IRUSR | libc::S_IWUSR,
            )
        };

        if raw_fd < 0 {
            return Err(BallistaError::General(format!(
                "Failed to open shared memory at {}, err {}",
                path,
                std::io::Error::last_os_error()
            )));
        }

        let file = unsafe { <File as std::os::fd::FromRawFd>::from_raw_fd(raw_fd) };

        NoBufReader::try_new(file, None).map_err(|e| {
            BallistaError::General(format!(
                "Failed to new arrow FileReader at {path}: {e:?}"
            ))
        })
    };

    let reader = reader.map_err(|e| {
        // return BallistaError::FetchFailed may let scheduler retry this task.
        BallistaError::FetchFailed(
            metadata.id.clone(),
            partition_id.stage_id,
            partition_id.partition_id,
            e.to_string(),
        )
    })?;
    Ok(Box::pin(LocalShuffleStream::new(reader)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution_plans::ShuffleWriterExec;
    use crate::serde::scheduler::{ExecutorMetadata, ExecutorSpecification, PartitionId};
    use crate::utils::{self, RemoteMemoryMode};
    use datafusion::arrow::array::{Int32Array, StringArray, UInt32Array};
    use datafusion::arrow::datatypes::{DataType, Field, Schema};
    use datafusion::arrow::ipc::writer::FileWriter;
    use datafusion::arrow::record_batch::RecordBatch;
    use datafusion::error::DataFusionError;
    use datafusion::physical_expr::expressions::Column;
    use datafusion::physical_plan::common;
    use datafusion::physical_plan::memory::MemoryExec;
    use datafusion::prelude::SessionContext;
    use tempfile::{tempdir, TempDir};

    #[tokio::test]
    async fn test_stats_for_partitions_empty() {
        let result = stats_for_partitions(0, std::iter::empty());

        let exptected = Statistics {
            num_rows: Precision::Exact(0),
            total_byte_size: Precision::Exact(0),
            column_statistics: vec![],
        };

        assert_eq!(result, exptected);
    }

    #[tokio::test]
    async fn test_stats_for_partitions_full() {
        let part_stats = vec![
            PartitionStats {
                num_rows: Some(10),
                num_bytes: Some(84),
                num_batches: Some(1),
                physical_bytes: Some(84),
                ht_bucket_mask: Some(0),
                ht_growth_left: Some(0),
                ht_items: Some(0),
            },
            PartitionStats {
                num_rows: Some(4),
                num_bytes: Some(65),
                num_batches: None,
                physical_bytes: Some(65),
                ht_bucket_mask: Some(0),
                ht_growth_left: Some(0),
                ht_items: Some(0),
            },
        ];

        let result = stats_for_partitions(0, part_stats.into_iter());

        let exptected = Statistics {
            num_rows: Precision::Exact(14),
            total_byte_size: Precision::Exact(149),
            column_statistics: vec![],
        };

        assert_eq!(result, exptected);
    }

    #[tokio::test]
    async fn test_stats_for_partitions_missing() {
        let part_stats = vec![
            PartitionStats {
                num_rows: Some(10),
                num_bytes: Some(84),
                num_batches: Some(1),
                physical_bytes: Some(84),
                ht_bucket_mask: Some(0),
                ht_growth_left: Some(0),
                ht_items: Some(0),
            },
            PartitionStats {
                num_rows: None,
                num_bytes: None,
                num_batches: None,
                physical_bytes: None,
                ht_bucket_mask: None,
                ht_growth_left: None,
                ht_items: None,
            },
        ];

        let result = stats_for_partitions(0, part_stats.into_iter());

        let exptected = Statistics {
            num_rows: Precision::Absent,
            total_byte_size: Precision::Absent,
            column_statistics: vec![],
        };

        assert_eq!(result, exptected);
    }

    #[tokio::test]
    async fn test_fetch_partitions_error_mapping() -> Result<()> {
        let session_ctx = SessionContext::new();
        let task_ctx = session_ctx.task_ctx();
        let schema = Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Int32, false),
            Field::new("c", DataType::Int32, false),
        ]);

        let job_id = "test_job_1";
        let mut partitions: Vec<PartitionLocation> = vec![];
        for partition_id in 0..4 {
            partitions.push(PartitionLocation {
                map_partition_id: 0,
                partition_id: PartitionId {
                    job_id: job_id.to_string(),
                    stage_id: 2,
                    partition_id,
                },
                executor_meta: ExecutorMetadata {
                    id: "executor_1".to_string(),
                    host: "executor_1".to_string(),
                    port: 7070,
                    grpc_port: 8080,
                    specification: ExecutorSpecification { task_slots: 1 },
                },
                partition_stats: Default::default(),
                path: "test_path".to_string(),
            })
        }
        let input_stage_id = 2;

        let shuffle_reader_exec = ShuffleReaderExec::try_new(
            input_stage_id,
            vec![partitions],
            Arc::new(schema),
            RemoteMemoryMode::default(),
        )?;
        let mut stream = shuffle_reader_exec.execute(0, task_ctx)?;
        let batches = utils::collect_stream(&mut stream).await;

        assert!(batches.is_err());

        // BallistaError::FetchFailed -> ArrowError::ExternalError -> ballistaError::FetchFailed
        let ballista_error = batches.unwrap_err();
        assert!(matches!(
            ballista_error,
            BallistaError::FetchFailed(_, _, _, _)
        ));

        Ok(())
    }

    #[tokio::test]
    async fn test_send_fetch_partitions_1() {
        test_send_fetch_partitions(1, 10).await;
    }

    #[tokio::test]
    async fn test_send_fetch_partitions_n() {
        test_send_fetch_partitions(4, 10).await;
    }

    #[tokio::test]
    async fn test_read_local_shuffle() {
        let session_ctx = SessionContext::new();
        let task_ctx = session_ctx.task_ctx();
        let work_dir = TempDir::new().unwrap();
        let input = ShuffleWriterExec::try_new(
            "local_file".to_owned(),
            1,
            create_test_data_plan().unwrap(),
            work_dir.into_path().to_str().unwrap().to_owned(),
            Some(Partitioning::Hash(vec![Arc::new(Column::new("a", 0))], 1)),
            RemoteMemoryMode::default(),
            utils::JoinInputSide::NotApplicable,
        )
        .unwrap();

        let mut stream = input.execute(0, task_ctx).unwrap();

        let batches = utils::collect_stream(&mut stream)
            .await
            .map_err(|e| DataFusionError::Execution(format!("{e:?}")))
            .unwrap();

        let path = batches[0].columns()[1]
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();

        // from to input partitions test the first one with two batches
        let file_path = path.value(0);
        let reader = fetch_partition_local_disk_inner(file_path).unwrap();

        let mut stream: Pin<Box<dyn RecordBatchStream + Send>> =
            async { Box::pin(LocalShuffleStream::new(reader)) }.await;

        let result = utils::collect_stream(&mut stream)
            .await
            .map_err(|e| DataFusionError::Execution(format!("{e:?}")))
            .unwrap();

        assert_eq!(result.len(), 2);
        for b in result {
            assert_eq!(b, create_test_batch())
        }
    }

    async fn test_send_fetch_partitions(max_request_num: usize, partition_num: usize) {
        let schema = get_test_partition_schema();
        let data_array = Int32Array::from(vec![1]);
        let batch =
            RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(data_array)])
                .unwrap();
        let tmp_dir = tempdir().unwrap();
        let file_path = tmp_dir.path().join("shuffle_data");
        let file = File::create(&file_path).unwrap();
        let mut writer = FileWriter::try_new(file, &schema).unwrap();
        writer.write(&batch).unwrap();
        writer.finish().unwrap();

        let partition_locations = get_test_partition_locations(
            partition_num,
            file_path.to_str().unwrap().to_string(),
        );

        let response_receiver =
            send_fetch_partitions_from_disk(partition_locations, max_request_num);

        let stream = RecordBatchStreamAdapter::new(
            Arc::new(schema),
            response_receiver.try_flatten(),
        );

        let result = common::collect(Box::pin(stream)).await.unwrap();
        assert_eq!(partition_num, result.len());
    }

    fn get_test_partition_locations(n: usize, path: String) -> Vec<PartitionLocation> {
        (0..n)
            .map(|partition_id| PartitionLocation {
                map_partition_id: 0,
                partition_id: PartitionId {
                    job_id: "job".to_string(),
                    stage_id: 1,
                    partition_id,
                },
                executor_meta: ExecutorMetadata {
                    id: format!("exec{partition_id}"),
                    host: "localhost".to_string(),
                    port: 50051,
                    grpc_port: 50052,
                    specification: ExecutorSpecification { task_slots: 12 },
                },
                partition_stats: Default::default(),
                path: path.clone(),
            })
            .collect()
    }

    fn get_test_partition_schema() -> Schema {
        Schema::new(vec![Field::new("id", DataType::Int32, false)])
    }

    // create two partitions each has two same batches
    fn create_test_data_plan() -> Result<Arc<dyn ExecutionPlan>> {
        let batch = create_test_batch();
        let partition = vec![batch.clone(), batch];
        let partitions = vec![partition.clone(), partition];
        Ok(Arc::new(MemoryExec::try_new(
            &partitions,
            create_test_schema(),
            None,
        )?))
    }

    fn create_test_batch() -> RecordBatch {
        RecordBatch::try_new(
            create_test_schema(),
            vec![
                Arc::new(UInt32Array::from(vec![Some(1), Some(2), Some(3)])),
                Arc::new(StringArray::from(vec![
                    Some("rust"),
                    Some("datafusion"),
                    Some("ballista"),
                ])),
            ],
        )
        .unwrap()
    }

    fn create_test_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("number", DataType::UInt32, true),
            Field::new("str", DataType::Utf8, true),
        ]))
    }
}
