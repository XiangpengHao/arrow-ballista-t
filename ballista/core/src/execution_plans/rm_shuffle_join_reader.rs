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
use std::os::fd::FromRawFd;
use std::result;
use std::sync::Arc;

use crate::serde::scheduler::{PartitionLocation, PartitionStats};

use datafusion::arrow::datatypes::SchemaRef;
use datafusion::arrow::ipc::reader::FileReader;

use datafusion::error::Result;
use datafusion::physical_plan::expressions::PhysicalSortExpr;
use datafusion::physical_plan::metrics::{ExecutionPlanMetricsSet, MetricsSet};
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning, SendableRecordBatchStream,
    Statistics,
};
use futures::TryStreamExt;

use crate::error::BallistaError;
use datafusion::execution::context::TaskContext;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use itertools::Itertools;
use log::{error, info};
use rand::prelude::SliceRandom;
use rand::thread_rng;
use tokio::sync::mpsc;

use super::shuffle_reader::{AbortableReceiverStream, LocalShuffleStream};

/// ShuffleReaderExec reads partitions that have already been materialized by a ShuffleWriterExec
/// being executed by an executor
#[derive(Debug, Clone)]
pub struct RemoteShuffleReaderExec {
    /// The query stage id to read from
    pub stage_id: usize,
    /// Each partition of a shuffle can read data from multiple locations
    pub partition: Vec<Vec<PartitionLocation>>,
    pub schema: SchemaRef,
    /// Execution metrics
    metrics: ExecutionPlanMetricsSet,
}

impl RemoteShuffleReaderExec {
    /// Create a new ShuffleReaderExec
    pub fn try_new(
        stage_id: usize,
        partition: Vec<Vec<PartitionLocation>>,
        schema: SchemaRef,
    ) -> Result<Self> {
        Ok(Self {
            stage_id,
            schema,
            partition,
            metrics: ExecutionPlanMetricsSet::new(),
        })
    }
}

impl ExecutionPlan for RemoteShuffleReaderExec {
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
        Ok(Arc::new(RemoteShuffleReaderExec::try_new(
            self.stage_id,
            self.partition.clone(),
            self.schema.clone(),
        )?))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let task_id = context.task_id().unwrap_or_else(|| partition.to_string());
        info!("ShuffleReaderExec::execute({})", task_id);

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

        let response_receiver =
            send_fetch_partitions(partition_locations, max_request_num);

        let result = RecordBatchStreamAdapter::new(
            Arc::new(self.schema.as_ref().clone()),
            response_receiver.try_flatten(),
        );
        Ok(Box::pin(result))
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }

    fn statistics(&self) -> Statistics {
        stats_for_partitions(
            self.partition
                .iter()
                .flatten()
                .map(|loc| loc.partition_stats),
        )
    }
}

impl DisplayAs for RemoteShuffleReaderExec {
    fn fmt_as(
        &self,
        t: DisplayFormatType,
        f: &mut std::fmt::Formatter,
    ) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(
                    f,
                    "RemoteShuffleReaderExec: partitions={}",
                    self.partition.len()
                )
            }
        }
    }
}

fn stats_for_partitions(
    partition_stats: impl Iterator<Item = PartitionStats>,
) -> Statistics {
    // TODO stats: add column statistics to PartitionStats
    partition_stats.fold(
        Statistics {
            is_exact: true,
            num_rows: Some(0),
            total_byte_size: Some(0),
            column_statistics: None,
        },
        |mut acc, part| {
            // if any statistic is unkown it makes the entire statistic unkown
            acc.num_rows = acc.num_rows.zip(part.num_rows).map(|(a, b)| a + b as usize);
            acc.total_byte_size = acc
                .total_byte_size
                .zip(part.num_bytes)
                .map(|(a, b)| a + b as usize);
            acc
        },
    )
}

fn send_fetch_partitions(
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
            let r = fetch_partition_local(&p).await;
            if let Err(e) = response_sender_c.send(r).await {
                error!("Fail to send response event to the channel due to {}", e);
            }
        }
    });
    join_handles.push(join_handle);

    AbortableReceiverStream::create(response_receiver, join_handles)
}

async fn fetch_partition_local(
    location: &PartitionLocation,
) -> result::Result<SendableRecordBatchStream, BallistaError> {
    let path = &location.path;
    let metadata = &location.executor_meta;
    let partition_id = &location.partition_id;

    let reader = fetch_partition_local_inner(path).map_err(|e| {
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

fn fetch_partition_local_inner(
    path: &str,
) -> result::Result<FileReader<File>, BallistaError> {
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
            "Failed to open shared memory at {path}",
            path = path
        )));
    }

    let file = unsafe { File::from_raw_fd(raw_fd) };

    FileReader::try_new(file, None).map_err(|e| {
        BallistaError::General(format!("Failed to new arrow FileReader at {path}: {e:?}"))
    })
}
