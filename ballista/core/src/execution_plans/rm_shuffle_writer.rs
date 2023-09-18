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

//! ShuffleWriterExec represents a section of a query plan that has consistent partitioning and
//! can be executed as one unit with each partition being executed in parallel. The output of each
//! partition is re-partitioned and streamed to disk in Arrow IPC format. Future stages of the query
//! will use the ShuffleReaderExec to read these results.
//!
//! Remote memory based shuffle writer, which writes the arrow IPC data to a remote memory pool (instead of local disk).
//! Later the remote memory shuffle reader will read the IPC data from the remote memory pool.
//! The hope is that this is much more efficient.

use datafusion::physical_plan::expressions::PhysicalSortExpr;

use std::any::Any;
use std::future::Future;
use std::iter::Iterator;
use std::str::FromStr;
use std::sync::Arc;
use std::time::Instant;

use crate::utils;

use crate::serde::protobuf::ShuffleWritePartition;
use crate::serde::scheduler::PartitionStats;
use datafusion::arrow::array::{
    ArrayBuilder, ArrayRef, StringBuilder, StructBuilder, UInt32Builder, UInt64Builder,
};
use datafusion::arrow::datatypes::{DataType, Field, Schema, SchemaRef};

use datafusion::arrow::record_batch::RecordBatch;
use datafusion::error::{DataFusionError, Result};
use datafusion::physical_plan::memory::MemoryStream;
use datafusion::physical_plan::metrics::{
    self, ExecutionPlanMetricsSet, MetricBuilder, MetricsSet,
};

use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning, SendableRecordBatchStream,
    Statistics,
};
use futures::{StreamExt, TryFutureExt, TryStreamExt};

use datafusion::arrow::error::ArrowError;
use datafusion::execution::context::TaskContext;
use datafusion::physical_plan::repartition::BatchPartitioner;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use log::{debug, info};

use super::sm_writer::SharedMemoryWriter;
use super::ShuffleWriter;

/// ShuffleWriterExec represents a section of a query plan that has consistent partitioning and
/// can be executed as one unit with each partition being executed in parallel. The output of each
/// partition is re-partitioned and streamed to disk in Arrow IPC format. Future stages of the query
/// will use the ShuffleReaderExec to read these results.
#[derive(Debug, Clone)]
pub struct RemoteShuffleWriterExec {
    /// Unique ID for the job (query) that this stage is a part of
    job_id: String,
    /// Unique query stage ID within the job
    stage_id: usize,
    /// Physical execution plan for this query stage
    plan: Arc<dyn ExecutionPlan>,
    /// Path to write output streams to
    work_dir: String,
    /// Optional shuffle output partitioning
    shuffle_output_partitioning: Option<Partitioning>,
    /// Execution metrics
    metrics: ExecutionPlanMetricsSet,
}

#[derive(Debug, Clone)]
struct ShuffleWriteMetrics {
    /// Time spend writing batches to shuffle files
    write_time: metrics::Time,
    repart_time: metrics::Time,
    input_rows: metrics::Count,
    output_rows: metrics::Count,
}

impl ShuffleWriteMetrics {
    fn new(partition: usize, metrics: &ExecutionPlanMetricsSet) -> Self {
        let write_time = MetricBuilder::new(metrics).subset_time("write_time", partition);
        let repart_time =
            MetricBuilder::new(metrics).subset_time("repart_time", partition);

        let input_rows = MetricBuilder::new(metrics).counter("input_rows", partition);

        let output_rows = MetricBuilder::new(metrics).output_rows(partition);

        Self {
            write_time,
            repart_time,
            input_rows,
            output_rows,
        }
    }
}

impl ShuffleWriter for RemoteShuffleWriterExec {
    /// Get the Job ID for this query stage
    fn job_id(&self) -> &str {
        &self.job_id
    }

    /// Get the Stage ID for this query stage
    fn stage_id(&self) -> usize {
        self.stage_id
    }

    fn use_remote_memory() -> bool {
        true
    }

    /// Get the true output partitioning
    fn shuffle_output_partitioning(&self) -> Option<&Partitioning> {
        self.shuffle_output_partitioning.as_ref()
    }

    /// Create a new shuffle writer
    fn try_new(
        job_id: String,
        stage_id: usize,
        plan: Arc<dyn ExecutionPlan>,
        work_dir: String,
        shuffle_output_partitioning: Option<Partitioning>,
    ) -> Result<Self> {
        info!("Creating shuffle writer for stage {}", stage_id);
        Ok(Self {
            job_id,
            stage_id,
            plan,
            work_dir,
            shuffle_output_partitioning,
            metrics: ExecutionPlanMetricsSet::new(),
        })
    }
}

impl RemoteShuffleWriterExec {
    pub fn execute_shuffle_write(
        &self,
        input_partition: usize,
        context: Arc<TaskContext>,
    ) -> impl Future<Output = Result<Vec<ShuffleWritePartition>>> {
        let mut identifier = String::from_str("/").unwrap();
        identifier.push_str(&self.job_id);
        identifier.push_str(&format!("-{}", self.stage_id));

        let write_metrics = ShuffleWriteMetrics::new(input_partition, &self.metrics);
        let output_partitioning = self.shuffle_output_partitioning.clone();
        let plan = self.plan.clone();

        async move {
            let now = Instant::now();
            let mut stream = plan.execute(input_partition, context)?;

            match output_partitioning {
                None => {
                    let timer = write_metrics.write_time.timer();
                    identifier.push_str(&format!("{input_partition}"));
                    identifier.push_str("data.arrow");
                    debug!("Writing results to {}", &identifier);

                    // stream results to disk
                    let stats = utils::write_stream_to_disk(
                        &mut stream,
                        &identifier,
                        &write_metrics.write_time,
                        true,
                    )
                    .await
                    .map_err(|e| DataFusionError::Execution(format!("{e:?}")))?;

                    write_metrics
                        .input_rows
                        .add(stats.num_rows.unwrap_or(0) as usize);
                    write_metrics
                        .output_rows
                        .add(stats.num_rows.unwrap_or(0) as usize);
                    timer.done();

                    info!(
                        "Executed partition {} in {} seconds. Statistics: {}",
                        input_partition,
                        now.elapsed().as_secs(),
                        stats
                    );

                    Ok(vec![ShuffleWritePartition {
                        partition_id: input_partition as u64,
                        path: identifier,
                        num_batches: stats.num_batches.unwrap_or(0),
                        num_rows: stats.num_rows.unwrap_or(0),
                        num_bytes: stats.num_bytes.unwrap_or(0),
                    }])
                }

                Some(Partitioning::Hash(exprs, num_output_partitions)) => {
                    // we won't necessary produce output for every possible partition, so we
                    // create writers on demand
                    let mut writers: Vec<Option<SharedMemoryWriter>> = vec![];
                    for _ in 0..num_output_partitions {
                        writers.push(None);
                    }

                    let mut partitioner = BatchPartitioner::try_new(
                        Partitioning::Hash(exprs, num_output_partitions),
                        write_metrics.repart_time.clone(),
                    )?;

                    while let Some(result) = stream.next().await {
                        let input_batch = result?;

                        write_metrics.input_rows.add(input_batch.num_rows());

                        partitioner.partition(
                            input_batch,
                            |output_partition, output_batch| {
                                // partition func in datafusion make sure not write empty output_batch.
                                let timer = write_metrics.write_time.timer();
                                match &mut writers[output_partition] {
                                    Some(w) => {
                                        w.write(&output_batch)?;
                                    }
                                    None => {
                                        let mut idt = identifier.clone();
                                        idt.push_str(&format!("-{output_partition}"));
                                        std::fs::create_dir_all(&idt)?;

                                        idt.push_str(&format!(
                                            "-data-{input_partition}.arrow"
                                        ));
                                        debug!("Writing results to {:?}", idt);

                                        let mut writer = SharedMemoryWriter::new(
                                            idt,
                                            stream.schema().as_ref(),
                                        )?;

                                        writer.write(&output_batch)?;
                                        writers[output_partition] = Some(writer);
                                    }
                                }
                                write_metrics.output_rows.add(output_batch.num_rows());
                                timer.done();
                                Ok(())
                            },
                        )?;
                    }

                    let mut part_locs = vec![];

                    for (i, w) in writers.iter_mut().enumerate() {
                        match w {
                            Some(w) => {
                                w.finish()?;
                                debug!(
                                    "Finished writing shuffle partition {} at {:?}. Batches: {}. Rows: {}. Bytes: {}.",
                                    i,
                                    w.identifier(),
                                    w.num_batches,
                                    w.num_rows,
                                    w.num_bytes
                                );

                                part_locs.push(ShuffleWritePartition {
                                    partition_id: i as u64,
                                    path: w.identifier().to_owned(),
                                    num_batches: w.num_batches,
                                    num_rows: w.num_rows,
                                    num_bytes: w.num_bytes,
                                });
                            }
                            None => {}
                        }
                    }
                    Ok(part_locs)
                }

                _ => Err(DataFusionError::Execution(
                    "Invalid shuffle partitioning scheme".to_owned(),
                )),
            }
        }
    }
}

impl ExecutionPlan for RemoteShuffleWriterExec {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        self.plan.schema()
    }

    fn output_partitioning(&self) -> Partitioning {
        // This operator needs to be executed once for each *input* partition and there
        // isn't really a mechanism yet in DataFusion to support this use case so we report
        // the input partitioning as the output partitioning here. The executor reports
        // output partition meta data back to the scheduler.
        self.plan.output_partitioning()
    }

    fn output_ordering(&self) -> Option<&[PhysicalSortExpr]> {
        None
    }

    fn children(&self) -> Vec<Arc<dyn ExecutionPlan>> {
        vec![self.plan.clone()]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(RemoteShuffleWriterExec::try_new(
            self.job_id.clone(),
            self.stage_id,
            children[0].clone(),
            self.work_dir.clone(),
            self.shuffle_output_partitioning.clone(),
        )?))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let schema = result_schema();

        let schema_captured = schema.clone();
        let fut_stream = self
            .execute_shuffle_write(partition, context)
            .and_then(|part_loc| async move {
                // build metadata result batch
                let num_writers = part_loc.len();
                let mut partition_builder = UInt32Builder::with_capacity(num_writers);
                let mut path_builder =
                    StringBuilder::with_capacity(num_writers, num_writers * 100);
                let mut num_rows_builder = UInt64Builder::with_capacity(num_writers);
                let mut num_batches_builder = UInt64Builder::with_capacity(num_writers);
                let mut num_bytes_builder = UInt64Builder::with_capacity(num_writers);

                for loc in &part_loc {
                    path_builder.append_value(loc.path.clone());
                    partition_builder.append_value(loc.partition_id as u32);
                    num_rows_builder.append_value(loc.num_rows);
                    num_batches_builder.append_value(loc.num_batches);
                    num_bytes_builder.append_value(loc.num_bytes);
                }

                // build arrays
                let partition_num: ArrayRef = Arc::new(partition_builder.finish());
                let path: ArrayRef = Arc::new(path_builder.finish());
                let field_builders: Vec<Box<dyn ArrayBuilder>> = vec![
                    Box::new(num_rows_builder),
                    Box::new(num_batches_builder),
                    Box::new(num_bytes_builder),
                ];
                let mut stats_builder = StructBuilder::new(
                    PartitionStats::default().arrow_struct_fields(),
                    field_builders,
                );
                for _ in 0..num_writers {
                    stats_builder.append(true);
                }
                let stats = Arc::new(stats_builder.finish());

                // build result batch containing metadata
                let batch = RecordBatch::try_new(
                    schema_captured.clone(),
                    vec![partition_num, path, stats],
                )?;

                debug!("RESULTS METADATA:\n{:?}", batch);

                MemoryStream::try_new(vec![batch], schema_captured, None)
            })
            .map_err(|e| ArrowError::ExternalError(Box::new(e)));

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            schema,
            futures::stream::once(fut_stream).try_flatten(),
        )))
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }

    fn statistics(&self) -> Statistics {
        self.plan.statistics()
    }
}

impl DisplayAs for RemoteShuffleWriterExec {
    fn fmt_as(
        &self,
        t: DisplayFormatType,
        f: &mut std::fmt::Formatter,
    ) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(
                    f,
                    "RemoteShuffleWriterExec: {:?}",
                    self.shuffle_output_partitioning
                )
            }
        }
    }
}

fn result_schema() -> SchemaRef {
    let stats = PartitionStats::default();
    Arc::new(Schema::new(vec![
        Field::new("partition", DataType::UInt32, false),
        Field::new("path", DataType::Utf8, false),
        stats.arrow_struct_repr(),
    ]))
}
