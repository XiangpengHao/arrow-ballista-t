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

use ahash::RandomState;
use datafusion::physical_plan::expressions::PhysicalSortExpr;
use datafusion::physical_plan::hash_utils::create_hashes;

use std::any::Any;
use std::future::Future;
use std::iter::Iterator;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use crate::execution_plans::join_utils::JoinHashMap;
use crate::execution_plans::rm_writer::{
    BuffedDirectWriter, MemoryReader, NoBufReader, SharedMemoryByteWriter,
    SharedMemoryFileWriter,
};
use crate::utils::{self, JoinInputSide, RemoteMemoryMode};

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

/// ShuffleWriterExec represents a section of a query plan that has consistent partitioning and
/// can be executed as one unit with each partition being executed in parallel. The output of each
/// partition is re-partitioned and streamed to disk in Arrow IPC format. Future stages of the query
/// will use the ShuffleReaderExec to read these results.
#[derive(Debug, Clone)]
pub struct ShuffleWriterExec {
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

    remote_mode: RemoteMemoryMode,

    join_input_side: JoinInputSide,
}

#[derive(Debug, Clone)]
pub(super) struct ShuffleWriteMetrics {
    /// Time spend writing batches to shuffle files
    pub(super) write_time: metrics::Time,
    pub(super) repart_time: metrics::Time,
    pub(super) input_rows: metrics::Count,
    pub(super) output_rows: metrics::Count,
}

impl ShuffleWriteMetrics {
    pub(super) fn new(partition: usize, metrics: &ExecutionPlanMetricsSet) -> Self {
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

impl ShuffleWriterExec {
    /// Get the Job ID for this query stage
    pub fn job_id(&self) -> &str {
        &self.job_id
    }

    /// Get the Stage ID for this query stage
    pub fn stage_id(&self) -> usize {
        self.stage_id
    }

    /// Get the true output partitioning
    pub fn shuffle_output_partitioning(&self) -> Option<&Partitioning> {
        self.shuffle_output_partitioning.as_ref()
    }

    pub fn remote_memory_mode(&self) -> RemoteMemoryMode {
        self.remote_mode
    }

    pub fn join_input_side(&self) -> JoinInputSide {
        self.join_input_side
    }

    /// Create a new shuffle writer
    pub fn try_new(
        job_id: String,
        stage_id: usize,
        plan: Arc<dyn ExecutionPlan>,
        work_dir: String,
        shuffle_output_partitioning: Option<Partitioning>,
        remote_memory_mode: RemoteMemoryMode,
        join_input_side: JoinInputSide,
    ) -> Result<Self> {
        Ok(Self {
            job_id,
            stage_id,
            plan,
            work_dir,
            shuffle_output_partitioning,
            metrics: ExecutionPlanMetricsSet::new(),
            remote_mode: remote_memory_mode,
            join_input_side,
        })
    }
}

impl ShuffleWriterExec {
    pub fn execute_shuffle_write(
        &self,
        input_partition: usize,
        context: Arc<TaskContext>,
    ) -> impl Future<Output = Result<Vec<ShuffleWritePartition>>> {
        let write_metrics = ShuffleWriteMetrics::new(input_partition, &self.metrics);
        let output_partitioning = self.shuffle_output_partitioning.clone();
        let plan = self.plan.clone();
        let mode = self.remote_mode;

        let path = match mode {
            RemoteMemoryMode::DoNotUse => {
                let mut path = PathBuf::from(&self.work_dir);
                path.push(&self.job_id);
                path.push(&format!("{}", self.stage_id));
                path
            }
            RemoteMemoryMode::FileBasedShuffle
            | RemoteMemoryMode::MemoryBasedShuffle
            | RemoteMemoryMode::JoinOnRemote => {
                PathBuf::from(format!("/shm-{}-{}", self.job_id, self.stage_id))
            }
        };

        async move {
            match mode {
                RemoteMemoryMode::DoNotUse => {
                    execute_conventional_shuffle_write(
                        path,
                        plan,
                        output_partitioning,
                        write_metrics,
                        input_partition,
                        context,
                    )
                    .await
                }
                RemoteMemoryMode::FileBasedShuffle => {
                    execute_file_based_remote_shuffle_write(
                        path,
                        plan,
                        output_partitioning,
                        write_metrics,
                        input_partition,
                        context,
                    )
                    .await
                }
                RemoteMemoryMode::MemoryBasedShuffle => {
                    execute_memory_based_remote_shuffle_write(
                        path,
                        plan,
                        output_partitioning,
                        write_metrics,
                        input_partition,
                        context,
                    )
                    .await
                }

                RemoteMemoryMode::JoinOnRemote => {
                    execute_join_on_remote_shuffle_write(
                        path,
                        plan,
                        output_partitioning,
                        write_metrics,
                        input_partition,
                        context,
                    )
                    .await
                }
            }
        }
    }
}

async fn execute_file_based_remote_shuffle_write(
    identifier: PathBuf,
    plan: Arc<dyn ExecutionPlan>,
    output_partitioning: Option<Partitioning>,
    write_metrics: ShuffleWriteMetrics,
    input_partition: usize,
    context: Arc<TaskContext>,
) -> Result<Vec<ShuffleWritePartition>> {
    let mut path = identifier.to_str().unwrap().to_owned();
    let now = Instant::now();
    let mut stream = plan.execute(input_partition, context)?;

    match output_partitioning {
        None => {
            let timer = write_metrics.write_time.timer();
            path.push_str(&format!("-{input_partition}-data.arrow"));

            debug!("Writing results to {}", path);
            // stream results to disk
            let stats = utils::write_stream_to_disk(
                &mut stream,
                &path,
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
                path: path.to_owned(),
                num_batches: stats.num_batches.unwrap_or(0),
                num_rows: stats.num_rows.unwrap_or(0),
                num_bytes: stats.num_bytes.unwrap_or(0),
                physical_bytes: stats.physical_bytes.unwrap_or(0),
                ht_bucket_mask: 0,
                ht_growth_left: 0,
                ht_items: 0,
            }])
        }

        Some(Partitioning::Hash(exprs, num_output_partitions)) => {
            // we won't necessary produce output for every possible partition, so we
            // create writers on demand
            let mut writers: Vec<Option<SharedMemoryFileWriter>> = vec![];
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
                                let mut idt = path.clone();
                                idt.push_str(&format!("-{output_partition}"));

                                idt.push_str(&format!("-data-{input_partition}.arrow"));
                                debug!("Writing results to {:?}", idt);

                                let mut writer = SharedMemoryFileWriter::new(
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
                                w.path(),
                                w.num_batches,
                                w.num_rows,
                                w.num_bytes
                            );

                        part_locs.push(ShuffleWritePartition {
                            partition_id: i as u64,
                            path: w.path().to_owned(),
                            num_batches: w.num_batches,
                            num_rows: w.num_rows,
                            num_bytes: w.num_bytes,
                            physical_bytes: w.num_batches,
                            ht_bucket_mask: 0,
                            ht_growth_left: 0,
                            ht_items: 0,
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

async fn execute_conventional_shuffle_write(
    mut path: PathBuf,
    plan: Arc<dyn ExecutionPlan>,
    output_partitioning: Option<Partitioning>,
    write_metrics: ShuffleWriteMetrics,
    input_partition: usize,
    context: Arc<TaskContext>,
) -> Result<Vec<ShuffleWritePartition>> {
    let now = Instant::now();
    let mut stream = plan.execute(input_partition, context)?;

    match output_partitioning {
        None => {
            let timer = write_metrics.write_time.timer();
            path.push(&format!("{input_partition}"));
            std::fs::create_dir_all(&path)?;
            path.push("data.arrow");
            let path = path.to_str().unwrap();
            debug!("Writing results to {}", path);

            // stream results to disk
            let stats = utils::write_stream_to_disk(
                &mut stream,
                path,
                &write_metrics.write_time,
                false,
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
                path: path.to_owned(),
                num_batches: stats.num_batches.unwrap_or(0),
                num_rows: stats.num_rows.unwrap_or(0),
                num_bytes: stats.num_bytes.unwrap_or(0),
                physical_bytes: stats.physical_bytes.unwrap_or(0),
                ht_bucket_mask: 0,
                ht_growth_left: 0,
                ht_items: 0,
            }])
        }

        Some(Partitioning::Hash(exprs, num_output_partitions)) => {
            // we won't necessary produce output for every possible partition, so we
            // create writers on demand
            let mut writers: Vec<Option<BuffedDirectWriter>> = vec![];
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
                                let mut path = path.clone();
                                path.push(&format!("{output_partition}"));
                                std::fs::create_dir_all(&path)?;

                                path.push(format!("data-{input_partition}.arrow"));
                                debug!("Writing results to {:?}", path);

                                let mut writer = BuffedDirectWriter::new(
                                    &path,
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
                                w.path(),
                                w.num_batches,
                                w.num_rows,
                                w.num_bytes
                            );

                        part_locs.push(ShuffleWritePartition {
                            partition_id: i as u64,
                            path: w.path().to_string_lossy().to_string(),
                            num_batches: w.num_batches,
                            num_rows: w.num_rows,
                            num_bytes: w.num_bytes,
                            physical_bytes: w.num_bytes,
                            ht_bucket_mask: 0,
                            ht_growth_left: 0,
                            ht_items: 0,
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

async fn execute_memory_based_remote_shuffle_write(
    path: PathBuf,
    plan: Arc<dyn ExecutionPlan>,
    output_partitioning: Option<Partitioning>,
    write_metrics: ShuffleWriteMetrics,
    input_partition: usize,
    context: Arc<TaskContext>,
) -> Result<Vec<ShuffleWritePartition>> {
    let mut path = path.to_str().unwrap().to_owned();
    let now = Instant::now();
    let mut stream = plan.execute(input_partition, context)?;

    match output_partitioning {
        None => {
            let timer = write_metrics.write_time.timer();
            path.push_str(&format!("{input_partition}-data.arrow"));

            debug!("Writing results to {}", path);
            // stream results to disk
            let stats = utils::write_stream_to_disk(
                &mut stream,
                &path,
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
                path: path.to_owned(),
                num_batches: stats.num_batches.unwrap_or(0),
                num_rows: stats.num_rows.unwrap_or(0),
                num_bytes: stats.num_bytes.unwrap_or(0),
                physical_bytes: stats.physical_bytes.unwrap_or(0),
                ht_bucket_mask: 0,
                ht_growth_left: 0,
                ht_items: 0,
            }])
        }

        Some(Partitioning::Hash(exprs, num_output_partitions)) => {
            // we won't necessary produce output for every possible partition, so we
            // create writers on demand
            let mut writers: Vec<Option<SharedMemoryByteWriter>> = vec![];
            for _ in 0..num_output_partitions {
                writers.push(None);
            }

            let mut partitioner = BatchPartitioner::try_new(
                Partitioning::Hash(exprs, num_output_partitions),
                write_metrics.repart_time.clone(),
            )?;

            // TODO: we need a real cardinality estimation here.
            let estimated_size_per_partition = 1024 * 1024 * 512;

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
                                let mut idt = path.clone();
                                idt.push_str(&format!(
                                    "-{output_partition}-data-{input_partition}.arrow"
                                ));

                                debug!("Writing results to {:?}", idt);

                                let mut writer = SharedMemoryByteWriter::new(
                                    idt,
                                    stream.schema().as_ref(),
                                    estimated_size_per_partition,
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
                        info!(
                                "Finished writing shuffle partition {} at {:?}. Batches: {}. Rows: {}. Bytes: {}. Physical: {}",
                                i,
                                w.identifier(),
                                w.num_batches,
                                w.num_rows,
                                w.num_bytes,
                                w.get_physical_written_bytes()
                            );

                        let physical_size = w.get_physical_written_bytes();

                        part_locs.push(ShuffleWritePartition {
                            partition_id: i as u64,
                            path: w.identifier().to_owned(),
                            num_batches: w.num_batches,
                            num_rows: w.num_rows,
                            num_bytes: w.num_bytes,
                            physical_bytes: physical_size as u64,
                            ht_bucket_mask: 0,
                            ht_growth_left: 0,
                            ht_items: 0,
                        });
                    }
                    None => {}
                }
            }
            Ok(part_locs)
        }

        Some(Partitioning::RoundRobinBatch(num_output_partitions)) => {
            // RounRobin partition happens when we deal with RHS join input.
            // (FYI, the LHS join input will build the hash table)

            // we won't necessary produce output for every possible partition, so we
            // create writers on demand
            let mut writers: Vec<Option<SharedMemoryByteWriter>> = vec![];
            for _ in 0..num_output_partitions {
                writers.push(None);
            }

            let mut partitioner = BatchPartitioner::try_new(
                Partitioning::RoundRobinBatch(num_output_partitions),
                write_metrics.repart_time.clone(),
            )?;

            // TODO: we need a real cardinality estimation here.
            let estimated_size_per_partition = 1024 * 1024 * 512;
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
                                let mut idt = path.clone();
                                idt.push_str(&format!(
                                    "-{output_partition}-data-{input_partition}.arrow"
                                ));

                                debug!("Writing results to {:?}", idt);

                                let mut writer = SharedMemoryByteWriter::new(
                                    idt,
                                    stream.schema().as_ref(),
                                    estimated_size_per_partition,
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
                        info!(
                                "Finished writing shuffle partition {} at {:?}. Batches: {}. Rows: {}. Bytes: {}. Physical: {}",
                                i,
                                w.identifier(),
                                w.num_batches,
                                w.num_rows,
                                w.num_bytes,
                                w.get_physical_written_bytes()
                            );

                        let physical_size = w.get_physical_written_bytes();

                        part_locs.push(ShuffleWritePartition {
                            partition_id: i as u64,
                            path: w.identifier().to_owned(),
                            num_batches: w.num_batches,
                            num_rows: w.num_rows,
                            num_bytes: w.num_bytes,
                            physical_bytes: physical_size as u64,
                            ht_bucket_mask: 0,
                            ht_growth_left: 0,
                            ht_items: 0,
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

async fn execute_join_on_remote_shuffle_write(
    path: PathBuf,
    plan: Arc<dyn ExecutionPlan>,
    output_partitioning: Option<Partitioning>,
    write_metrics: ShuffleWriteMetrics,
    input_partition: usize,
    context: Arc<TaskContext>,
) -> Result<Vec<ShuffleWritePartition>> {
    let mut path = path.to_str().unwrap().to_owned();
    let now = Instant::now();
    let mut stream = plan.execute(input_partition, context)?;

    match output_partitioning {
        None => {
            let timer = write_metrics.write_time.timer();
            path.push_str(&format!("{input_partition}-data.arrow"));

            debug!("Writing results to {}", path);
            // stream results to disk
            let stats = utils::write_stream_to_disk(
                &mut stream,
                &path,
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
                path: path.to_owned(),
                num_batches: stats.num_batches.unwrap_or(0),
                num_rows: stats.num_rows.unwrap_or(0),
                num_bytes: stats.num_bytes.unwrap_or(0),
                physical_bytes: stats.physical_bytes.unwrap_or(0),
                ht_bucket_mask: 0,
                ht_growth_left: 0,
                ht_items: 0,
            }])
        }

        Some(Partitioning::Hash(exprs, num_output_partitions)) => {
            let mut idt = path.clone();
            idt.push_str(&format!("-jointable-data-{input_partition}"));

            debug!("Writing results to {:?}", idt);

            // TODO: we need a real cardinality estimation here.
            let estimated_size_per_partition = 1024 * 1024 * 512;

            let mut writer = SharedMemoryByteWriter::new(
                idt.clone(),
                stream.schema().as_ref(),
                estimated_size_per_partition,
            )?;

            let mut total_row_cnt = 0;

            while let Some(result) = stream.next().await {
                let input_batch = result?;
                write_metrics.input_rows.add(input_batch.num_rows());
                total_row_cnt += input_batch.num_rows();

                writer.write(&input_batch)?;
            }

            writer.finish()?;

            let physical_bytes = writer.get_physical_written_bytes();

            info!("Finished writing shuffle join table at {:?}. Batches: {}. Rows: {}. Bytes: {}. Physical: {}",
                    writer.identifier(),
                    writer.num_batches,
                    writer.num_rows,
                    writer.num_bytes,
                    physical_bytes
                );

            let reader = MemoryReader::new(writer.writer.writer.ptr, physical_bytes);
            let reader = NoBufReader::try_new(reader, None).unwrap();

            let mut hashes_buffer: Vec<u64> = Vec::new();
            let mut offset = 0;
            let random_state = RandomState::with_seeds(0, 0, 0, 0);

            let mut hashmap = JoinHashMap::with_capacity(total_row_cnt, idt);

            for b in reader {
                let input_batch = b?;
                hashes_buffer.clear();
                hashes_buffer.resize(input_batch.num_rows(), 0);

                let key_values = exprs
                    .iter()
                    .map(|c| {
                        Ok(c.evaluate(&input_batch)?
                            .into_array(input_batch.num_rows())
                            .unwrap())
                    })
                    .collect::<Result<Vec<_>>>()?;

                let hash_values =
                    create_hashes(&key_values, &random_state, &mut hashes_buffer)?;

                let (mut_map, mut_list) = hashmap.get_mut();
                for (row, hash_value) in hash_values.iter().enumerate() {
                    let item =
                        mut_map.get_mut(*hash_value, |(hash, _)| *hash_value == *hash);
                    if let Some((_, index)) = item {
                        // Already exists: add index to next array
                        let prev_index = *index;
                        // Store new value inside hashmap
                        *index = (row + offset + 1) as u64;
                        // Update chained Vec at row + offset with previous value
                        mut_list[row + offset] = prev_index;
                    } else {
                        mut_map.insert(
                            *hash_value,
                            // store the value + 1 as 0 value reserved for end of list
                            (*hash_value, (row + offset + 1) as u64),
                            |(hash, _)| *hash,
                        );
                        // chained list at (row + offset) is already initialized with 0
                        // meaning end of list
                    }
                }
                offset += input_batch.num_rows();
            }

            let ht_raw_parts = hashmap.into_raw_parts();

            let part0 = ShuffleWritePartition {
                partition_id: 0,
                path: writer.identifier().to_owned(),
                num_batches: writer.num_batches,
                num_rows: writer.num_rows,
                num_bytes: writer.num_bytes,
                physical_bytes: physical_bytes as u64,
                ht_bucket_mask: ht_raw_parts.1 as u64,
                ht_growth_left: ht_raw_parts.2 as u64,
                ht_items: ht_raw_parts.3 as u64,
            };

            let mut partitions = Vec::with_capacity(num_output_partitions);

            for i in 0..num_output_partitions as u64 {
                let mut p = part0.clone();
                p.partition_id = i;
                partitions.push(p);
            }

            Ok(partitions)
        }

        Some(Partitioning::RoundRobinBatch(_num_output_partitions)) => {
            Err(DataFusionError::Execution(
                "Roundrobin parition is no longer supported!".to_owned(),
            ))
        }

        _ => Err(DataFusionError::Execution(
            "Invalid shuffle partitioning scheme".to_owned(),
        )),
    }
}

impl ExecutionPlan for ShuffleWriterExec {
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
        Ok(Arc::new(ShuffleWriterExec::try_new(
            self.job_id.clone(),
            self.stage_id,
            children[0].clone(),
            self.work_dir.clone(),
            self.shuffle_output_partitioning.clone(),
            self.remote_mode,
            self.join_input_side,
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
                let mut physical_bytes_builder =
                    UInt64Builder::with_capacity(num_writers);
                let mut ht_bucket_mask_builder =
                    UInt64Builder::with_capacity(num_writers);
                let mut ht_growth_left_builder =
                    UInt64Builder::with_capacity(num_writers);
                let mut ht_items_builder = UInt64Builder::with_capacity(num_writers);

                for loc in &part_loc {
                    path_builder.append_value(loc.path.clone());
                    partition_builder.append_value(loc.partition_id as u32);
                    num_rows_builder.append_value(loc.num_rows);
                    num_batches_builder.append_value(loc.num_batches);
                    num_bytes_builder.append_value(loc.num_bytes);
                    physical_bytes_builder.append_value(loc.physical_bytes);
                    ht_bucket_mask_builder.append_value(loc.ht_bucket_mask);
                    ht_growth_left_builder.append_value(loc.ht_growth_left);
                    ht_items_builder.append_value(loc.ht_items);
                }

                // build arrays
                let partition_num: ArrayRef = Arc::new(partition_builder.finish());
                let path: ArrayRef = Arc::new(path_builder.finish());
                let field_builders: Vec<Box<dyn ArrayBuilder>> = vec![
                    Box::new(num_rows_builder),
                    Box::new(num_batches_builder),
                    Box::new(num_bytes_builder),
                    Box::new(physical_bytes_builder),
                    Box::new(ht_bucket_mask_builder),
                    Box::new(ht_growth_left_builder),
                    Box::new(ht_items_builder),
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

    fn statistics(&self) -> Result<Statistics> {
        self.plan.statistics()
    }
}

impl DisplayAs for ShuffleWriterExec {
    fn fmt_as(
        &self,
        t: DisplayFormatType,
        f: &mut std::fmt::Formatter,
    ) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(
                    f,
                    "ShuffleWriterExec: {:?}, mode: {:?}, side: {:?}",
                    self.shuffle_output_partitioning,
                    self.remote_memory_mode(),
                    self.join_input_side()
                )
            }
        }
    }
}

pub(super) fn result_schema() -> SchemaRef {
    let stats = PartitionStats::default();
    Arc::new(Schema::new(vec![
        Field::new("partition", DataType::UInt32, false),
        Field::new("path", DataType::Utf8, false),
        stats.arrow_struct_repr(),
    ]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::arrow::array::{StringArray, StructArray, UInt32Array, UInt64Array};
    use datafusion::physical_plan::coalesce_partitions::CoalescePartitionsExec;
    use datafusion::physical_plan::expressions::Column;

    use datafusion::physical_plan::memory::MemoryExec;
    use datafusion::prelude::SessionContext;
    use tempfile::TempDir;

    #[tokio::test]
    // number of rows in each partition is a function of the hash output, so don't test here
    #[cfg(not(feature = "force_hash_collisions"))]
    async fn test() -> Result<()> {
        let session_ctx = SessionContext::new();
        let task_ctx = session_ctx.task_ctx();

        let input_plan = Arc::new(CoalescePartitionsExec::new(create_input_plan()?));
        let work_dir = TempDir::new()?;
        let query_stage = ShuffleWriterExec::try_new(
            "jobOne".to_owned(),
            1,
            input_plan,
            work_dir.into_path().to_str().unwrap().to_owned(),
            Some(Partitioning::Hash(vec![Arc::new(Column::new("a", 0))], 2)),
            RemoteMemoryMode::default(),
            JoinInputSide::NotApplicable,
        )?;
        let mut stream = query_stage.execute(0, task_ctx)?;
        let batches = utils::collect_stream(&mut stream)
            .await
            .map_err(|e| DataFusionError::Execution(format!("{e:?}")))?;

        println!("{}", query_stage.metrics().unwrap());
        assert_eq!(1, batches.len());
        let batch = &batches[0];
        assert_eq!(3, batch.num_columns());
        assert_eq!(2, batch.num_rows());
        let path = batch.columns()[1]
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();

        let file0 = path.value(0);
        assert!(
            file0.ends_with("/jobOne/1/0/data-0.arrow")
                || file0.ends_with("\\jobOne\\1\\0\\data-0.arrow")
        );
        let file1 = path.value(1);
        assert!(
            file1.ends_with("/jobOne/1/1/data-0.arrow")
                || file1.ends_with("\\jobOne\\1\\1\\data-0.arrow")
        );

        let stats = batch.columns()[2]
            .as_any()
            .downcast_ref::<StructArray>()
            .unwrap();

        let num_rows = stats
            .column_by_name("num_rows")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        assert_eq!(4, num_rows.value(0));
        assert_eq!(4, num_rows.value(1));

        Ok(())
    }

    #[tokio::test]
    // number of rows in each partition is a function of the hash output, so don't test here
    #[cfg(not(feature = "force_hash_collisions"))]
    async fn test_partitioned() -> Result<()> {
        let session_ctx = SessionContext::new();
        let task_ctx = session_ctx.task_ctx();

        let input_plan = create_input_plan()?;
        let work_dir = TempDir::new()?;
        let query_stage = ShuffleWriterExec::try_new(
            "jobOne".to_owned(),
            1,
            input_plan,
            work_dir.into_path().to_str().unwrap().to_owned(),
            Some(Partitioning::Hash(vec![Arc::new(Column::new("a", 0))], 2)),
            RemoteMemoryMode::default(),
            JoinInputSide::NotApplicable,
        )?;
        let mut stream = query_stage.execute(0, task_ctx)?;
        let batches = utils::collect_stream(&mut stream)
            .await
            .map_err(|e| DataFusionError::Execution(format!("{e:?}")))?;

        assert_eq!(1, batches.len());
        let batch = &batches[0];
        assert_eq!(3, batch.num_columns());
        assert_eq!(2, batch.num_rows());
        let stats = batch.columns()[2]
            .as_any()
            .downcast_ref::<StructArray>()
            .unwrap();
        let num_rows = stats
            .column_by_name("num_rows")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        assert_eq!(2, num_rows.value(0));
        assert_eq!(2, num_rows.value(1));

        Ok(())
    }

    fn create_input_plan() -> Result<Arc<dyn ExecutionPlan>> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::UInt32, true),
            Field::new("b", DataType::Utf8, true),
        ]));

        // define data.
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt32Array::from(vec![Some(1), Some(2)])),
                Arc::new(StringArray::from(vec![Some("hello"), Some("world")])),
            ],
        )?;
        let partition = vec![batch.clone(), batch];
        let partitions = vec![partition.clone(), partition];
        Ok(Arc::new(MemoryExec::try_new(&partitions, schema, None)?))
    }
}
