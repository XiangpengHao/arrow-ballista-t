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

//! This module contains execution plans that are needed to distribute DataFusion's execution plans into
//! several Ballista executors.

mod distributed_query;
mod rm_shuffle_reader;
mod rm_shuffle_writer;
mod shuffle_reader;
mod shuffle_writer;
mod sm_writer;
mod unresolved_shuffle;

use datafusion::error::Result;
use datafusion::physical_plan::{ExecutionPlan, Partitioning};
pub use rm_shuffle_reader::RemoteShuffleReaderExec;
pub use rm_shuffle_writer::RemoteShuffleWriterExec;
use std::sync::Arc;

pub use distributed_query::DistributedQueryExec;
pub use shuffle_reader::ShuffleReaderExec;
pub use shuffle_writer::ShuffleWriterExec;
pub use unresolved_shuffle::UnresolvedShuffleExec;

pub trait ShuffleWriter: Sized + ExecutionPlan {
    fn job_id(&self) -> &str;

    fn stage_id(&self) -> usize;

    fn use_remote_memory() -> bool;

    /// Get the true output partitioning
    fn shuffle_output_partitioning(&self) -> Option<&Partitioning>;

    /// Create a new shuffle writer
    fn try_new(
        job_id: String,
        stage_id: usize,
        plan: Arc<dyn ExecutionPlan>,
        work_dir: String,
        shuffle_output_partitioning: Option<Partitioning>,
    ) -> Result<Self>;
}
