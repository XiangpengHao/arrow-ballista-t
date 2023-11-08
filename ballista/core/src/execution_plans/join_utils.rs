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

//! This file contains common subroutines for regular and symmetric hash join
//! related functionality, used both in join calculations and optimization rules.

use std::fmt::Debug;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::{fmt, usize};

use datafusion::arrow::array::{
    downcast_array, new_null_array, Array, BooleanBufferBuilder, UInt32Array,
    UInt32Builder, UInt64Array,
};
use datafusion::arrow::compute;
use datafusion::arrow::datatypes::Schema;
use datafusion::arrow::record_batch::{RecordBatch, RecordBatchOptions};
use datafusion::common::cast::as_boolean_array;
use datafusion::common::{DataFusionError, Result};
use datafusion::error::SharedResult;

use datafusion::physical_plan::joins::utils::{
    ColumnIndex, JoinFilter, JoinOn, JoinSide,
};
use datafusion::physical_plan::metrics::{self, ExecutionPlanMetricsSet, MetricBuilder};
use datafusion::physical_plan::{ExecutionPlan, Statistics};
use datafusion::prelude::JoinType;
use futures::future::{BoxFuture, Shared};
use futures::{ready, Future, FutureExt};
use hashbrown::raw::RawTable;
use parking_lot::Mutex;

// Maps a `u64` hash value based on the build side ["on" values] to a list of indices with this key's value.
// By allocating a `HashMap` with capacity for *at least* the number of rows for entries at the build side,
// we make sure that we don't have to re-hash the hashmap, which needs access to the key (the hash in this case) value.
// E.g. 1 -> [3, 6, 8] indicates that the column values map to rows 3, 6 and 8 for hash value 1
// As the key is a hash value, we need to check possible hash collisions in the probe stage
// During this stage it might be the case that a row is contained the same hashmap value,
// but the values don't match. Those are checked in the [equal_rows] macro
// The indices (values) are stored in a separate chained list stored in the `Vec<u64>`.
// The first value (+1) is stored in the hashmap, whereas the next value is stored in array at the position value.
// The chain can be followed until the value "0" has been reached, meaning the end of the list.
// Also see chapter 5.3 of [Balancing vectorized query execution with bandwidth-optimized storage](https://dare.uva.nl/search?identifier=5ccbb60a-38b8-4eeb-858a-e7735dd37487)
// See the example below:
// Insert (1,1)
// map:
// ---------
// | 1 | 2 |
// ---------
// next:
// ---------------------
// | 0 | 0 | 0 | 0 | 0 |
// ---------------------
// Insert (2,2)
// map:
// ---------
// | 1 | 2 |
// | 2 | 3 |
// ---------
// next:
// ---------------------
// | 0 | 0 | 0 | 0 | 0 |
// ---------------------
// Insert (1,3)
// map:
// ---------
// | 1 | 4 |
// | 2 | 3 |
// ---------
// next:
// ---------------------
// | 0 | 0 | 0 | 2 | 0 |  <--- hash value 1 maps to 4,2 (which means indices values 3,1)
// ---------------------
// Insert (1,4)
// map:
// ---------
// | 1 | 5 |
// | 2 | 3 |
// ---------
// next:
// ---------------------
// | 0 | 0 | 0 | 2 | 4 | <--- hash value 1 maps to 5,4,2 (which means indices values 4,3,1)
// ---------------------
// TODO: speed up collision checks
// https://github.com/apache/arrow-datafusion/issues/50
pub struct JoinHashMap {
    // Stores hash value to last row index
    pub map: RawTable<(u64, u64), douhua::RemoteAlloc>,
    // Stores indices in chained list data structure
    pub next: Vec<u64>,
}

impl JoinHashMap {
    pub(crate) fn with_capacity(capacity: usize) -> Self {
        JoinHashMap {
            map: RawTable::with_capacity_in(capacity, douhua::RemoteAlloc::new()),
            next: vec![0; capacity],
        }
    }
}

/// Implementation of `JoinHashMapType` for `JoinHashMap`.
impl JoinHashMap {
    // Void implementation
    pub(crate) fn extend_zero(&mut self, _: usize) {}

    /// Get mutable references to the hash map and the next.
    pub(crate) fn get_mut(
        &mut self,
    ) -> (
        &mut RawTable<(u64, u64), douhua::RemoteAlloc>,
        &mut Vec<u64>,
    ) {
        (&mut self.map, &mut self.next)
    }

    /// Get a reference to the hash map.
    pub(crate) fn get_map(&self) -> &RawTable<(u64, u64), douhua::RemoteAlloc> {
        &self.map
    }

    /// Get a reference to the next.
    pub(crate) fn get_list(&self) -> &Vec<u64> {
        &self.next
    }
}

impl fmt::Debug for JoinHashMap {
    fn fmt(&self, _f: &mut fmt::Formatter) -> fmt::Result {
        Ok(())
    }
}

/// Some type `join_type` of join need to maintain the matched indices bit map for the left side, and
/// use the bit map to generate the part of result of the join.
///
/// For example of the `Left` join, in each iteration of right side, can get the matched result, but need
/// to maintain the matched indices bit map to get the unmatched row for the left side.
pub(crate) fn need_produce_result_in_final(join_type: JoinType) -> bool {
    matches!(
        join_type,
        JoinType::Left | JoinType::LeftAnti | JoinType::LeftSemi | JoinType::Full
    )
}

pub(crate) fn apply_join_filter_to_indices(
    build_input_buffer: &RecordBatch,
    probe_batch: &RecordBatch,
    build_indices: UInt64Array,
    probe_indices: UInt32Array,
    filter: &JoinFilter,
    build_side: JoinSide,
) -> Result<(UInt64Array, UInt32Array)> {
    if build_indices.is_empty() && probe_indices.is_empty() {
        return Ok((build_indices, probe_indices));
    };

    let intermediate_batch = build_batch_from_indices(
        filter.schema(),
        build_input_buffer,
        probe_batch,
        &build_indices,
        &probe_indices,
        filter.column_indices(),
        build_side,
    )?;
    let filter_result = filter
        .expression()
        .evaluate(&intermediate_batch)?
        .into_array(intermediate_batch.num_rows());
    let mask = as_boolean_array(&filter_result)?;

    let left_filtered = compute::filter(&build_indices, mask)?;
    let right_filtered = compute::filter(&probe_indices, mask)?;
    Ok((
        downcast_array(left_filtered.as_ref()),
        downcast_array(right_filtered.as_ref()),
    ))
}

/// Returns a new [RecordBatch] by combining the `left` and `right` according to `indices`.
/// The resulting batch has [Schema] `schema`.
pub(crate) fn build_batch_from_indices(
    schema: &Schema,
    build_input_buffer: &RecordBatch,
    probe_batch: &RecordBatch,
    build_indices: &UInt64Array,
    probe_indices: &UInt32Array,
    column_indices: &[ColumnIndex],
    build_side: JoinSide,
) -> Result<RecordBatch> {
    if schema.fields().is_empty() {
        let options = RecordBatchOptions::new()
            .with_match_field_names(true)
            .with_row_count(Some(build_indices.len()));

        return Ok(RecordBatch::try_new_with_options(
            Arc::new(schema.clone()),
            vec![],
            &options,
        )?);
    }

    // build the columns of the new [RecordBatch]:
    // 1. pick whether the column is from the left or right
    // 2. based on the pick, `take` items from the different RecordBatches
    let mut columns: Vec<Arc<dyn Array>> = Vec::with_capacity(schema.fields().len());

    for column_index in column_indices {
        let array = if column_index.side == build_side {
            let array = build_input_buffer.column(column_index.index);
            if array.is_empty() || build_indices.null_count() == build_indices.len() {
                // Outer join would generate a null index when finding no match at our side.
                // Therefore, it's possible we are empty but need to populate an n-length null array,
                // where n is the length of the index array.
                assert_eq!(build_indices.null_count(), build_indices.len());
                new_null_array(array.data_type(), build_indices.len())
            } else {
                compute::take(array.as_ref(), build_indices, None)?
            }
        } else {
            let array = probe_batch.column(column_index.index);
            if array.is_empty() || probe_indices.null_count() == probe_indices.len() {
                assert_eq!(probe_indices.null_count(), probe_indices.len());
                new_null_array(array.data_type(), probe_indices.len())
            } else {
                compute::take(array.as_ref(), probe_indices, None)?
            }
        };
        columns.push(array);
    }
    Ok(RecordBatch::try_new(Arc::new(schema.clone()), columns)?)
}

/// The input is the matched indices for left and right and
/// adjust the indices according to the join type
pub(crate) fn adjust_indices_by_join_type(
    left_indices: UInt64Array,
    right_indices: UInt32Array,
    count_right_batch: usize,
    join_type: JoinType,
) -> (UInt64Array, UInt32Array) {
    match join_type {
        JoinType::Inner => {
            // matched
            (left_indices, right_indices)
        }
        JoinType::Left => {
            // matched
            (left_indices, right_indices)
            // unmatched left row will be produced in the end of loop, and it has been set in the left visited bitmap
        }
        JoinType::Right | JoinType::Full => {
            // matched
            // unmatched right row will be produced in this batch
            let right_unmatched_indices =
                get_anti_indices(count_right_batch, &right_indices);
            // combine the matched and unmatched right result together
            append_right_indices(left_indices, right_indices, right_unmatched_indices)
        }
        JoinType::RightSemi => {
            // need to remove the duplicated record in the right side
            let right_indices = get_semi_indices(count_right_batch, &right_indices);
            // the left_indices will not be used later for the `right semi` join
            (left_indices, right_indices)
        }
        JoinType::RightAnti => {
            // need to remove the duplicated record in the right side
            // get the anti index for the right side
            let right_indices = get_anti_indices(count_right_batch, &right_indices);
            // the left_indices will not be used later for the `right anti` join
            (left_indices, right_indices)
        }
        JoinType::LeftSemi | JoinType::LeftAnti => {
            // matched or unmatched left row will be produced in the end of loop
            // When visit the right batch, we can output the matched left row and don't need to wait the end of loop
            (
                UInt64Array::from_iter_values(vec![]),
                UInt32Array::from_iter_values(vec![]),
            )
        }
    }
}

/// Get unmatched and deduplicated indices
pub(crate) fn get_anti_indices(
    row_count: usize,
    input_indices: &UInt32Array,
) -> UInt32Array {
    let mut bitmap = BooleanBufferBuilder::new(row_count);
    bitmap.append_n(row_count, false);
    input_indices.iter().flatten().for_each(|v| {
        bitmap.set_bit(v as usize, true);
    });

    // get the anti index
    (0..row_count)
        .filter_map(|idx| (!bitmap.get_bit(idx)).then_some(idx as u32))
        .collect::<UInt32Array>()
}

/// Appends the `right_unmatched_indices` to the `right_indices`,
/// and fills Null to tail of `left_indices` to
/// keep the length of `right_indices` and `left_indices` consistent.
pub(crate) fn append_right_indices(
    left_indices: UInt64Array,
    right_indices: UInt32Array,
    right_unmatched_indices: UInt32Array,
) -> (UInt64Array, UInt32Array) {
    // left_indices, right_indices and right_unmatched_indices must not contain the null value
    if right_unmatched_indices.is_empty() {
        (left_indices, right_indices)
    } else {
        let unmatched_size = right_unmatched_indices.len();
        // the new left indices: left_indices + null array
        // the new right indices: right_indices + right_unmatched_indices
        let new_left_indices = left_indices
            .iter()
            .chain(std::iter::repeat(None).take(unmatched_size))
            .collect::<UInt64Array>();
        let new_right_indices = right_indices
            .iter()
            .chain(right_unmatched_indices.iter())
            .collect::<UInt32Array>();
        (new_left_indices, new_right_indices)
    }
}

/// Get matched and deduplicated indices
pub(crate) fn get_semi_indices(
    row_count: usize,
    input_indices: &UInt32Array,
) -> UInt32Array {
    let mut bitmap = BooleanBufferBuilder::new(row_count);
    bitmap.append_n(row_count, false);
    input_indices.iter().flatten().for_each(|v| {
        bitmap.set_bit(v as usize, true);
    });

    // get the semi index
    (0..row_count)
        .filter_map(|idx| (bitmap.get_bit(idx)).then_some(idx as u32))
        .collect::<UInt32Array>()
}

/// In the end of join execution, need to use bit map of the matched
/// indices to generate the final left and right indices.
///
/// For example:
///
/// 1. left_bit_map: `[true, false, true, true, false]`
/// 2. join_type: `Left`
///
/// The result is: `([1,4], [null, null])`
pub(crate) fn get_final_indices_from_bit_map(
    left_bit_map: &BooleanBufferBuilder,
    join_type: JoinType,
) -> (UInt64Array, UInt32Array) {
    let left_size = left_bit_map.len();
    let left_indices = if join_type == JoinType::LeftSemi {
        (0..left_size)
            .filter_map(|idx| (left_bit_map.get_bit(idx)).then_some(idx as u64))
            .collect::<UInt64Array>()
    } else {
        // just for `Left`, `LeftAnti` and `Full` join
        // `LeftAnti`, `Left` and `Full` will produce the unmatched left row finally
        (0..left_size)
            .filter_map(|idx| (!left_bit_map.get_bit(idx)).then_some(idx as u64))
            .collect::<UInt64Array>()
    };
    // right_indices
    // all the element in the right side is None
    let mut builder = UInt32Builder::with_capacity(left_indices.len());
    builder.append_nulls(left_indices.len());
    let right_indices = builder.finish();
    (left_indices, right_indices)
}

/// A [`OnceAsync`] can be used to run an async closure once, with subsequent calls
/// to [`OnceAsync::once`] returning a [`OnceFut`] to the same asynchronous computation
///
/// This is useful for joins where the results of one child are buffered in memory
/// and shared across potentially multiple output partitions
pub(crate) struct OnceAsync<T> {
    fut: Mutex<Option<OnceFut<T>>>,
}

impl<T> Default for OnceAsync<T> {
    fn default() -> Self {
        Self {
            fut: Mutex::new(None),
        }
    }
}

impl<T> std::fmt::Debug for OnceAsync<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "OnceAsync")
    }
}

impl<T: 'static> OnceAsync<T> {
    /// If this is the first call to this function on this object, will invoke
    /// `f` to obtain a future and return a [`OnceFut`] referring to this
    ///
    /// If this is not the first call, will return a [`OnceFut`] referring
    /// to the same future as was returned by the first call
    pub(crate) fn once<F, Fut>(&self, f: F) -> OnceFut<T>
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = Result<T>> + Send + 'static,
    {
        self.fut
            .lock()
            .get_or_insert_with(|| OnceFut::new(f()))
            .clone()
    }
}

/// The shared future type used internally within [`OnceAsync`]
type OnceFutPending<T> = Shared<BoxFuture<'static, SharedResult<Arc<T>>>>;

/// A [`OnceFut`] represents a shared asynchronous computation, that will be evaluated
/// once for all [`Clone`]'s, with [`OnceFut::get`] providing a non-consuming interface
/// to drive the underlying [`Future`] to completion
pub(crate) struct OnceFut<T> {
    state: OnceFutState<T>,
}

impl<T> Clone for OnceFut<T> {
    fn clone(&self) -> Self {
        Self {
            state: self.state.clone(),
        }
    }
}

enum OnceFutState<T> {
    Pending(OnceFutPending<T>),
    Ready(SharedResult<Arc<T>>),
}

impl<T> Clone for OnceFutState<T> {
    fn clone(&self) -> Self {
        match self {
            Self::Pending(p) => Self::Pending(p.clone()),
            Self::Ready(r) => Self::Ready(r.clone()),
        }
    }
}

impl<T: 'static> OnceFut<T> {
    /// Create a new [`OnceFut`] from a [`Future`]
    pub(crate) fn new<Fut>(fut: Fut) -> Self
    where
        Fut: Future<Output = Result<T>> + Send + 'static,
    {
        Self {
            state: OnceFutState::Pending(
                fut.map(|res| res.map(Arc::new).map_err(Arc::new))
                    .boxed()
                    .shared(),
            ),
        }
    }

    /// Get the result of the computation if it is ready, without consuming it
    pub(crate) fn get(&mut self, cx: &mut Context<'_>) -> Poll<Result<&T>> {
        if let OnceFutState::Pending(fut) = &mut self.state {
            let r = ready!(fut.poll_unpin(cx));
            self.state = OnceFutState::Ready(r);
        }

        // Cannot use loop as this would trip up the borrow checker
        match &self.state {
            OnceFutState::Pending(_) => unreachable!(),
            OnceFutState::Ready(r) => Poll::Ready(
                r.as_ref()
                    .map(|r| r.as_ref())
                    .map_err(|e| DataFusionError::External(Box::new(e.clone()))),
            ),
        }
    }
}

/// Metrics for build & probe joins
#[derive(Clone, Debug)]
pub(crate) struct BuildProbeJoinMetrics {
    /// Total time for collecting build-side of join
    pub(crate) build_time: metrics::Time,
    /// Number of batches consumed by build-side
    pub(crate) build_input_batches: metrics::Count,
    /// Number of rows consumed by build-side
    pub(crate) build_input_rows: metrics::Count,
    /// Memory used by build-side in bytes
    pub(crate) build_mem_used: metrics::Gauge,
    /// Total time for joining probe-side batches to the build-side batches
    pub(crate) join_time: metrics::Time,
    /// Number of batches consumed by probe-side of this operator
    pub(crate) input_batches: metrics::Count,
    /// Number of rows consumed by probe-side this operator
    pub(crate) input_rows: metrics::Count,
    /// Number of batches produced by this operator
    pub(crate) output_batches: metrics::Count,
    /// Number of rows produced by this operator
    pub(crate) output_rows: metrics::Count,
}

impl BuildProbeJoinMetrics {
    pub fn new(partition: usize, metrics: &ExecutionPlanMetricsSet) -> Self {
        let join_time = MetricBuilder::new(metrics).subset_time("join_time", partition);

        let build_time = MetricBuilder::new(metrics).subset_time("build_time", partition);

        let build_input_batches =
            MetricBuilder::new(metrics).counter("build_input_batches", partition);

        let build_input_rows =
            MetricBuilder::new(metrics).counter("build_input_rows", partition);

        let build_mem_used =
            MetricBuilder::new(metrics).gauge("build_mem_used", partition);

        let input_batches =
            MetricBuilder::new(metrics).counter("input_batches", partition);

        let input_rows = MetricBuilder::new(metrics).counter("input_rows", partition);

        let output_batches =
            MetricBuilder::new(metrics).counter("output_batches", partition);

        let output_rows = MetricBuilder::new(metrics).output_rows(partition);

        Self {
            build_time,
            build_input_batches,
            build_input_rows,
            build_mem_used,
            join_time,
            input_batches,
            input_rows,
            output_batches,
            output_rows,
        }
    }
}

/// Estimate the statistics for the given join's output.
pub(crate) fn estimate_join_statistics(
    left: Arc<dyn ExecutionPlan>,
    _right: Arc<dyn ExecutionPlan>,
    _on: JoinOn,
    _join_type: &JoinType,
    _schema: &Schema,
) -> Statistics {
    // TODO: this is not correct
    left.statistics()
}
