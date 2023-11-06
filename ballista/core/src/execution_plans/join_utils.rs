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

use std::collections::{HashMap, VecDeque};
use std::fmt::Debug;
use std::ops::IndexMut;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::{fmt, usize};

use arrow::array::{ArrowPrimitiveType, NativeAdapter, PrimitiveArray};
use arrow::compute::concat_batches;
use arrow::datatypes::{ArrowNativeType, SchemaRef};
use datafusion::arrow::array::{
    downcast_array, new_null_array, Array, BooleanBufferBuilder, UInt32Array,
    UInt32Builder, UInt64Array,
};
use datafusion::arrow::datatypes::Schema;
use datafusion::arrow::record_batch::{RecordBatch, RecordBatchOptions};
use datafusion::arrow::{self, compute};
use datafusion::common::{
    cast::as_boolean_array,
    tree_node::{Transformed, TreeNode},
};
use datafusion::common::{DataFusionError, Result, ScalarValue};
use datafusion::error::SharedResult;
use datafusion::physical_expr::expressions::Column;
use datafusion::physical_expr::intervals::{Interval, IntervalBound};
use datafusion::physical_expr::utils::collect_columns;
use datafusion::physical_expr::{PhysicalExpr, PhysicalSortExpr};

use datafusion::physical_plan::joins::utils::{
    ColumnIndex, JoinFilter, JoinOn, JoinSide,
};
use datafusion::physical_plan::metrics::{self, ExecutionPlanMetricsSet, MetricBuilder};
use datafusion::physical_plan::{ExecutionPlan, Statistics};
use datafusion::prelude::JoinType;
use futures::future::{BoxFuture, Shared};
use futures::{ready, Future, FutureExt};
use hashbrown::raw::RawTable;
use hashbrown::HashSet;
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
    pub map: RawTable<(u64, u64)>,
    // Stores indices in chained list data structure
    pub next: Vec<u64>,
}

impl JoinHashMap {
    pub(crate) fn with_capacity(capacity: usize) -> Self {
        JoinHashMap {
            map: RawTable::with_capacity(capacity),
            next: vec![0; capacity],
        }
    }
}

/// Trait defining methods that must be implemented by a hash map type to be used for joins.
pub trait JoinHashMapType {
    /// The type of list used to store the hash values.
    type NextType: IndexMut<usize, Output = u64>;
    /// Extend with zero
    fn extend_zero(&mut self, len: usize);
    /// Returns mutable references to the hash map and the next.
    fn get_mut(&mut self) -> (&mut RawTable<(u64, u64)>, &mut Self::NextType);
    /// Returns a reference to the hash map.
    fn get_map(&self) -> &RawTable<(u64, u64)>;
    /// Returns a reference to the next.
    fn get_list(&self) -> &Self::NextType;
}

/// Implementation of `JoinHashMapType` for `JoinHashMap`.
impl JoinHashMapType for JoinHashMap {
    type NextType = Vec<u64>;

    // Void implementation
    fn extend_zero(&mut self, _: usize) {}

    /// Get mutable references to the hash map and the next.
    fn get_mut(&mut self) -> (&mut RawTable<(u64, u64)>, &mut Self::NextType) {
        (&mut self.map, &mut self.next)
    }

    /// Get a reference to the hash map.
    fn get_map(&self) -> &RawTable<(u64, u64)> {
        &self.map
    }

    /// Get a reference to the next.
    fn get_list(&self) -> &Self::NextType {
        &self.next
    }
}

/// Implementation of `JoinHashMapType` for `PruningJoinHashMap`.
impl JoinHashMapType for PruningJoinHashMap {
    type NextType = VecDeque<u64>;

    // Extend with zero
    fn extend_zero(&mut self, len: usize) {
        self.next.resize(self.next.len() + len, 0)
    }

    /// Get mutable references to the hash map and the next.
    fn get_mut(&mut self) -> (&mut RawTable<(u64, u64)>, &mut Self::NextType) {
        (&mut self.map, &mut self.next)
    }

    /// Get a reference to the hash map.
    fn get_map(&self) -> &RawTable<(u64, u64)> {
        &self.map
    }

    /// Get a reference to the next.
    fn get_list(&self) -> &Self::NextType {
        &self.next
    }
}

impl fmt::Debug for JoinHashMap {
    fn fmt(&self, _f: &mut fmt::Formatter) -> fmt::Result {
        Ok(())
    }
}

/// The `PruningJoinHashMap` is similar to a regular `JoinHashMap`, but with
/// the capability of pruning elements in an efficient manner. This structure
/// is particularly useful for cases where it's necessary to remove elements
/// from the map based on their buffer order.
///
/// # Example
///
/// ``` text
/// Let's continue the example of `JoinHashMap` and then show how `PruningJoinHashMap` would
/// handle the pruning scenario.
///
/// Insert the pair (1,4) into the `PruningJoinHashMap`:
/// map:
/// ---------
/// | 1 | 5 |
/// | 2 | 3 |
/// ---------
/// list:
/// ---------------------
/// | 0 | 0 | 0 | 2 | 4 | <--- hash value 1 maps to 5,4,2 (which means indices values 4,3,1)
/// ---------------------
///
/// Now, let's prune 3 rows from `PruningJoinHashMap`:
/// map:
/// ---------
/// | 1 | 5 |
/// ---------
/// list:
/// ---------
/// | 2 | 4 | <--- hash value 1 maps to 2 (5 - 3), 1 (4 - 3), NA (2 - 3) (which means indices values 1,0)
/// ---------
///
/// After pruning, the | 2 | 3 | entry is deleted from `PruningJoinHashMap` since
/// there are no values left for this key.
/// ```
pub struct PruningJoinHashMap {
    /// Stores hash value to last row index
    pub map: RawTable<(u64, u64)>,
    /// Stores indices in chained list data structure
    pub next: VecDeque<u64>,
}

impl PruningJoinHashMap {
    /// Constructs a new `PruningJoinHashMap` with the given capacity.
    /// Both the map and the list are pre-allocated with the provided capacity.
    ///
    /// # Arguments
    /// * `capacity`: The initial capacity of the hash map.
    ///
    /// # Returns
    /// A new instance of `PruningJoinHashMap`.
    pub(crate) fn with_capacity(capacity: usize) -> Self {
        PruningJoinHashMap {
            map: RawTable::with_capacity(capacity),
            next: VecDeque::with_capacity(capacity),
        }
    }

    /// Shrinks the capacity of the hash map, if necessary, based on the
    /// provided scale factor.
    ///
    /// # Arguments
    /// * `scale_factor`: The scale factor that determines how conservative the
    ///   shrinking strategy is. The capacity will be reduced by 1/`scale_factor`
    ///   when necessary.
    ///
    /// # Note
    /// Increasing the scale factor results in less aggressive capacity shrinking,
    /// leading to potentially higher memory usage but fewer resizes. Conversely,
    /// decreasing the scale factor results in more aggressive capacity shrinking,
    /// potentially leading to lower memory usage but more frequent resizing.
    pub(crate) fn shrink_if_necessary(&mut self, scale_factor: usize) {
        let capacity = self.map.capacity();

        if capacity > scale_factor * self.map.len() {
            let new_capacity = (capacity * (scale_factor - 1)) / scale_factor;
            // Resize the map with the new capacity.
            self.map.shrink_to(new_capacity, |(hash, _)| *hash)
        }
    }

    /// Calculates the size of the `PruningJoinHashMap` in bytes.
    ///
    /// # Returns
    /// The size of the hash map in bytes.
    pub(crate) fn size(&self) -> usize {
        self.map.allocation_info().1.size()
            + self.next.capacity() * std::mem::size_of::<u64>()
    }

    /// Removes hash values from the map and the list based on the given pruning
    /// length and deleting offset.
    ///
    /// # Arguments
    /// * `prune_length`: The number of elements to remove from the list.
    /// * `deleting_offset`: The offset used to determine which hash values to remove from the map.
    ///
    /// # Returns
    /// A `Result` indicating whether the operation was successful.
    pub(crate) fn prune_hash_values(
        &mut self,
        prune_length: usize,
        deleting_offset: u64,
        shrink_factor: usize,
    ) -> Result<()> {
        // Remove elements from the list based on the pruning length.
        self.next.drain(0..prune_length);

        // Calculate the keys that should be removed from the map.
        let removable_keys = unsafe {
            self.map
                .iter()
                .map(|bucket| bucket.as_ref())
                .filter_map(|(hash, tail_index)| {
                    (*tail_index < prune_length as u64 + deleting_offset).then_some(*hash)
                })
                .collect::<Vec<_>>()
        };

        // Remove the keys from the map.
        removable_keys.into_iter().for_each(|hash_value| {
            self.map
                .remove_entry(hash_value, |(hash, _)| hash_value == *hash);
        });

        // Shrink the map if necessary.
        self.shrink_if_necessary(shrink_factor);
        Ok(())
    }
}

fn check_filter_expr_contains_sort_information(
    expr: &Arc<dyn PhysicalExpr>,
    reference: &Arc<dyn PhysicalExpr>,
) -> bool {
    expr.eq(reference)
        || expr
            .children()
            .iter()
            .any(|e| check_filter_expr_contains_sort_information(e, reference))
}

/// Create a one to one mapping from main columns to filter columns using
/// filter column indices. A column index looks like:
/// ```text
/// ColumnIndex {
///     index: 0, // field index in main schema
///     side: JoinSide::Left, // child side
/// }
/// ```
pub fn map_origin_col_to_filter_col(
    filter: &JoinFilter,
    schema: &SchemaRef,
    side: &JoinSide,
) -> Result<HashMap<Column, Column>> {
    let filter_schema = filter.schema();
    let mut col_to_col_map: HashMap<Column, Column> = HashMap::new();
    for (filter_schema_index, index) in filter.column_indices().iter().enumerate() {
        if index.side.eq(side) {
            // Get the main field from column index:
            let main_field = schema.field(index.index);
            // Create a column expression:
            let main_col = Column::new_with_schema(main_field.name(), schema.as_ref())?;
            // Since the order of by filter.column_indices() is the same with
            // that of intermediate schema fields, we can get the column directly.
            let filter_field = filter_schema.field(filter_schema_index);
            let filter_col = Column::new(filter_field.name(), filter_schema_index);
            // Insert mapping:
            col_to_col_map.insert(main_col, filter_col);
        }
    }
    Ok(col_to_col_map)
}

/// This function analyzes [`PhysicalSortExpr`] graphs with respect to monotonicity
/// (sorting) properties. This is necessary since monotonically increasing and/or
/// decreasing expressions are required when using join filter expressions for
/// data pruning purposes.
///
/// The method works as follows:
/// 1. Maps the original columns to the filter columns using the [`map_origin_col_to_filter_col`] function.
/// 2. Collects all columns in the sort expression using the [`collect_columns`] function.
/// 3. Checks if all columns are included in the map we obtain in the first step.
/// 4. If all columns are included, the sort expression is converted into a filter expression using
///    the [`convert_filter_columns`] function.
/// 5. Searches for the converted filter expression in the filter expression using the
///    [`check_filter_expr_contains_sort_information`] function.
/// 6. If an exact match is found, returns the converted filter expression as [`Some(Arc<dyn PhysicalExpr>)`].
/// 7. If all columns are not included or an exact match is not found, returns [`None`].
///
/// Examples:
/// Consider the filter expression "a + b > c + 10 AND a + b < c + 100".
/// 1. If the expression "a@ + d@" is sorted, it will not be accepted since the "d@" column is not part of the filter.
/// 2. If the expression "d@" is sorted, it will not be accepted since the "d@" column is not part of the filter.
/// 3. If the expression "a@ + b@ + c@" is sorted, all columns are represented in the filter expression. However,
///    there is no exact match, so this expression does not indicate pruning.
pub fn convert_sort_expr_with_filter_schema(
    side: &JoinSide,
    filter: &JoinFilter,
    schema: &SchemaRef,
    sort_expr: &PhysicalSortExpr,
) -> Result<Option<Arc<dyn PhysicalExpr>>> {
    let column_map = map_origin_col_to_filter_col(filter, schema, side)?;
    let expr = sort_expr.expr.clone();
    // Get main schema columns:
    let expr_columns = collect_columns(&expr);
    // Calculation is possible with `column_map` since sort exprs belong to a child.
    let all_columns_are_included =
        expr_columns.iter().all(|col| column_map.contains_key(col));
    if all_columns_are_included {
        // Since we are sure that one to one column mapping includes all columns, we convert
        // the sort expression into a filter expression.
        let converted_filter_expr = expr.transform_up(&|p| {
            convert_filter_columns(p.as_ref(), &column_map).map(|transformed| {
                match transformed {
                    Some(transformed) => Transformed::Yes(transformed),
                    None => Transformed::No(p),
                }
            })
        })?;
        // Search the converted `PhysicalExpr` in filter expression; if an exact
        // match is found, use this sorted expression in graph traversals.
        if check_filter_expr_contains_sort_information(
            filter.expression(),
            &converted_filter_expr,
        ) {
            return Ok(Some(converted_filter_expr));
        }
    }
    Ok(None)
}

/// This function is used to build the filter expression based on the sort order of input columns.
///
/// It first calls the [`convert_sort_expr_with_filter_schema`] method to determine if the sort
/// order of columns can be used in the filter expression. If it returns a [`Some`] value, the
/// method wraps the result in a [`SortedFilterExpr`] instance with the original sort expression and
/// the converted filter expression. Otherwise, this function returns an error.
///
/// The `SortedFilterExpr` instance contains information about the sort order of columns that can
/// be used in the filter expression, which can be used to optimize the query execution process.
pub fn build_filter_input_order(
    side: JoinSide,
    filter: &JoinFilter,
    schema: &SchemaRef,
    order: &PhysicalSortExpr,
) -> Result<Option<SortedFilterExpr>> {
    let opt_expr = convert_sort_expr_with_filter_schema(&side, filter, schema, order)?;
    Ok(opt_expr.map(|filter_expr| SortedFilterExpr::new(order.clone(), filter_expr)))
}

/// Convert a physical expression into a filter expression using the given
/// column mapping information.
fn convert_filter_columns(
    input: &dyn PhysicalExpr,
    column_map: &HashMap<Column, Column>,
) -> Result<Option<Arc<dyn PhysicalExpr>>> {
    // Attempt to downcast the input expression to a Column type.
    Ok(if let Some(col) = input.as_any().downcast_ref::<Column>() {
        // If the downcast is successful, retrieve the corresponding filter column.
        column_map.get(col).map(|c| Arc::new(c.clone()) as _)
    } else {
        // If the downcast fails, return the input expression as is.
        None
    })
}

/// The [SortedFilterExpr] object represents a sorted filter expression. It
/// contains the following information: The origin expression, the filter
/// expression, an interval encapsulating expression bounds, and a stable
/// index identifying the expression in the expression DAG.
///
/// Physical schema of a [JoinFilter]'s intermediate batch combines two sides
/// and uses new column names. In this process, a column exchange is done so
/// we can utilize sorting information while traversing the filter expression
/// DAG for interval calculations. When evaluating the inner buffer, we use
/// `origin_sorted_expr`.
#[derive(Debug, Clone)]
pub struct SortedFilterExpr {
    /// Sorted expression from a join side (i.e. a child of the join)
    origin_sorted_expr: PhysicalSortExpr,
    /// Expression adjusted for filter schema.
    filter_expr: Arc<dyn PhysicalExpr>,
    /// Interval containing expression bounds
    interval: Interval,
    /// Node index in the expression DAG
    node_index: usize,
}

impl SortedFilterExpr {
    /// Constructor
    pub fn new(
        origin_sorted_expr: PhysicalSortExpr,
        filter_expr: Arc<dyn PhysicalExpr>,
    ) -> Self {
        Self {
            origin_sorted_expr,
            filter_expr,
            interval: Interval::default(),
            node_index: 0,
        }
    }
    /// Get origin expr information
    pub fn origin_sorted_expr(&self) -> &PhysicalSortExpr {
        &self.origin_sorted_expr
    }
    /// Get filter expr information
    pub fn filter_expr(&self) -> &Arc<dyn PhysicalExpr> {
        &self.filter_expr
    }
    /// Get interval information
    pub fn interval(&self) -> &Interval {
        &self.interval
    }
    /// Sets interval
    pub fn set_interval(&mut self, interval: Interval) {
        self.interval = interval;
    }
    /// Node index in ExprIntervalGraph
    pub fn node_index(&self) -> usize {
        self.node_index
    }
    /// Node index setter in ExprIntervalGraph
    pub fn set_node_index(&mut self, node_index: usize) {
        self.node_index = node_index;
    }
}

/// Calculate the filter expression intervals.
///
/// This function updates the `interval` field of each `SortedFilterExpr` based
/// on the first or the last value of the expression in `build_input_buffer`
/// and `probe_batch`.
///
/// # Arguments
///
/// * `build_input_buffer` - The [RecordBatch] on the build side of the join.
/// * `build_sorted_filter_expr` - Build side [SortedFilterExpr] to update.
/// * `probe_batch` - The `RecordBatch` on the probe side of the join.
/// * `probe_sorted_filter_expr` - Probe side `SortedFilterExpr` to update.
///
/// ### Note
/// ```text
///
/// Interval arithmetic is used to calculate viable join ranges for build-side
/// pruning. This is done by first creating an interval for join filter values in
/// the build side of the join, which spans [-∞, FV] or [FV, ∞] depending on the
/// ordering (descending/ascending) of the filter expression. Here, FV denotes the
/// first value on the build side. This range is then compared with the probe side
/// interval, which either spans [-∞, LV] or [LV, ∞] depending on the ordering
/// (ascending/descending) of the probe side. Here, LV denotes the last value on
/// the probe side.
///
/// As a concrete example, consider the following query:
///
///   SELECT * FROM left_table, right_table
///   WHERE
///     left_key = right_key AND
///     a > b - 3 AND
///     a < b + 10
///
/// where columns "a" and "b" come from tables "left_table" and "right_table",
/// respectively. When a new `RecordBatch` arrives at the right side, the
/// condition a > b - 3 will possibly indicate a prunable range for the left
/// side. Conversely, when a new `RecordBatch` arrives at the left side, the
/// condition a < b + 10 will possibly indicate prunability for the right side.
/// Let’s inspect what happens when a new RecordBatch` arrives at the right
/// side (i.e. when the left side is the build side):
///
///         Build      Probe
///       +-------+  +-------+
///       | a | z |  | b | y |
///       |+--|--+|  |+--|--+|
///       | 1 | 2 |  | 4 | 3 |
///       |+--|--+|  |+--|--+|
///       | 3 | 1 |  | 4 | 3 |
///       |+--|--+|  |+--|--+|
///       | 5 | 7 |  | 6 | 1 |
///       |+--|--+|  |+--|--+|
///       | 7 | 1 |  | 6 | 3 |
///       +-------+  +-------+
///
/// In this case, the interval representing viable (i.e. joinable) values for
/// column "a" is [1, ∞], and the interval representing possible future values
/// for column "b" is [6, ∞]. With these intervals at hand, we next calculate
/// intervals for the whole filter expression and propagate join constraint by
/// traversing the expression graph.
/// ```
pub fn calculate_filter_expr_intervals(
    build_input_buffer: &RecordBatch,
    build_sorted_filter_expr: &mut SortedFilterExpr,
    probe_batch: &RecordBatch,
    probe_sorted_filter_expr: &mut SortedFilterExpr,
) -> Result<()> {
    // If either build or probe side has no data, return early:
    if build_input_buffer.num_rows() == 0 || probe_batch.num_rows() == 0 {
        return Ok(());
    }
    // Calculate the interval for the build side filter expression (if present):
    update_filter_expr_interval(
        &build_input_buffer.slice(0, 1),
        build_sorted_filter_expr,
    )?;
    // Calculate the interval for the probe side filter expression (if present):
    update_filter_expr_interval(
        &probe_batch.slice(probe_batch.num_rows() - 1, 1),
        probe_sorted_filter_expr,
    )
}

/// This is a subroutine of the function [`calculate_filter_expr_intervals`].
/// It constructs the current interval using the given `batch` and updates
/// the filter expression (i.e. `sorted_expr`) with this interval.
pub fn update_filter_expr_interval(
    batch: &RecordBatch,
    sorted_expr: &mut SortedFilterExpr,
) -> Result<()> {
    // Evaluate the filter expression and convert the result to an array:
    let array = sorted_expr
        .origin_sorted_expr()
        .expr
        .evaluate(batch)?
        .into_array(1);
    // Convert the array to a ScalarValue:
    let value = ScalarValue::try_from_array(&array, 0)?;
    // Create a ScalarValue representing positive or negative infinity for the same data type:
    let unbounded = IntervalBound::make_unbounded(value.data_type())?;
    // Update the interval with lower and upper bounds based on the sort option:
    let interval = if sorted_expr.origin_sorted_expr().options.descending {
        Interval::new(unbounded, IntervalBound::new(value, false))
    } else {
        Interval::new(IntervalBound::new(value, false), unbounded)
    };
    // Set the calculated interval for the sorted filter expression:
    sorted_expr.set_interval(interval);
    Ok(())
}

/// Get the anti join indices from the visited hash set.
///
/// This method returns the indices from the original input that were not present in the visited hash set.
///
/// # Arguments
///
/// * `prune_length` - The length of the pruned record batch.
/// * `deleted_offset` - The offset to the indices.
/// * `visited_rows` - The hash set of visited indices.
///
/// # Returns
///
/// A `PrimitiveArray` of the anti join indices.
pub fn get_pruning_anti_indices<T: ArrowPrimitiveType>(
    prune_length: usize,
    deleted_offset: usize,
    visited_rows: &HashSet<usize>,
) -> PrimitiveArray<T>
where
    NativeAdapter<T>: From<<T as ArrowPrimitiveType>::Native>,
{
    let mut bitmap = BooleanBufferBuilder::new(prune_length);
    bitmap.append_n(prune_length, false);
    // mark the indices as true if they are present in the visited hash set
    for v in 0..prune_length {
        let row = v + deleted_offset;
        bitmap.set_bit(v, visited_rows.contains(&row));
    }
    // get the anti index
    (0..prune_length)
        .filter_map(|idx| (!bitmap.get_bit(idx)).then_some(T::Native::from_usize(idx)))
        .collect()
}

/// This method creates a boolean buffer from the visited rows hash set
/// and the indices of the pruned record batch slice.
///
/// It gets the indices from the original input that were present in the visited hash set.
///
/// # Arguments
///
/// * `prune_length` - The length of the pruned record batch.
/// * `deleted_offset` - The offset to the indices.
/// * `visited_rows` - The hash set of visited indices.
///
/// # Returns
///
/// A [PrimitiveArray] of the specified type T, containing the semi indices.
pub fn get_pruning_semi_indices<T: ArrowPrimitiveType>(
    prune_length: usize,
    deleted_offset: usize,
    visited_rows: &HashSet<usize>,
) -> PrimitiveArray<T>
where
    NativeAdapter<T>: From<<T as ArrowPrimitiveType>::Native>,
{
    let mut bitmap = BooleanBufferBuilder::new(prune_length);
    bitmap.append_n(prune_length, false);
    // mark the indices as true if they are present in the visited hash set
    (0..prune_length).for_each(|v| {
        let row = &(v + deleted_offset);
        bitmap.set_bit(v, visited_rows.contains(row));
    });
    // get the semi index
    (0..prune_length)
        .filter_map(|idx| (bitmap.get_bit(idx)).then_some(T::Native::from_usize(idx)))
        .collect::<PrimitiveArray<T>>()
}

pub fn combine_two_batches(
    output_schema: &SchemaRef,
    left_batch: Option<RecordBatch>,
    right_batch: Option<RecordBatch>,
) -> Result<Option<RecordBatch>> {
    match (left_batch, right_batch) {
        (Some(batch), None) | (None, Some(batch)) => {
            // If only one of the batches are present, return it:
            Ok(Some(batch))
        }
        (Some(left_batch), Some(right_batch)) => {
            // If both batches are present, concatenate them:
            concat_batches(output_schema, &[left_batch, right_batch])
                .map_err(DataFusionError::ArrowError)
                .map(Some)
        }
        (None, None) => {
            // If neither is present, return an empty batch:
            Ok(None)
        }
    }
}

/// Records the visited indices from the input `PrimitiveArray` of type `T` into the given hash set `visited`.
/// This function will insert the indices (offset by `offset`) into the `visited` hash set.
///
/// # Arguments
///
/// * `visited` - A hash set to store the visited indices.
/// * `offset` - An offset to the indices in the `PrimitiveArray`.
/// * `indices` - The input `PrimitiveArray` of type `T` which stores the indices to be recorded.
///
pub fn record_visited_indices<T: ArrowPrimitiveType>(
    visited: &mut HashSet<usize>,
    offset: usize,
    indices: &PrimitiveArray<T>,
) {
    for i in indices.values() {
        visited.insert(i.as_usize() + offset);
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
    right: Arc<dyn ExecutionPlan>,
    on: JoinOn,
    join_type: &JoinType,
    schema: &Schema,
) -> Statistics {
    // TODO: this is not correct
    left.statistics()
}
