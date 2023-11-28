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

//! Utilities for producing dot diagrams from execution graphs

use crate::api::get_elapsed_compute_nanos;
use crate::state::execution_graph::ExecutionGraph;
use ballista_core::execution_plans::{
    RMHashJoinExec, ShuffleReaderExec, ShuffleWriterExec, UnresolvedShuffleExec,
};
use datafusion::datasource::listing::PartitionedFile;
use datafusion::datasource::physical_plan::{
    AvroExec, CsvExec, FileScanConfig, NdJsonExec, ParquetExec,
};
use datafusion::physical_plan::aggregates::AggregateExec;
use datafusion::physical_plan::coalesce_batches::CoalesceBatchesExec;
use datafusion::physical_plan::coalesce_partitions::CoalescePartitionsExec;
use datafusion::physical_plan::filter::FilterExec;
use datafusion::physical_plan::joins::CrossJoinExec;
use datafusion::physical_plan::joins::HashJoinExec;
use datafusion::physical_plan::limit::{GlobalLimitExec, LocalLimitExec};
use datafusion::physical_plan::memory::MemoryExec;
use datafusion::physical_plan::metrics::MetricsSet;
use datafusion::physical_plan::projection::ProjectionExec;
use datafusion::physical_plan::repartition::RepartitionExec;
use datafusion::physical_plan::sorts::sort::SortExec;
use datafusion::physical_plan::union::UnionExec;
use datafusion::physical_plan::{ExecutionPlan, Partitioning, PhysicalExpr};
use log::debug;
use object_store::path::Path;
use std::collections::HashMap;
use std::fmt::{self, Write};
use std::sync::Arc;

use super::execution_graph::ExecutionStage;

/// Utility for producing dot diagrams from execution graphs
pub struct ExecutionGraphDot<'a> {
    graph: &'a ExecutionGraph,
}

const DEFAULT_FONT: &str = "Courier";

impl<'a> ExecutionGraphDot<'a> {
    /// Create a DOT graph from the provided ExecutionGraph
    pub fn generate(graph: &'a ExecutionGraph) -> Result<String, fmt::Error> {
        let mut dot = Self { graph };
        dot.generate_inner()
    }

    /// Create a DOT graph for one query stage from the provided ExecutionGraph
    pub fn generate_for_query_stage(
        graph: &ExecutionGraph,
        stage_id: usize,
    ) -> Result<String, fmt::Error> {
        if let Some(stage) = graph.stages().get(&stage_id) {
            let mut dot = String::new();
            writeln!(&mut dot, "digraph \"{}\" {{", graph.job_name())?;
            writeln!(&mut dot, "graph [fontname=\"{}\"];", DEFAULT_FONT)?;
            writeln!(&mut dot, "node [fontname=\"{}\"];", DEFAULT_FONT)?;
            writeln!(&mut dot, "edge [fontname=\"{}\"];", DEFAULT_FONT)?;
            let stage_name = format!("stage_{stage_id}");

            StagePlanWriter::new(&mut dot, &stage_name, &vec![])
                .write(stage.plan(), 0)?;
            writeln!(&mut dot, "}}")?;
            Ok(dot)
        } else {
            Err(fmt::Error)
        }
    }

    fn generate_inner(&mut self) -> Result<String, fmt::Error> {
        // sort the stages by key for deterministic output for tests
        let stages = self.graph.stages();
        let mut stage_ids: Vec<usize> = stages.keys().cloned().collect();
        stage_ids.sort();

        let mut dot = String::new();

        writeln!(&mut dot, "digraph \"{}\" {{", self.graph.job_name())?;
        writeln!(&mut dot, "graph [fontname=\"{}\"];", DEFAULT_FONT)?;
        writeln!(&mut dot, "node [fontname=\"{}\"];", DEFAULT_FONT)?;
        writeln!(&mut dot, "edge [fontname=\"{}\"];", DEFAULT_FONT)?;

        let mut cluster = 0;
        let mut stage_meta = vec![];

        #[allow(clippy::explicit_counter_loop)]
        for id in &stage_ids {
            let stage = stages.get(id).unwrap(); // safe unwrap

            let dummy_metrics = vec![];
            let (elapsed_time, metrics) = match stage {
                ExecutionStage::Successful(s) => {
                    let e = get_elapsed_compute_nanos(&s.stage_metrics);
                    (format!("({})", e), &s.stage_metrics)
                }
                _ => ("".to_string(), &dummy_metrics),
            };

            let stage_name = format!("stage_{id}");
            writeln!(&mut dot, "\tsubgraph cluster{cluster} {{")?;
            writeln!(
                &mut dot,
                "\t\tlabel = \"Stage {} [{}{}]\";",
                id,
                stage.variant_name(),
                elapsed_time,
            )?;

            let mut stage_plan_writer =
                StagePlanWriter::new(&mut dot, &stage_name, metrics);
            stage_meta.push(stage_plan_writer.write(stage.plan(), 0)?);
            cluster += 1;
            writeln!(&mut dot, "\t}}")?; // end of subgraph
        }

        // write links between stages
        for meta in &stage_meta {
            let mut links = vec![];
            for (reader_node, parent_stage_id) in &meta.readers {
                // shuffle write node is always node zero
                let parent_shuffle_write_node = format!("stage_{parent_stage_id}_0");
                links.push(format!("{parent_shuffle_write_node} -> {reader_node}"));
            }
            // keep the order deterministic
            links.sort();
            for link in links {
                writeln!(&mut dot, "\t{link}")?;
            }
        }

        writeln!(&mut dot, "}}")?; // end of digraph

        Ok(dot)
    }
}

struct StagePlanWriter<'a> {
    f: &'a mut String,
    prefix: &'a str,
    metrics: &'a Vec<MetricsSet>,
    metrics_idx: usize,
}

impl<'a> StagePlanWriter<'a> {
    fn new(
        dot: &'a mut String,
        prefix: &'a str,
        plan_metrics: &'a Vec<MetricsSet>,
    ) -> Self {
        Self {
            f: dot,
            prefix,
            metrics: plan_metrics,
            metrics_idx: 0,
        }
    }

    fn write(
        &mut self,
        plan: &dyn ExecutionPlan,
        i: usize,
    ) -> Result<StagePlanState, fmt::Error> {
        let mut state = StagePlanState {
            readers: HashMap::new(),
        };
        self.metrics_idx = 0;
        self.write_plan_recursive(self.prefix, plan, i, &mut state)?;
        Ok(state)
    }

    fn write_plan_recursive(
        &mut self,
        prefix: &str,
        plan: &dyn ExecutionPlan,
        i: usize,
        state: &mut StagePlanState,
    ) -> Result<(), fmt::Error> {
        let node_name = format!("{}_{i}", prefix);
        let display_name = get_operator_name(plan, self.metrics.get(self.metrics_idx));

        if let Some(reader) = plan.as_any().downcast_ref::<ShuffleReaderExec>() {
            for part in &reader.partition {
                for loc in part {
                    state
                        .readers
                        .insert(node_name.clone(), loc.partition_id.stage_id);
                }
            }
        } else if let Some(reader) = plan.as_any().downcast_ref::<UnresolvedShuffleExec>()
        {
            state.readers.insert(node_name.clone(), reader.stage_id);
        }

        let mut metrics_str = vec![];
        if let Some(metrics) = plan.metrics() {
            if let Some(x) = metrics.output_rows() {
                metrics_str.push(format!("output_rows={x}"))
            }
            if let Some(x) = metrics.elapsed_compute() {
                metrics_str.push(format!("elapsed_compute={x}"))
            }
        }
        if metrics_str.is_empty() {
            writeln!(
                self.f,
                "\t\t{node_name} [shape=box, label=\"{display_name}\"]"
            )?;
        } else {
            writeln!(
                self.f,
                "\t\t{} [shape=box, label=\"{}
    {}\"]",
                node_name,
                display_name,
                metrics_str.join(", ")
            )?;
        }

        for (j, child) in plan.children().into_iter().enumerate() {
            self.metrics_idx += 1;
            self.write_plan_recursive(&node_name, child.as_ref(), j, state)?;
            // write link from child to parent
            writeln!(&mut self.f, "\t\t{node_name}_{j} -> {node_name}")?;
        }

        Ok(())
    }
}

#[derive(Debug)]
struct StagePlanState {
    /// map from reader node name to parent stage id
    readers: HashMap<String, usize>,
}

/// Make strings dot-friendly
fn sanitize_dot_label(str: &str) -> String {
    // TODO make max length configurable eventually
    sanitize(str, Some(100))
}

/// Make strings dot-friendly
fn sanitize(str: &str, max_len: Option<usize>) -> String {
    let mut sanitized = String::new();
    for ch in str.chars() {
        match ch {
            '"' => sanitized.push('`'),
            ' ' | '_' | '+' | '-' | '*' | '/' | '(' | ')' | '[' | ']' | '{' | '}'
            | '!' | '@' | '#' | '$' | '%' | '&' | '=' | ':' | ';' | '\\' | '\'' | '.'
            | ',' | '<' | '>' | '`' => sanitized.push(ch),
            _ if ch.is_ascii_alphanumeric() || ch.is_ascii_whitespace() => {
                sanitized.push(ch)
            }
            _ => sanitized.push('?'),
        }
    }
    // truncate after translation because we know we only have ASCII chars at this point
    // so the slice is safe (not splitting unicode character bytes)
    if let Some(limit) = max_len {
        if sanitized.len() > limit {
            sanitized.truncate(limit);
            return sanitized + " ...";
        }
    }
    sanitized
}

fn wrap_text_by_comma(s: &str, width: usize) -> String {
    let chunks = s.split(',').collect::<Vec<_>>();
    let mut result = String::new();
    let mut line = String::new();

    for chunk in chunks.iter() {
        if line.len() + chunk.len() > width {
            result.push_str(&line);
            result.push('\n');
            line.clear();
        }
        if !line.is_empty() {
            line.push(',');
        }
        line.push_str(chunk);
    }

    if !line.is_empty() {
        result.push_str(&line);
    }

    result
}

fn get_operator_name(plan: &dyn ExecutionPlan, metric: Option<&MetricsSet>) -> String {
    let metric_str = if let Some(m) = metric {
        wrap_text_by_comma(&m.aggregate_by_name().timestamps_removed().to_string(), 80)
    } else {
        "".to_string()
    };
    if let Some(exec) = plan.as_any().downcast_ref::<FilterExec>() {
        format!(
            "Filter: {}
            {}",
            exec.predicate(),
            metric_str
        )
    } else if let Some(exec) = plan.as_any().downcast_ref::<ProjectionExec>() {
        let expr = exec
            .expr()
            .iter()
            .map(|(e, _)| format!("{e}"))
            .collect::<Vec<String>>()
            .join(", ");
        format!("Projection: {}", sanitize_dot_label(&expr))
    } else if let Some(exec) = plan.as_any().downcast_ref::<SortExec>() {
        let sort_expr = exec
            .expr()
            .iter()
            .map(|e| {
                let asc = if e.options.descending { " DESC" } else { "" };
                let nulls = if e.options.nulls_first {
                    " NULLS FIRST"
                } else {
                    ""
                };
                format!("{}{}{}", e.expr, asc, nulls)
            })
            .collect::<Vec<String>>()
            .join(", ");
        format!("Sort: {}", sanitize_dot_label(&sort_expr))
    } else if let Some(exec) = plan.as_any().downcast_ref::<AggregateExec>() {
        let group_exprs_with_alias = exec.group_expr().expr();
        let group_expr = group_exprs_with_alias
            .iter()
            .map(|(e, _)| format!("{e}"))
            .collect::<Vec<String>>()
            .join(", ");
        let aggr_expr = exec
            .aggr_expr()
            .iter()
            .map(|e| e.name().to_owned())
            .collect::<Vec<String>>()
            .join(", ");
        format!(
            "Aggregate
groupBy=[{}]
aggr=[{}]
{}",
            sanitize_dot_label(&group_expr),
            sanitize_dot_label(&aggr_expr),
            metric_str,
        )
    } else if let Some(exec) = plan.as_any().downcast_ref::<CoalesceBatchesExec>() {
        format!("CoalesceBatches [batchSize={}]", exec.target_batch_size())
    } else if let Some(exec) = plan.as_any().downcast_ref::<CoalescePartitionsExec>() {
        format!(
            "CoalescePartitions [{}]",
            format_partitioning(exec.output_partitioning())
        )
    } else if let Some(exec) = plan.as_any().downcast_ref::<RepartitionExec>() {
        format!(
            "RepartitionExec [{}]",
            format_partitioning(exec.output_partitioning())
        )
    } else if let Some(exec) = plan.as_any().downcast_ref::<HashJoinExec>() {
        let join_expr = exec
            .on()
            .iter()
            .map(|(l, r)| format!("{l} = {r}"))
            .collect::<Vec<String>>()
            .join(" AND ");
        let filter_expr = if let Some(f) = exec.filter() {
            format!("{}", f.expression())
        } else {
            "".to_string()
        };
        format!(
            "HashJoin
join_expr={}
filter_expr={}
{}",
            sanitize_dot_label(&join_expr),
            sanitize_dot_label(&filter_expr),
            metric_str
        )
    } else if let Some(exec) = plan.as_any().downcast_ref::<RMHashJoinExec>() {
        let join_expr = exec
            .on()
            .iter()
            .map(|(l, r)| format!("{l} = {r}"))
            .collect::<Vec<String>>()
            .join(" AND ");
        let filter_expr = if let Some(f) = exec.filter() {
            format!("{}", f.expression())
        } else {
            "".to_string()
        };
        format!(
            "RMHashJoin
join_expr={}
filter_expr={}
{}",
            sanitize_dot_label(&join_expr),
            sanitize_dot_label(&filter_expr),
            metric_str
        )
    } else if plan.as_any().downcast_ref::<CrossJoinExec>().is_some() {
        "CrossJoin".to_string()
    } else if plan.as_any().downcast_ref::<UnionExec>().is_some() {
        "Union".to_string()
    } else if let Some(exec) = plan.as_any().downcast_ref::<UnresolvedShuffleExec>() {
        format!("UnresolvedShuffleExec [stage_id={}]", exec.stage_id)
    } else if let Some(exec) = plan.as_any().downcast_ref::<ShuffleReaderExec>() {
        format!(
            "ShuffleReader [part: {}, mode: {} ]
            {}",
            exec.partition.len(),
            exec.remote_memory_mode(),
            metric_str
        )
    } else if let Some(exec) = plan.as_any().downcast_ref::<ShuffleWriterExec>() {
        format!(
            "ShuffleWriter [part: {}, mode: {}]
            {}",
            format_optioned_partition(exec.shuffle_output_partitioning()),
            exec.remote_memory_mode(),
            metric_str,
        )
    } else if plan.as_any().downcast_ref::<MemoryExec>().is_some() {
        "MemoryExec".to_string()
    } else if let Some(exec) = plan.as_any().downcast_ref::<CsvExec>() {
        let parts = exec.output_partitioning().partition_count();
        format!(
            "CSV: {} [{} partitions]
            {}",
            get_file_scan(exec.base_config()),
            parts,
            metric_str,
        )
    } else if let Some(exec) = plan.as_any().downcast_ref::<NdJsonExec>() {
        let parts = exec.output_partitioning().partition_count();
        format!("JSON [{parts} partitions]")
    } else if let Some(exec) = plan.as_any().downcast_ref::<AvroExec>() {
        let parts = exec.output_partitioning().partition_count();
        format!(
            "Avro: {} [{} partitions]",
            get_file_scan(exec.base_config()),
            parts
        )
    } else if let Some(exec) = plan.as_any().downcast_ref::<ParquetExec>() {
        let parts = exec.output_partitioning().partition_count();
        format!(
            "Parquet: {} [{} partitions]
            {}",
            get_file_scan(exec.base_config()),
            parts,
            metric_str,
        )
    } else if let Some(exec) = plan.as_any().downcast_ref::<GlobalLimitExec>() {
        format!(
            "GlobalLimit(skip={}, fetch={:?})",
            exec.skip(),
            exec.fetch()
        )
    } else if let Some(exec) = plan.as_any().downcast_ref::<LocalLimitExec>() {
        format!("LocalLimit({})", exec.fetch())
    } else {
        debug!(
            "Unknown physical operator when producing DOT graph: {:?}",
            plan
        );
        "Unknown Operator".to_string()
    }
}

fn format_partitioning(x: Partitioning) -> String {
    match x {
        Partitioning::UnknownPartitioning(n) | Partitioning::RoundRobinBatch(n) => {
            format!("{n} partitions")
        }
        Partitioning::Hash(expr, n) => {
            format!(
                "{} partitions, expr={}",
                n,
                sanitize(&format_expr_list(&expr), Some(100))
            )
        }
    }
}

fn format_optioned_partition(x: Option<&Partitioning>) -> String {
    match x {
        Some(partitioning) => format_partitioning(partitioning.to_owned()),
        None => "None".to_string(),
    }
}

fn format_expr_list(exprs: &[Arc<dyn PhysicalExpr>]) -> String {
    let expr_strings: Vec<String> = exprs.iter().map(|e| format!("{e}")).collect();
    expr_strings.join(", ")
}

/// Get summary of file scan locations
fn get_file_scan(scan: &FileScanConfig) -> String {
    if !scan.file_groups.is_empty() {
        let partitioned_files: Vec<PartitionedFile> = scan
            .file_groups
            .iter()
            .flat_map(|part_file| part_file.clone())
            .collect();
        let paths: Vec<Path> = partitioned_files
            .iter()
            .map(|part_file| part_file.object_meta.location.clone())
            .collect();
        match paths.len() {
            0 => "No files found".to_owned(),
            1 => {
                // single file
                format!("{}", paths[0])
            }
            _ => {
                // multiple files so show parent directory
                let path = format!("{}", paths[0]);
                let path = if let Some(i) = path.rfind('/') {
                    &path[0..i]
                } else {
                    &path
                };
                format!("{} [{} files]", path, paths.len())
            }
        }
    } else {
        "".to_string()
    }
}

#[cfg(test)]
mod tests {
    use crate::state::execution_graph::ExecutionGraph;
    use crate::state::execution_graph_dot::ExecutionGraphDot;
    use crate::state::RemoteMemoryMode;
    use ballista_core::error::{BallistaError, Result};
    use datafusion::arrow::datatypes::{DataType, Field, Schema};
    use datafusion::datasource::MemTable;
    use datafusion::prelude::{SessionConfig, SessionContext};
    use std::sync::Arc;

    #[tokio::test]
    async fn dot() -> Result<()> {
        let graph = test_graph().await?;
        let dot = ExecutionGraphDot::generate(&graph)
            .map_err(|e| BallistaError::Internal(format!("{e:?}")))?;

        let expected = r#"digraph "job_name" {
graph [fontname="Courier"];
node [fontname="Courier"];
edge [fontname="Courier"];
	subgraph cluster0 {
		label = "Stage 1 [Resolved]";
		stage_1_0 [shape=box, label="ShuffleWriter [48 partitions, expr=a@0]"]
		stage_1_0_0 [shape=box, label="MemoryExec"]
		stage_1_0_0 -> stage_1_0
	}
	subgraph cluster1 {
		label = "Stage 2 [Resolved]";
		stage_2_0 [shape=box, label="ShuffleWriter [48 partitions, expr=a@0]"]
		stage_2_0_0 [shape=box, label="MemoryExec"]
		stage_2_0_0 -> stage_2_0
	}
	subgraph cluster2 {
		label = "Stage 3 [Unresolved]";
		stage_3_0 [shape=box, label="ShuffleWriter [48 partitions, expr=b@3]"]
		stage_3_0_0 [shape=box, label="CoalesceBatches [batchSize=4096]"]
		stage_3_0_0_0 [shape=box, label="HashJoin
join_expr=a@0 = a@0
filter_expr="]
		stage_3_0_0_0_0 [shape=box, label="CoalesceBatches [batchSize=4096]"]
		stage_3_0_0_0_0_0 [shape=box, label="UnresolvedShuffleExec [stage_id=1]"]
		stage_3_0_0_0_0_0 -> stage_3_0_0_0_0
		stage_3_0_0_0_0 -> stage_3_0_0_0
		stage_3_0_0_0_1 [shape=box, label="CoalesceBatches [batchSize=4096]"]
		stage_3_0_0_0_1_0 [shape=box, label="UnresolvedShuffleExec [stage_id=2]"]
		stage_3_0_0_0_1_0 -> stage_3_0_0_0_1
		stage_3_0_0_0_1 -> stage_3_0_0_0
		stage_3_0_0_0 -> stage_3_0_0
		stage_3_0_0 -> stage_3_0
	}
	subgraph cluster3 {
		label = "Stage 4 [Resolved]";
		stage_4_0 [shape=box, label="ShuffleWriter [48 partitions, expr=b@1]"]
		stage_4_0_0 [shape=box, label="MemoryExec"]
		stage_4_0_0 -> stage_4_0
	}
	subgraph cluster4 {
		label = "Stage 5 [Unresolved]";
		stage_5_0 [shape=box, label="ShuffleWriter [None]"]
		stage_5_0_0 [shape=box, label="CoalesceBatches [batchSize=4096]"]
		stage_5_0_0_0 [shape=box, label="HashJoin
join_expr=b@3 = b@1
filter_expr="]
		stage_5_0_0_0_0 [shape=box, label="CoalesceBatches [batchSize=4096]"]
		stage_5_0_0_0_0_0 [shape=box, label="UnresolvedShuffleExec [stage_id=3]"]
		stage_5_0_0_0_0_0 -> stage_5_0_0_0_0
		stage_5_0_0_0_0 -> stage_5_0_0_0
		stage_5_0_0_0_1 [shape=box, label="CoalesceBatches [batchSize=4096]"]
		stage_5_0_0_0_1_0 [shape=box, label="UnresolvedShuffleExec [stage_id=4]"]
		stage_5_0_0_0_1_0 -> stage_5_0_0_0_1
		stage_5_0_0_0_1 -> stage_5_0_0_0
		stage_5_0_0_0 -> stage_5_0_0
		stage_5_0_0 -> stage_5_0
	}
	stage_1_0 -> stage_3_0_0_0_0_0
	stage_2_0 -> stage_3_0_0_0_1_0
	stage_3_0 -> stage_5_0_0_0_0_0
	stage_4_0 -> stage_5_0_0_0_1_0
}
"#;
        assert_eq!(expected, &dot);
        Ok(())
    }

    #[tokio::test]
    async fn query_stage() -> Result<()> {
        let graph = test_graph().await?;
        let dot = ExecutionGraphDot::generate_for_query_stage(&graph, 3)
            .map_err(|e| BallistaError::Internal(format!("{e:?}")))?;

        let expected = r#"digraph "job_name" {
graph [fontname="Courier"];
node [fontname="Courier"];
edge [fontname="Courier"];
		stage_3_0 [shape=box, label="ShuffleWriter [48 partitions, expr=b@3]"]
		stage_3_0_0 [shape=box, label="CoalesceBatches [batchSize=4096]"]
		stage_3_0_0_0 [shape=box, label="HashJoin
join_expr=a@0 = a@0
filter_expr="]
		stage_3_0_0_0_0 [shape=box, label="CoalesceBatches [batchSize=4096]"]
		stage_3_0_0_0_0_0 [shape=box, label="UnresolvedShuffleExec [stage_id=1]"]
		stage_3_0_0_0_0_0 -> stage_3_0_0_0_0
		stage_3_0_0_0_0 -> stage_3_0_0_0
		stage_3_0_0_0_1 [shape=box, label="CoalesceBatches [batchSize=4096]"]
		stage_3_0_0_0_1_0 [shape=box, label="UnresolvedShuffleExec [stage_id=2]"]
		stage_3_0_0_0_1_0 -> stage_3_0_0_0_1
		stage_3_0_0_0_1 -> stage_3_0_0_0
		stage_3_0_0_0 -> stage_3_0_0
		stage_3_0_0 -> stage_3_0
}
"#;
        assert_eq!(expected, &dot);
        Ok(())
    }

    #[tokio::test]
    async fn dot_optimized() -> Result<()> {
        let graph = test_graph_optimized().await?;
        let dot = ExecutionGraphDot::generate(&graph)
            .map_err(|e| BallistaError::Internal(format!("{e:?}")))?;

        let expected = r#"digraph "job_name" {
graph [fontname="Courier"];
node [fontname="Courier"];
edge [fontname="Courier"];
	subgraph cluster0 {
		label = "Stage 1 [Resolved]";
		stage_1_0 [shape=box, label="ShuffleWriter [48 partitions, expr=a@0]"]
		stage_1_0_0 [shape=box, label="MemoryExec"]
		stage_1_0_0 -> stage_1_0
	}
	subgraph cluster1 {
		label = "Stage 2 [Resolved]";
		stage_2_0 [shape=box, label="ShuffleWriter [48 partitions, expr=a@0]"]
		stage_2_0_0 [shape=box, label="MemoryExec"]
		stage_2_0_0 -> stage_2_0
	}
	subgraph cluster2 {
		label = "Stage 3 [Resolved]";
		stage_3_0 [shape=box, label="ShuffleWriter [48 partitions, expr=a@0]"]
		stage_3_0_0 [shape=box, label="MemoryExec"]
		stage_3_0_0 -> stage_3_0
	}
	subgraph cluster3 {
		label = "Stage 4 [Unresolved]";
		stage_4_0 [shape=box, label="ShuffleWriter [None]"]
		stage_4_0_0 [shape=box, label="CoalesceBatches [batchSize=4096]"]
		stage_4_0_0_0 [shape=box, label="HashJoin
join_expr=a@1 = a@0
filter_expr="]
		stage_4_0_0_0_0 [shape=box, label="CoalesceBatches [batchSize=4096]"]
		stage_4_0_0_0_0_0 [shape=box, label="HashJoin
join_expr=a@0 = a@0
filter_expr="]
		stage_4_0_0_0_0_0_0 [shape=box, label="CoalesceBatches [batchSize=4096]"]
		stage_4_0_0_0_0_0_0_0 [shape=box, label="UnresolvedShuffleExec [stage_id=1]"]
		stage_4_0_0_0_0_0_0_0 -> stage_4_0_0_0_0_0_0
		stage_4_0_0_0_0_0_0 -> stage_4_0_0_0_0_0
		stage_4_0_0_0_0_0_1 [shape=box, label="CoalesceBatches [batchSize=4096]"]
		stage_4_0_0_0_0_0_1_0 [shape=box, label="UnresolvedShuffleExec [stage_id=2]"]
		stage_4_0_0_0_0_0_1_0 -> stage_4_0_0_0_0_0_1
		stage_4_0_0_0_0_0_1 -> stage_4_0_0_0_0_0
		stage_4_0_0_0_0_0 -> stage_4_0_0_0_0
		stage_4_0_0_0_0 -> stage_4_0_0_0
		stage_4_0_0_0_1 [shape=box, label="CoalesceBatches [batchSize=4096]"]
		stage_4_0_0_0_1_0 [shape=box, label="UnresolvedShuffleExec [stage_id=3]"]
		stage_4_0_0_0_1_0 -> stage_4_0_0_0_1
		stage_4_0_0_0_1 -> stage_4_0_0_0
		stage_4_0_0_0 -> stage_4_0_0
		stage_4_0_0 -> stage_4_0
	}
	stage_1_0 -> stage_4_0_0_0_0_0_0_0
	stage_2_0 -> stage_4_0_0_0_0_0_1_0
	stage_3_0 -> stage_4_0_0_0_1_0
}
"#;
        assert_eq!(expected, &dot);
        Ok(())
    }

    #[tokio::test]
    async fn query_stage_optimized() -> Result<()> {
        let graph = test_graph_optimized().await?;
        let dot = ExecutionGraphDot::generate_for_query_stage(&graph, 4)
            .map_err(|e| BallistaError::Internal(format!("{e:?}")))?;

        let expected = r#"digraph "job_name" {
graph [fontname="Courier"];
node [fontname="Courier"];
edge [fontname="Courier"];
		stage_4_0 [shape=box, label="ShuffleWriter [None]"]
		stage_4_0_0 [shape=box, label="CoalesceBatches [batchSize=4096]"]
		stage_4_0_0_0 [shape=box, label="HashJoin
join_expr=a@1 = a@0
filter_expr="]
		stage_4_0_0_0_0 [shape=box, label="CoalesceBatches [batchSize=4096]"]
		stage_4_0_0_0_0_0 [shape=box, label="HashJoin
join_expr=a@0 = a@0
filter_expr="]
		stage_4_0_0_0_0_0_0 [shape=box, label="CoalesceBatches [batchSize=4096]"]
		stage_4_0_0_0_0_0_0_0 [shape=box, label="UnresolvedShuffleExec [stage_id=1]"]
		stage_4_0_0_0_0_0_0_0 -> stage_4_0_0_0_0_0_0
		stage_4_0_0_0_0_0_0 -> stage_4_0_0_0_0_0
		stage_4_0_0_0_0_0_1 [shape=box, label="CoalesceBatches [batchSize=4096]"]
		stage_4_0_0_0_0_0_1_0 [shape=box, label="UnresolvedShuffleExec [stage_id=2]"]
		stage_4_0_0_0_0_0_1_0 -> stage_4_0_0_0_0_0_1
		stage_4_0_0_0_0_0_1 -> stage_4_0_0_0_0_0
		stage_4_0_0_0_0_0 -> stage_4_0_0_0_0
		stage_4_0_0_0_0 -> stage_4_0_0_0
		stage_4_0_0_0_1 [shape=box, label="CoalesceBatches [batchSize=4096]"]
		stage_4_0_0_0_1_0 [shape=box, label="UnresolvedShuffleExec [stage_id=3]"]
		stage_4_0_0_0_1_0 -> stage_4_0_0_0_1
		stage_4_0_0_0_1 -> stage_4_0_0_0
		stage_4_0_0_0 -> stage_4_0_0
		stage_4_0_0 -> stage_4_0
}
"#;
        assert_eq!(expected, &dot);
        Ok(())
    }

    async fn test_graph() -> Result<ExecutionGraph> {
        let mut config = SessionConfig::new()
            .with_target_partitions(48)
            .with_batch_size(4096);
        config
            .options_mut()
            .optimizer
            .enable_round_robin_repartition = false;
        let ctx = SessionContext::new_with_config(config);
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::UInt32, false),
            Field::new("b", DataType::UInt32, false),
        ]));
        let table = Arc::new(MemTable::try_new(schema.clone(), vec![])?);
        ctx.register_table("foo", table.clone())?;
        ctx.register_table("bar", table.clone())?;
        ctx.register_table("baz", table)?;
        let df = ctx
            .sql("SELECT * FROM foo JOIN bar ON foo.a = bar.a JOIN baz on bar.b = baz.b")
            .await?;
        let plan = df.into_optimized_plan()?;
        let plan = ctx.state().create_physical_plan(&plan).await?;
        ExecutionGraph::new(
            "scheduler_id",
            "job_id",
            "job_name",
            "session_id",
            plan,
            0,
            RemoteMemoryMode::DoNotUse,
        )
    }

    // With the improvement of https://github.com/apache/arrow-datafusion/pull/4122,
    // Redundant RepartitionExec can be removed so that the stage number will be reduced
    async fn test_graph_optimized() -> Result<ExecutionGraph> {
        let mut config = SessionConfig::new()
            .with_target_partitions(48)
            .with_batch_size(4096);
        config
            .options_mut()
            .optimizer
            .enable_round_robin_repartition = false;
        let ctx = SessionContext::new_with_config(config);
        let schema =
            Arc::new(Schema::new(vec![Field::new("a", DataType::UInt32, false)]));
        let table = Arc::new(MemTable::try_new(schema.clone(), vec![])?);
        ctx.register_table("foo", table.clone())?;
        ctx.register_table("bar", table.clone())?;
        ctx.register_table("baz", table)?;
        let df = ctx
            .sql("SELECT * FROM foo JOIN bar ON foo.a = bar.a JOIN baz on bar.a = baz.a")
            .await?;
        let plan = df.into_optimized_plan()?;
        let plan = ctx.state().create_physical_plan(&plan).await?;
        ExecutionGraph::new(
            "scheduler_id",
            "job_id",
            "job_name",
            "session_id",
            plan,
            0,
            RemoteMemoryMode::DoNotUse,
        )
    }
}
