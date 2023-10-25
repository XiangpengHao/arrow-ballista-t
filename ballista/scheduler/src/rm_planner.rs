use std::sync::Arc;

use ballista_core::{
    error::{BallistaError, Result},
    execution_plans::{RemoteShuffleWriterExec, ShuffleWriter, UnresolvedShuffleExec},
    utils::JoinParentSide,
};
use datafusion::physical_plan::{
    coalesce_partitions::CoalescePartitionsExec, joins::HashJoinExec,
    repartition::RepartitionExec, sorts::sort_preserving_merge::SortPreservingMergeExec,
    windows::WindowAggExec, with_new_children_if_necessary, ExecutionPlan, Partitioning,
};
use log::info;

pub struct RemoteMemoryPlanner {
    next_stage_id: usize,
}

impl RemoteMemoryPlanner {
    pub fn new() -> Self {
        Self { next_stage_id: 0 }
    }
}

impl Default for RemoteMemoryPlanner {
    fn default() -> Self {
        Self::new()
    }
}
impl RemoteMemoryPlanner {
    pub fn plan_query_stages(
        &mut self,
        job_id: &str,
        execution_plan: Arc<dyn ExecutionPlan>,
    ) -> Result<Vec<Arc<RemoteShuffleWriterExec>>> {
        info!("[rm planner] planning query stages for job {}", job_id);

        let (new_plan, mut stages) = self.plan_query_stages_internal(
            job_id,
            execution_plan,
            JoinParentSide::NotApplicable,
        )?;

        stages.push(create_shuffle_writer(
            job_id,
            self.next_stage_id(),
            new_plan,
            None,
            JoinParentSide::NotApplicable,
        )?);

        Ok(stages)
    }

    fn plan_query_stages_internal(
        &mut self,
        job_id: &str,
        execution_plan: Arc<dyn ExecutionPlan>,
        parent_side: JoinParentSide,
    ) -> Result<(Arc<dyn ExecutionPlan>, Vec<Arc<RemoteShuffleWriterExec>>)> {
        if execution_plan.children().is_empty() {
            return Ok((execution_plan, vec![]));
        }

        let mut stages = vec![];
        let mut children = vec![];

        if let Some(hash_join) = execution_plan.as_any().downcast_ref::<HashJoinExec>() {
            let (left_child, mut left_stages) = self.plan_query_stages_internal(
                job_id,
                hash_join.left().clone(),
                JoinParentSide::Left,
            )?;
            children.push(left_child);
            stages.append(&mut left_stages);

            let (right_child, mut right_stages) = self.plan_query_stages_internal(
                job_id,
                hash_join.right().clone(),
                JoinParentSide::Right,
            )?;
            children.push(right_child);
            stages.append(&mut right_stages);
        } else {
            for child in execution_plan.children() {
                let (new_child, mut child_stages) =
                    self.plan_query_stages_internal(job_id, child.clone(), parent_side)?;
                children.push(new_child);
                stages.append(&mut child_stages);
            }
        }

        if let Some(_coalesce) = execution_plan
            .as_any()
            .downcast_ref::<CoalescePartitionsExec>()
        {
            let shuffle_writer = create_shuffle_writer(
                job_id,
                self.next_stage_id(),
                children[0].clone(),
                None,
                JoinParentSide::NotApplicable,
            )?;
            let unresolved_shuffle = create_unresolved_shuffle(shuffle_writer.as_ref());
            stages.push(shuffle_writer);
            Ok((
                with_new_children_if_necessary(execution_plan, vec![unresolved_shuffle])?
                    .into(),
                stages,
            ))
        } else if let Some(_sort_preserving_merge) = execution_plan
            .as_any()
            .downcast_ref::<SortPreservingMergeExec>(
        ) {
            let shuffle_writer = create_shuffle_writer(
                job_id,
                self.next_stage_id(),
                children[0].clone(),
                None,
                JoinParentSide::NotApplicable,
            )?;
            let unresolved_shuffle = create_unresolved_shuffle(shuffle_writer.as_ref());
            stages.push(shuffle_writer);
            Ok((
                with_new_children_if_necessary(execution_plan, vec![unresolved_shuffle])?
                    .into(),
                stages,
            ))
        } else if let Some(repart) =
            execution_plan.as_any().downcast_ref::<RepartitionExec>()
        {
            match repart.output_partitioning() {
                Partitioning::Hash(_, _) => {
                    let shuffle_writer = create_shuffle_writer(
                        job_id,
                        self.next_stage_id(),
                        children[0].clone(),
                        Some(repart.partitioning().to_owned()),
                        parent_side,
                    )?;
                    let unresolved_shuffle =
                        create_unresolved_shuffle(shuffle_writer.as_ref());
                    stages.push(shuffle_writer);
                    Ok((unresolved_shuffle, stages))
                }
                _ => {
                    // remove any non-hash repartition from the distributed plan
                    Ok((children[0].clone(), stages))
                }
            }
        } else if let Some(window) =
            execution_plan.as_any().downcast_ref::<WindowAggExec>()
        {
            Err(BallistaError::NotImplemented(format!(
                "WindowAggExec with window {window:?}"
            )))
        } else {
            Ok((
                with_new_children_if_necessary(execution_plan, children)?.into(),
                stages,
            ))
        }
    }

    /// Generate a new stage ID
    fn next_stage_id(&mut self) -> usize {
        self.next_stage_id += 1;
        self.next_stage_id
    }
}

fn create_unresolved_shuffle(
    shuffle_writer: &RemoteShuffleWriterExec,
) -> Arc<UnresolvedShuffleExec> {
    Arc::new(UnresolvedShuffleExec::new(
        shuffle_writer.stage_id(),
        shuffle_writer.schema(),
        shuffle_writer.output_partitioning().partition_count(),
        shuffle_writer
            .shuffle_output_partitioning()
            .map(|p| p.partition_count())
            .unwrap_or_else(|| shuffle_writer.output_partitioning().partition_count()),
        RemoteShuffleWriterExec::use_remote_memory(),
    ))
}

fn create_shuffle_writer(
    job_id: &str,
    stage_id: usize,
    plan: Arc<dyn ExecutionPlan>,
    partitioning: Option<Partitioning>,
    join_side: JoinParentSide,
) -> Result<Arc<RemoteShuffleWriterExec>> {
    let mut sf = RemoteShuffleWriterExec::try_new(
        job_id.to_owned(),
        stage_id,
        plan,
        "".to_owned(), // executor will decide on the work_dir path
        partitioning,
    )?;
    sf.with_join_side(join_side);
    Ok(Arc::new(sf))
}

#[cfg(test)]
mod test {
    use ballista_core::{error::BallistaError, execution_plans::{UnresolvedShuffleExec, ShuffleWriter}};
    use datafusion::physical_plan::{displayable, ExecutionPlan, aggregates::AggregateExec, projection::ProjectionExec, coalesce_batches::CoalesceBatchesExec, joins::HashJoinExec};
    use uuid::Uuid;

    use crate::{rm_planner::RemoteMemoryPlanner, test_utils::datafusion_test_context};

    macro_rules! downcast_exec {
        ($exec: expr, $ty: ty) => {
            $exec.as_any().downcast_ref::<$ty>().unwrap()
        };
    }

    #[tokio::test]
    async fn distributed_join_plan() -> Result<(), BallistaError> {
        let ctx = datafusion_test_context("testdata").await?;
        let session_state = ctx.state();

        // simplified form of TPC-H query 12
        let df = ctx
            .sql(
                "select
    l_shipmode,
    sum(case
            when o_orderpriority = '1-URGENT'
                or o_orderpriority = '2-HIGH'
                then 1
            else 0
        end) as high_line_count,
    sum(case
            when o_orderpriority <> '1-URGENT'
                and o_orderpriority <> '2-HIGH'
                then 1
            else 0
        end) as low_line_count
from
    lineitem
        join
    orders
    on
            l_orderkey = o_orderkey
where
        l_shipmode in ('MAIL', 'SHIP')
  and l_commitdate < l_receiptdate
  and l_shipdate < l_commitdate
  and l_receiptdate >= date '1994-01-01'
  and l_receiptdate < date '1995-01-01'
group by
    l_shipmode
order by
    l_shipmode;
",
            )
            .await?;

        let plan = df.into_optimized_plan()?;
        let plan = session_state.optimize(&plan)?;
        let plan = session_state.create_physical_plan(&plan).await?;

        let mut planner = RemoteMemoryPlanner::new();
        let job_uuid = Uuid::new_v4();
        let stages = planner.plan_query_stages(&job_uuid.to_string(), plan)?;
        for stage in &stages {
            println!("{}", displayable(stage.as_ref()).indent(false));
        }

        /* Expected result:

        ShuffleWriterExec: Some(Hash([Column { name: "l_orderkey", index: 0 }], 2))
          ProjectionExec: expr=[l_orderkey@0 as l_orderkey, l_shipmode@4 as l_shipmode]
            CoalesceBatchesExec: target_batch_size=8192
              FilterExec: (l_shipmode@4 = SHIP OR l_shipmode@4 = MAIL) AND l_commitdate@2 < l_receiptdate@3 AND l_shipdate@1 < l_commitdate@2 AND l_receiptdate@3 >= 8766 AND l_receiptdate@3 < 9131
                CsvExec: files={2 groups: [[testdata/lineitem/partition0.tbl], [testdata/lineitem/partition1.tbl]]}, has_header=false, limit=None, projection=[l_orderkey, l_shipdate, l_commitdate, l_receiptdate, l_shipmode]

        ShuffleWriterExec: Some(Hash([Column { name: "o_orderkey", index: 0 }], 2))
          CsvExec: files={1 group: [[testdata/orders/orders.tbl]]}, has_header=false, limit=None, projection=[o_orderkey, o_orderpriority]

        ShuffleWriterExec: Some(Hash([Column { name: "l_shipmode", index: 0 }], 2))
          AggregateExec: mode=Partial, gby=[l_shipmode@0 as l_shipmode], aggr=[SUM(CASE WHEN orders.o_orderpriority = Utf8("1-URGENT") OR orders.o_orderpriority = Utf8("2-HIGH") THEN Int64(1) ELSE Int64(0) END), SUM(CASE WHEN orders.o_orderpriority != Utf8("1-URGENT") AND orders.o_orderpriority != Utf8("2-HIGH") THEN Int64(1) ELSE Int64(0) END)]
            ProjectionExec: expr=[l_shipmode@1 as l_shipmode, o_orderpriority@3 as o_orderpriority]
              CoalesceBatchesExec: target_batch_size=8192
                HashJoinExec: mode=Partitioned, join_type=Inner, on=[(Column { name: "l_orderkey", index: 0 }, Column { name: "o_orderkey", index: 0 })]
                  CoalesceBatchesExec: target_batch_size=8192
                    UnresolvedShuffleExec
                  CoalesceBatchesExec: target_batch_size=8192
                    UnresolvedShuffleExec

        ShuffleWriterExec: None
          SortExec: expr=[l_shipmode@0 ASC NULLS LAST]
            ProjectionExec: expr=[l_shipmode@0 as l_shipmode, SUM(CASE WHEN orders.o_orderpriority = Utf8("1-URGENT") OR orders.o_orderpriority = Utf8("2-HIGH") THEN Int64(1) ELSE Int64(0) END)@1 as high_line_count, SUM(CASE WHEN orders.o_orderpriority != Utf8("1-URGENT") AND orders.o_orderpriority != Utf8("2-HIGH") THEN Int64(1) ELSE Int64(0) END)@2 as low_line_count]
              AggregateExec: mode=FinalPartitioned, gby=[l_shipmode@0 as l_shipmode], aggr=[SUM(CASE WHEN orders.o_orderpriority = Utf8("1-URGENT") OR orders.o_orderpriority = Utf8("2-HIGH") THEN Int64(1) ELSE Int64(0) END), SUM(CASE WHEN orders.o_orderpriority != Utf8("1-URGENT") AND orders.o_orderpriority != Utf8("2-HIGH") THEN Int64(1) ELSE Int64(0) END)]
                CoalesceBatchesExec: target_batch_size=8192
                  UnresolvedShuffleExec

        ShuffleWriterExec: None
          SortPreservingMergeExec: [l_shipmode@0 ASC NULLS LAST]
            UnresolvedShuffleExec
        */

        assert_eq!(5, stages.len());

        // verify partitioning for each stage

        // csv "lineitem" (2 files)
        assert_eq!(
            2,
            stages[0].children()[0]
                .output_partitioning()
                .partition_count()
        );
        assert_eq!(
            2,
            stages[0]
                .shuffle_output_partitioning()
                .unwrap()
                .partition_count()
        );

        // csv "orders" (1 file)
        assert_eq!(
            1,
            stages[1].children()[0]
                .output_partitioning()
                .partition_count()
        );
        assert_eq!(
            2,
            stages[1]
                .shuffle_output_partitioning()
                .unwrap()
                .partition_count()
        );

        // join and partial hash aggregate
        let input = stages[2].children()[0].clone();
        assert_eq!(2, input.output_partitioning().partition_count());
        assert_eq!(
            2,
            stages[2]
                .shuffle_output_partitioning()
                .unwrap()
                .partition_count()
        );

        let hash_agg = downcast_exec!(input, AggregateExec);

        let projection = hash_agg.children()[0].clone();
        let projection = downcast_exec!(projection, ProjectionExec);

        let coalesce_batches = projection.children()[0].clone();
        let coalesce_batches = downcast_exec!(coalesce_batches, CoalesceBatchesExec);

        let join = coalesce_batches.children()[0].clone();
        let join = downcast_exec!(join, HashJoinExec);

        let join_input_1 = join.children()[0].clone();
        // skip CoalesceBatches
        let join_input_1 = join_input_1.children()[0].clone();
        let unresolved_shuffle_reader_1 =
            downcast_exec!(join_input_1, UnresolvedShuffleExec);
        assert_eq!(unresolved_shuffle_reader_1.input_partition_count, 2); // lineitem
        assert_eq!(unresolved_shuffle_reader_1.output_partition_count, 2);

        let join_input_2 = join.children()[1].clone();
        // skip CoalesceBatches
        let join_input_2 = join_input_2.children()[0].clone();
        let unresolved_shuffle_reader_2 =
            downcast_exec!(join_input_2, UnresolvedShuffleExec);
        assert_eq!(unresolved_shuffle_reader_2.input_partition_count, 1); // orders
        assert_eq!(unresolved_shuffle_reader_2.output_partition_count, 2);

        // final partitioned hash aggregate
        assert_eq!(
            2,
            stages[3].children()[0]
                .output_partitioning()
                .partition_count()
        );
        assert!(stages[3].shuffle_output_partitioning().is_none());

        // coalesce partitions and sort
        assert_eq!(
            1,
            stages[4].children()[0]
                .output_partitioning()
                .partition_count()
        );
        assert!(stages[4].shuffle_output_partitioning().is_none());

        Ok(())
    }
}
