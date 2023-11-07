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

//! This crate contains code generated from the Ballista Protocol Buffer Definition as well
//! as convenience code for interacting with the generated code.

use crate::{error::BallistaError, serde::scheduler::Action as BallistaAction};

use arrow_flight::sql::ProstMessageExt;
use datafusion::common::DataFusionError;
use datafusion::execution::FunctionRegistry;
use datafusion::physical_plan::expressions::Column;
use datafusion::physical_plan::joins::utils::{ColumnIndex, JoinFilter};
use datafusion::physical_plan::joins::PartitionMode;
use datafusion::physical_plan::{ExecutionPlan, Partitioning};
use datafusion::prelude::JoinType;
use datafusion_proto::common::proto_error;
use datafusion_proto::into_required;
use datafusion_proto::physical_plan::from_proto::{
    parse_physical_expr, parse_protobuf_hash_partitioning,
};
use datafusion_proto::protobuf::{LogicalPlanNode, PhysicalPlanNode};
use datafusion_proto::{
    convert_required,
    logical_plan::{AsLogicalPlan, DefaultLogicalExtensionCodec, LogicalExtensionCodec},
    physical_plan::{AsExecutionPlan, PhysicalExtensionCodec},
};

use prost::Message;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::sync::Arc;
use std::{convert::TryInto, io::Cursor};

use crate::execution_plans::{
    RMHashJoinExec, ShuffleReaderExec, ShuffleWriter, ShuffleWriterExec,
    UnresolvedShuffleExec,
};
use crate::serde::protobuf::ballista_physical_plan_node::PhysicalPlanType;
use crate::serde::scheduler::PartitionLocation;
pub use generated::ballista as protobuf;

pub mod generated;
pub mod scheduler;

impl ProstMessageExt for protobuf::Action {
    fn type_url() -> &'static str {
        "type.googleapis.com/arrow.flight.protocol.sql.Action"
    }

    fn as_any(&self) -> arrow_flight::sql::Any {
        arrow_flight::sql::Any {
            type_url: protobuf::Action::type_url().to_string(),
            value: self.encode_to_vec().into(),
        }
    }
}

pub fn decode_protobuf(bytes: &[u8]) -> Result<BallistaAction, BallistaError> {
    let mut buf = Cursor::new(bytes);

    protobuf::Action::decode(&mut buf)
        .map_err(|e| BallistaError::Internal(format!("{e:?}")))
        .and_then(|node| node.try_into())
}

#[derive(Clone, Debug)]
pub struct BallistaCodec<
    T: 'static + AsLogicalPlan = LogicalPlanNode,
    U: 'static + AsExecutionPlan = PhysicalPlanNode,
> {
    logical_extension_codec: Arc<dyn LogicalExtensionCodec>,
    physical_extension_codec: Arc<dyn PhysicalExtensionCodec>,
    logical_plan_repr: PhantomData<T>,
    physical_plan_repr: PhantomData<U>,
}

impl Default for BallistaCodec {
    fn default() -> Self {
        Self {
            logical_extension_codec: Arc::new(DefaultLogicalExtensionCodec {}),
            physical_extension_codec: Arc::new(BallistaPhysicalExtensionCodec {}),
            logical_plan_repr: PhantomData,
            physical_plan_repr: PhantomData,
        }
    }
}

impl<T: 'static + AsLogicalPlan, U: 'static + AsExecutionPlan> BallistaCodec<T, U> {
    pub fn new(
        logical_extension_codec: Arc<dyn LogicalExtensionCodec>,
        physical_extension_codec: Arc<dyn PhysicalExtensionCodec>,
    ) -> Self {
        Self {
            logical_extension_codec,
            physical_extension_codec,
            logical_plan_repr: PhantomData,
            physical_plan_repr: PhantomData,
        }
    }

    pub fn logical_extension_codec(&self) -> &dyn LogicalExtensionCodec {
        self.logical_extension_codec.as_ref()
    }

    pub fn physical_extension_codec(&self) -> &dyn PhysicalExtensionCodec {
        self.physical_extension_codec.as_ref()
    }
}

#[derive(Debug)]
pub struct BallistaPhysicalExtensionCodec {}

impl PhysicalExtensionCodec for BallistaPhysicalExtensionCodec {
    fn try_decode(
        &self,
        buf: &[u8],
        inputs: &[Arc<dyn ExecutionPlan>],
        registry: &dyn FunctionRegistry,
    ) -> Result<Arc<dyn ExecutionPlan>, DataFusionError> {
        let ballista_plan: protobuf::BallistaPhysicalPlanNode =
            protobuf::BallistaPhysicalPlanNode::decode(buf).map_err(|e| {
                DataFusionError::Internal(format!(
                    "Could not deserialize BallistaPhysicalPlanNode: {e}"
                ))
            })?;

        let ballista_plan =
            ballista_plan.physical_plan_type.as_ref().ok_or_else(|| {
                DataFusionError::Internal(
                    "Could not deserialize BallistaPhysicalPlanNode because it's physical_plan_type is none".to_string()
                )
            })?;

        match ballista_plan {
            PhysicalPlanType::ShuffleWriter(shuffle_writer) => {
                let input = inputs[0].clone();

                let shuffle_output_partitioning = parse_protobuf_hash_partitioning(
                    shuffle_writer.output_partitioning.as_ref(),
                    registry,
                    input.schema().as_ref(),
                )?;

                let mode = protobuf::RemoteMemoryMode::from_i32(
                    shuffle_writer.remote_memory_mode,
                )
                .unwrap();

                Ok(Arc::new(ShuffleWriterExec::try_new(
                    shuffle_writer.job_id.clone(),
                    shuffle_writer.stage_id as usize,
                    input,
                    "".to_string(), // this is intentional but hacky - the executor will fill this in
                    shuffle_output_partitioning,
                    mode.into(),
                )?))
            }
            PhysicalPlanType::ShuffleReader(shuffle_reader) => {
                let stage_id = shuffle_reader.stage_id as usize;
                let schema = Arc::new(convert_required!(shuffle_reader.schema)?);
                let partition_location: Vec<Vec<PartitionLocation>> = shuffle_reader
                    .partition
                    .iter()
                    .map(|p| {
                        p.location
                            .iter()
                            .map(|l| {
                                l.clone().try_into().map_err(|e| {
                                    DataFusionError::Internal(format!(
                                        "Fail to get partition location due to {e:?}"
                                    ))
                                })
                            })
                            .collect::<Result<Vec<_>, _>>()
                    })
                    .collect::<Result<Vec<_>, DataFusionError>>()?;

                let mode = protobuf::RemoteMemoryMode::from_i32(
                    shuffle_reader.remote_memory_mode,
                )
                .unwrap();

                let shuffle_reader = ShuffleReaderExec::try_new(
                    stage_id,
                    partition_location,
                    schema,
                    mode.into(),
                )?;
                Ok(Arc::new(shuffle_reader))
            }
            PhysicalPlanType::UnresolvedShuffle(unresolved_shuffle) => {
                let schema = Arc::new(convert_required!(unresolved_shuffle.schema)?);
                let mode = protobuf::RemoteMemoryMode::from_i32(
                    unresolved_shuffle.remote_memory_mode,
                )
                .unwrap();

                Ok(Arc::new(UnresolvedShuffleExec {
                    stage_id: unresolved_shuffle.stage_id as usize,
                    schema,
                    input_partition_count: unresolved_shuffle.input_partition_count
                        as usize,
                    output_partition_count: unresolved_shuffle.output_partition_count
                        as usize,
                    remote_memory_mode: mode.into(),
                }))
            }
            PhysicalPlanType::HashJoin(hash_join) => {
                let left = inputs[0].clone();
                let right = inputs[1].clone();
                let on: Vec<(Column, Column)> = hash_join
                    .on
                    .iter()
                    .map(|col| {
                        let left = into_required!(col.left)?;
                        let right = into_required!(col.right)?;
                        Ok((left, right))
                    })
                    .collect::<datafusion::error::Result<_>>()?;

                let join_type =
                    datafusion_proto::protobuf::JoinType::from_i32(hash_join.join_type)
                        .expect("invalid join type");
                let join_type = JoinType::from(join_type);

                let filter = hash_join
                    .filter
                    .as_ref()
                    .map(|f| {
                        let schema = f
                            .schema
                            .as_ref()
                            .ok_or_else(|| proto_error("Missing JoinFilter schema"))?
                            .try_into()?;

                        let expression = parse_physical_expr(
                            f.expression.as_ref().ok_or_else(|| {
                                proto_error("Unexpected empty filter expression")
                            })?,
                            registry, &schema
                        )?;
                        let column_indices = f.column_indices
                            .iter()
                            .map(|i| {
                                let side = datafusion_proto:: protobuf::JoinSide::from_i32(i.side)
                                    .expect(
                                        "Received a HashJoinNode message with JoinSide in Filter"
                                    );

                                Ok(ColumnIndex{
                                    index: i.index as usize,
                                    side: side.into(),
                                })
                            })
                            .collect::<datafusion::error::Result<Vec<_>>>()?;

                        Ok(JoinFilter::new(expression, column_indices, schema))
                    })
                    .map_or(Ok(None), |v: datafusion::error::Result<JoinFilter>| v.map(Some))?;

                let partition_mode = datafusion_proto::protobuf::PartitionMode::from_i32(
                    hash_join.partition_mode,
                )
                .expect("Received a HashJoinNode message with unknown PartitionMode ");
                let partition_mode = match partition_mode {
                    datafusion_proto::protobuf::PartitionMode::CollectLeft => {
                        PartitionMode::CollectLeft
                    }
                    datafusion_proto::protobuf::PartitionMode::Partitioned => {
                        PartitionMode::Partitioned
                    }
                    datafusion_proto::protobuf::PartitionMode::Auto => {
                        PartitionMode::Auto
                    }
                };
                Ok(Arc::new(RMHashJoinExec::try_new(
                    left,
                    right,
                    on,
                    filter,
                    &join_type,
                    partition_mode,
                    hash_join.null_equals_null,
                )?))
            }
        }
    }

    fn try_encode(
        &self,
        node: Arc<dyn ExecutionPlan>,
        buf: &mut Vec<u8>,
    ) -> Result<(), DataFusionError> {
        if let Some(exec) = node.as_any().downcast_ref::<ShuffleWriterExec>() {
            // note that we use shuffle_output_partitioning() rather than output_partitioning()
            // to get the true output partitioning
            let output_partitioning = match exec.shuffle_output_partitioning() {
                Some(Partitioning::Hash(exprs, partition_count)) => {
                    Some(datafusion_proto::protobuf::PhysicalHashRepartition {
                        hash_expr: exprs
                            .iter()
                            .map(|expr| expr.clone().try_into())
                            .collect::<Result<Vec<_>, DataFusionError>>()?,
                        partition_count: *partition_count as u64,
                    })
                }
                None => None,
                other => {
                    return Err(DataFusionError::Internal(format!(
                        "physical_plan::to_proto() invalid partitioning for ShuffleWriterExec: {other:?}"
                    )));
                }
            };

            let mode: protobuf::RemoteMemoryMode = exec.remote_memory_mode().into();

            let proto = protobuf::BallistaPhysicalPlanNode {
                physical_plan_type: Some(PhysicalPlanType::ShuffleWriter(
                    protobuf::ShuffleWriterExecNode {
                        job_id: exec.job_id().to_string(),
                        stage_id: exec.stage_id() as u32,
                        input: None,
                        output_partitioning,
                        remote_memory_mode: mode.into(),
                    },
                )),
            };

            proto.encode(buf).map_err(|e| {
                DataFusionError::Internal(format!(
                    "failed to encode shuffle writer execution plan: {e:?}"
                ))
            })?;

            Ok(())
        } else if let Some(exec) = node.as_any().downcast_ref::<ShuffleReaderExec>() {
            let stage_id = exec.stage_id as u32;
            let mut partition = vec![];
            for location in &exec.partition {
                partition.push(protobuf::ShuffleReaderPartition {
                    location: location
                        .iter()
                        .map(|l| {
                            l.clone().try_into().map_err(|e| {
                                DataFusionError::Internal(format!(
                                    "Fail to get partition location due to {e:?}"
                                ))
                            })
                        })
                        .collect::<Result<Vec<_>, _>>()?,
                });
            }
            let mode: protobuf::RemoteMemoryMode = exec.remote_memory_mode().into();
            let proto = protobuf::BallistaPhysicalPlanNode {
                physical_plan_type: Some(PhysicalPlanType::ShuffleReader(
                    protobuf::ShuffleReaderExecNode {
                        stage_id,
                        partition,
                        schema: Some(exec.schema().as_ref().try_into()?),
                        remote_memory_mode: mode.into(),
                    },
                )),
            };
            proto.encode(buf).map_err(|e| {
                DataFusionError::Internal(format!(
                    "failed to encode shuffle reader execution plan: {e:?}"
                ))
            })?;

            Ok(())
        } else if let Some(exec) = node.as_any().downcast_ref::<UnresolvedShuffleExec>() {
            let mode: protobuf::RemoteMemoryMode = exec.remote_memory_mode.into();

            let proto = protobuf::BallistaPhysicalPlanNode {
                physical_plan_type: Some(PhysicalPlanType::UnresolvedShuffle(
                    protobuf::UnresolvedShuffleExecNode {
                        stage_id: exec.stage_id as u32,
                        schema: Some(exec.schema().as_ref().try_into()?),
                        input_partition_count: exec.input_partition_count as u32,
                        output_partition_count: exec.output_partition_count as u32,
                        remote_memory_mode: mode.into(),
                    },
                )),
            };
            proto.encode(buf).map_err(|e| {
                DataFusionError::Internal(format!(
                    "failed to encode unresolved shuffle execution plan: {e:?}"
                ))
            })?;

            Ok(())
        } else if let Some(exec) = node.as_any().downcast_ref::<RMHashJoinExec>() {
            let on: Vec<datafusion_proto::protobuf::JoinOn> = exec
                .on()
                .iter()
                .map(|tuple| datafusion_proto::protobuf::JoinOn {
                    left: Some(datafusion_proto::protobuf::PhysicalColumn {
                        name: tuple.0.name().to_string(),
                        index: tuple.0.index() as u32,
                    }),
                    right: Some(datafusion_proto::protobuf::PhysicalColumn {
                        name: tuple.1.name().to_string(),
                        index: tuple.1.index() as u32,
                    }),
                })
                .collect();
            let join_type: datafusion_proto::protobuf::JoinType =
                exec.join_type().to_owned().into();

            let filter = exec
                .filter()
                .as_ref()
                .map(|f| {
                    let expression = f.expression().to_owned().try_into()?;
                    let column_indices = f
                        .column_indices()
                        .iter()
                        .map(|i| {
                            let side: datafusion_proto::protobuf::JoinSide =
                                i.side.to_owned().into();
                            datafusion_proto::protobuf::ColumnIndex {
                                index: i.index as u32,
                                side: side.into(),
                            }
                        })
                        .collect();
                    let schema = f.schema().try_into()?;
                    Ok(datafusion_proto::protobuf::JoinFilter {
                        expression: Some(expression),
                        column_indices,
                        schema: Some(schema),
                    })
                })
                .map_or(
                    Ok(None),
                    |v: datafusion::error::Result<
                        datafusion_proto::protobuf::JoinFilter,
                    >| v.map(Some),
                )?;

            let partition_mode = match exec.partition_mode() {
                PartitionMode::CollectLeft => {
                    datafusion_proto::protobuf::PartitionMode::CollectLeft
                }
                PartitionMode::Partitioned => {
                    datafusion_proto::protobuf::PartitionMode::Partitioned
                }
                PartitionMode::Auto => datafusion_proto::protobuf::PartitionMode::Auto,
            };

            let proto = protobuf::BallistaPhysicalPlanNode {
                physical_plan_type: Some(PhysicalPlanType::HashJoin(
                    protobuf::RmHashJoinExecNode {
                        left: None,
                        right: None,
                        on,
                        join_type: join_type.into(),
                        filter,
                        partition_mode: partition_mode.into(),
                        null_equals_null: exec.null_equals_null(),
                    },
                )),
            };
            proto.encode(buf).map_err(|e| {
                DataFusionError::Internal(format!(
                    "failed to encode remote hash join execution plan: {e:?}"
                ))
            })?;
            Ok(())
        } else {
            Err(DataFusionError::Internal(format!(
                "physical_plan::to_proto() unsupported plan type: {:?}",
                node
            )))
        }
    }
}
