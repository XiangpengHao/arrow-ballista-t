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

import React, { useEffect, useState } from "react";
import { ExternalLinkIcon } from "@chakra-ui/icons";
import {
  CircularProgress,
  CircularProgressLabel,
  Skeleton,
  Flex,
  Box,
  useDisclosure,
  Button,
  Modal,
  ModalBody,
  ModalCloseButton,
  ModalContent,
  ModalFooter,
  ModalHeader,
  ModalOverlay,
  Link,
} from "@chakra-ui/react";
import { Column, DataTable } from "./DataTable";
import { FaStop } from "react-icons/fa";
import {
  GrDocumentDownload,
  GrOverview,
  GrZoomIn,
  GrZoomOut,
  GrContract,
} from "react-icons/gr";
import fileDownload from "js-file-download";
import SVG from "react-inlinesvg";
import { JobStagesQueries } from "./JobStagesMetrics";
import { graphviz, wasmFolder } from "@hpcc-js/wasm";
import { TransformWrapper, TransformComponent } from "react-zoom-pan-pinch";

export enum QueryStatus {
  QUEUED = "QUEUED",
  RUNNING = "RUNNING",
  FAILED = "FAILED",
  COMPLETED = "COMPLETED",
}

export interface Query {
  job_id: string;
  job_name: string;
  job_status: string;
  num_stages: number;
  percent_complete: number;
  start_time: number;
  elapsed: number;
  output_row_cnt: number;
}

export interface QueriesListProps {
  queries?: Query[];
  fetchJobs: () => void;
}

wasmFolder("https://cdn.jsdelivr.net/npm/@hpcc-js/wasm/dist");

export const ActionsCell = (props: any) => {
  const [dot_data, setData] = useState("");
  const [svg_data, setSvgData] = useState("");
  const { isOpen, onOpen, onClose } = useDisclosure();
  const ref = React.useRef<SVGElement>(null);

  const fetchDotData = async (url: string) => {
    const res = await fetch(url, {
      method: "GET",
      headers: {
        Accept: "application/json",
      },
    });
    const data = await res.text();
    setData(data);
    return data;
  };

  useEffect(() => {
    if (isOpen) {
      fetchDotData("/api/job/" + props.value.job_id + "/dot");
    }
  }, [isOpen, props.value.job_id]);

  useEffect(() => {
    if (isOpen && dot_data) {
      graphviz.dot(dot_data).then((svg) => {
        setSvgData(svg);
      });
    }
  }, [dot_data, isOpen]);

  const handleDownload = async (filename: string) => {
    let svgContent = svg_data;
    if (!svgContent) {
      let dotContent = dot_data;
      if (!dotContent) {
        dotContent = await fetchDotData("/api/job/" + filename + "/dot");
      }
      svgContent = await graphviz.dot(dotContent);
      setSvgData(svgContent);
    }
    console.log(svgContent);
    const blob = new Blob([svgContent], {
      type: "image/svg+xml;charset=utf-8",
    });
    const buffer = await blob.arrayBuffer();
    fileDownload(buffer, filename + ".svg");
  };
  return (
    <Flex>
      <button
        onClick={() => {
          fetch("api/job/" + props.value.job_id, {
            method: "PATCH",
            headers: {
              Accept: "application/json",
            },
          });
        }}
      >
        <FaStop color={"red"} title={"Stop this job"} />
      </button>
      <Box mx={2}></Box>
      <button
        onClick={async () => {
          await handleDownload(props.value.job_id);
        }}
      >
        <GrDocumentDownload title={"Download SVG Plan"} />
      </button>
      <Box mx={2}></Box>
      <button onClick={onOpen}>
        <GrOverview title={"View Graph"} />
      </button>
      <Modal isOpen={isOpen} size="large" onClose={onClose}>
        <ModalOverlay />
        <ModalContent>
          <ModalHeader textAlign={"center"}>
            {props.value.job_name} ({props.value.job_id})
          </ModalHeader>
          <ModalCloseButton />
          <ModalBody
            margin="auto"
            style={{
              width: "1200px",
              height: "100%",
              backgroundColor: "#aeaeae",
              padding: "0",
              display: "flex",
              justifyContent: "center",
              flexDirection: "column",
              alignItems: "center",
            }}
          >
            <TransformWrapper
              initialPositionX={100}
              initialPositionY={0}
              zoomAnimation={{ animationTime: 100 }}
            >
              {({ zoomIn, zoomOut, resetTransform, ...rest }) => (
                <React.Fragment>
                  <div className="tools">
                    <button onClick={() => zoomIn()}>
                      <GrZoomIn />
                    </button>
                    <button onClick={() => zoomOut()}>
                      <GrZoomOut />
                    </button>
                    <button onClick={() => resetTransform()}>
                      <GrContract />
                    </button>
                  </div>
                  <TransformComponent>
                    <SVG
                      innerRef={ref}
                      src={svg_data}
                      style={{ width: "1000", height: "1000" }}
                    />
                  </TransformComponent>
                </React.Fragment>
              )}
            </TransformWrapper>
          </ModalBody>
          <ModalFooter>
            <Button colorScheme="blue" mr={3} onClick={onClose}>
              Close
            </Button>
          </ModalFooter>
        </ModalContent>
      </Modal>
    </Flex>
  );
};

export const JobLinkCell = (props: any) => {
  const [stages, setData] = useState();
  const [loaded, setLoaded] = useState(false);
  const { isOpen, onOpen, onClose } = useDisclosure();

  const getStages = (url: string) => {
    fetch(url, {
      method: "GET",
      headers: {
        Accept: "application/json",
      },
    }).then(async (res) => {
      const jsonObj = await res.json();
      setData(jsonObj["stages"]);
    });
  };

  useEffect(() => {
    if (isOpen && !loaded) {
      getStages("/api/job/" + props.value + "/stages");
      setLoaded(true);
    }
  }, [stages, isOpen, loaded, props.value]);

  return (
    <Flex>
      <Link onClick={onOpen}>
        {props.value} <ExternalLinkIcon mx="2px" />
      </Link>
      <Modal isOpen={isOpen} size="small" onClose={onClose}>
        <ModalOverlay />
        <ModalContent>
          <ModalHeader textAlign={"center"}>
            Stages metrics for {props.value} job
          </ModalHeader>
          <ModalCloseButton />
          <ModalBody margin="auto">
            <JobStagesQueries stages={stages} />
          </ModalBody>
          <ModalFooter>
            <Button colorScheme="blue" mr={3} onClick={onClose}>
              Close
            </Button>
          </ModalFooter>
        </ModalContent>
      </Modal>
    </Flex>
  );
};

export const ProgressCell = (props: any) => {
  return (
    <CircularProgress value={props.value} color="orange.400">
      <CircularProgressLabel>{props.value}%</CircularProgressLabel>
    </CircularProgress>
  );
};

const columns: Column<Query>[] = [
  {
    Header: "Job ID",
    accessor: "job_id",
    Cell: JobLinkCell,
  },
  {
    Header: "Job Name",
    accessor: "job_name",
  },
  {
    Header: "Status",
    accessor: "job_status",
  },
  {
    Header: "Number of Stages",
    accessor: "num_stages",
  },
  {
    Header: "Output rows",
    accessor: "output_row_cnt",
  },
  {
    Header: "Start time",
    accessor: (row) => {
      const date = new Date((row as Query).start_time);
      return date.toLocaleString();
    },
  },
  {
    Header: "Elapsed",
    accessor: (row) => {
      const elapsed = (row as Query).elapsed;
      return elapsed + " ms";
    },
  },
  {
    Header: "Progress",
    accessor: "percent_complete",
    Cell: ProgressCell,
  },
  {
    Header: "Actions",
    accessor: (row) => ({
      job_id: (row as Query).job_id,
      job_name: (row as Query).job_name,
      status: (row as Query).job_status,
    }),
    id: "action_cell",
    Cell: ActionsCell,
  },
];

const getSkeleton = () => (
  <>
    <Skeleton height={5} />
    <Skeleton height={5} />
    <Skeleton height={5} />
    <Skeleton height={5} />
    <Skeleton height={5} />
    <Skeleton height={5} />
  </>
);

export const QueriesList: React.FunctionComponent<QueriesListProps> = ({
  queries,
  fetchJobs,
}) => {
  const isLoaded = typeof queries !== "undefined";

  return (
    <Box w={"100%"} flex={1}>
      <Button onClick={fetchJobs} colorScheme="blue" size="md" mb={5}>
        Refresh Job List
      </Button>
      {isLoaded ? (
        <DataTable
          columns={columns}
          data={queries || []}
          pageSize={10}
          pb={10}
        />
      ) : (
        getSkeleton()
      )}
    </Box>
  );
};
