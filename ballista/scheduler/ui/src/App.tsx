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

import React, { useState, useEffect } from "react";
import {
  Box,
  Grid,
  Tab,
  TabList,
  TabPanel,
  TabPanels,
  Tabs,
  VStack,
} from "@chakra-ui/react";
import { Header } from "./components/Header";
import { Summary } from "./components/Summary";
import { ExecutorsList } from "./components/ExecutorsList";
import { QueriesList } from "./components/QueriesList";
import { Footer } from "./components/Footer";
import "./App.css";

const App: React.FunctionComponent<any> = () => {
  const [schedulerState, setSchedulerState] = useState(undefined);
  const [jobs, setJobs] = useState(undefined);
  const [executors, setExecutors] = useState(undefined);

  async function getSchedulerState() {
    const res = await fetch(`/api/state`, {
      method: "POST",
      headers: {
        Accept: "application/json",
      },
    });
    const res_1 = await res.json();
    return setSchedulerState(res_1);
  }

  async function getJobs() {
    const res = await fetch(`/api/jobs`, {
      method: "POST",
      headers: {
        Accept: "application/json",
      },
    });
    const res_1 = await res.json();
    return setJobs(res_1);
  }

  async function getExecutors() {
    const res = await fetch(`/api/executors`, {
      method: "POST",
      headers: {
        Accept: "application/json",
      },
    });
    const res_1 = await res.json();
    return setExecutors(res_1);
  }

  useEffect(() => {
    getSchedulerState();
    getJobs();
    getExecutors();
  }, []);

  return (
    <Box>
      <Grid minH="100vh">
        <VStack alignItems={"flex-start"} spacing={0} width={"100%"}>
          <Header schedulerState={schedulerState} />
          <Summary schedulerState={schedulerState} />
          <Tabs width={"100%"}>
            <TabList>
              <Tab>Jobs</Tab>
              <Tab>Executors</Tab>
            </TabList>
            <TabPanels>
              <TabPanel>
                <QueriesList queries={jobs} fetchJobs={getJobs} />
              </TabPanel>
              <TabPanel>
                <ExecutorsList executors={executors} />
              </TabPanel>
            </TabPanels>
          </Tabs>
          <Footer />
        </VStack>
      </Grid>
    </Box>
  );
};

export default App;
