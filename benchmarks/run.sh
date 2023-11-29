#!/bin/bash
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

set -e
set -x

remote_how_arg=$1
shift # Shift the arguments to start processing the next ones as query numbers

# Rest of your script description

# Loop through all the remaining arguments which are the query numbers
for query in "$@"
do
  /root/tpch benchmark ballista --host ballista-scheduler --port 50050 --query $query --path /data/data-parquet --format parquet --iterations 1 --remote-how $remote_how_arg
done
