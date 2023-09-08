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

FROM ubuntu:22.04 

ARG EXT_UID

ENV RUST_LOG=info
ENV RUST_BACKTRACE=full
ENV DEBIAN_FRONTEND=noninteractive
ENV NODE_MAJOR=18

RUN apt-get update && \
    apt-get -y install libssl-dev openssl zlib1g zlib1g-dev ca-certificates gnupg libpq-dev cmake protobuf-compiler netcat curl unzip

RUN mkdir -p /etc/apt/keyrings && \
    curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg && \
    echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_${NODE_MAJOR}.x nodistro main" | tee /etc/apt/sources.list.d/nodesource.list && \
    apt-get update && \
    apt-get install -y nodejs && \
    npm install -g yarn

# create build user with same UID as 
RUN adduser -q -u $EXT_UID builder --home /home/builder && \
    mkdir -p /home/builder/workspace
USER builder

ENV HOME=/home/builder
ENV PATH=$HOME/.cargo/bin:$PATH

# prepare rust
RUN curl https://sh.rustup.rs -sSf | bash -s -- -y && \
    rustup component add rustfmt && \
    cargo install cargo-chef --version 0.1.34

WORKDIR /home/builder/workspace

COPY dev/docker/builder-entrypoint.sh /home/builder
ENTRYPOINT ["/home/builder/builder-entrypoint.sh"]
