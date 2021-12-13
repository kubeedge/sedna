# Copyright 2021 The KubeEdge Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Add cross buildx improvement
# _speed_buildx_for_go_
FROM golang:1.16-alpine3.15 AS builder
LABEL stage=builder

ARG GO_LDFLAGS

# install build tools
RUN apk update
RUN apk add build-base bash

WORKDIR /code
# copy source
COPY . .
RUN make build WHAT=gm GO_LDFLAGS=$GO_LDFLAGS OUT_DIR=_output

# ALT: just using go build
# RUN CGO_ENABLED=0 go build -o _output/bin/sedna-gm -ldflags "$GO_LDFLAGS -w -s" cmd/sedna-gm/sedna-gm.go

FROM alpine:3.11

COPY --from=builder /code/_output/bin/sedna-gm /usr/local/bin/sedna-gm

COPY build/gm/gm-config.yaml /gm.yaml

CMD ["sedna-gm", "--config", "/gm.yaml"]
