#!/usr/bin/env bash

# Copyright 2020 The KubeEdge Authors.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -o errexit
set -o nounset
set -o pipefail

SEDNA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"

source "${SEDNA_ROOT}/hack/lib/init.sh"

export PATH=$PATH:$(go env GOPATH)/bin

install_golangci-lint() {
  echo "installing golangci-lint."
  curl -sSfL https://raw.githubusercontent.com/golangci/golangci-lint/master/install.sh | sh -s -- -b $(go env GOPATH)/bin v1.34.1
  if [[ $? -ne 0 ]]; then
    echo "failed to install golangci-lint, exiting."
    exit 1
  fi
}

check_golangci-lint() {
  echo "checking golangci-lint"
  
  if ! command -v golangci-lint >/dev/null; then
    install_golangci-lint
    # check again
    command -v golangci-lint >/dev/null
  fi
}

check_golangci-lint

golangci-lint run -c .golangci.yml -v
