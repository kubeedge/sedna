#!/bin/bash

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

set -o errexit
set -o nounset
set -o pipefail

SEDNA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"

cd "$SEDNA_ROOT"

# Prepare all-in-one env
{
  __WITH_SOURCE__=true
  # this will export KUBECONFIG
  echo "Prepare all-in-one env" 
  cat scripts/installation/all-in-one.sh | KUBEEDGE_VERSION=v1.8.0 NUM_EDGE_NODES=0 bash -
}

# Running e2e
echo "Running e2e..."
go test ./test/e2e -v


# Clean all-in-one env
echo "Chean all-in-one env"
cat scripts/installation/all-in-one.sh | bash /dev/stdin clean
