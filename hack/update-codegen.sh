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

source ${SEDNA_ROOT}/hack/lib/init.sh

API_VERSIONS=$(ls -d "${SEDNA_ROOT}/pkg/apis/sedna"/*/ 2>/dev/null | xargs -n 1 basename | sort | paste -sd, -)

if [[ -z "${API_VERSIONS}" ]]; then
    echo "Error: No API versions found in ${SEDNA_GO_PACKAGE}/pkg/apis/sedna"
    exit 1
fi

GROUP_VERSIONS="sedna:${API_VERSIONS}"

"${SEDNA_ROOT}/hack/generate-groups.sh" "deepcopy,client,informer,lister" \
  "${SEDNA_GO_PACKAGE}/pkg/client" "${SEDNA_GO_PACKAGE}/pkg/apis" \
  "${GROUP_VERSIONS}" \
  --go-header-file "${SEDNA_ROOT}/hack/boilerplate/boilerplate.generatego.txt"
