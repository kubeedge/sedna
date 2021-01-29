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

CLIENT_ROOT="${SEDNA_ROOT}/pkg/client"
ZZ_FILE="zz_generated.deepcopy.go"

UPDATE_SCRIPT="hack/update-codegen.sh"
"${SEDNA_ROOT}/$UPDATE_SCRIPT"

if git status --short 2>/dev/null | grep -qE "${CLIENT_ROOT}|${ZZ_FILE}"; then
  echo "FAILED: codegen verify failed." >&2
  echo "Please run the command to update your codegen files: $UPDATE_SCRIPT" >&2
  exit 1
else
  echo "SUCCESS: codegen verified."
fi
