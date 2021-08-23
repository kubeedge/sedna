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

CLIENT_PATTERN="pkg/client/"
ZZ_FILE="zz_generated.deepcopy.go"
CODEGEN_PATTERN="${CLIENT_PATTERN}|${ZZ_FILE}"

UPDATE_SCRIPT="hack/update-codegen.sh"

"${SEDNA_ROOT}/$UPDATE_SCRIPT"

if dirty_files=$(
      git status --porcelain |
      awk '$0=$2' |
      grep -E "$CODEGEN_PATTERN" |
      sed 's/^/  /'); then
  echo "FAILED: codegen verify failed." >&2
  echo "Please run the command '$UPDATE_SCRIPT' to update your codegen files:" >&2
  echo "$dirty_files" >&2
  exit 1

elif [ $? -eq 1 ]; then  # grep exit 1 when no matches
  echo "SUCCESS: codegen verified."
fi
