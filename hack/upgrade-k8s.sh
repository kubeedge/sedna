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

version=v${1#"v"}
if [ -z "$version" ]; then
    echo "Must specify the Kubernetes version!"
    exit 1
fi

go mod edit "-require=k8s.io/kubernetes@$version"
bash "$SEDNA_ROOT/hack/fix-k8s-import.sh" $version

script_and_directories=$(cat <<EOF
hack/update-vendor.sh              vendor go.mod go.sum
hack/update-vendor-licenses.sh     LICENSES
hack/update-codegen.sh             pkg/apis pkg/client
EOF
)

while read run_script directories; do
  if [ -f "$run_script" ]; then
    bash "$SEDNA_ROOT/$run_script"
    git add $directories
  fi
done < <(echo "$script_and_directories")

make crds
git add build/crds
