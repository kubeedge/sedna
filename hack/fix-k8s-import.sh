#!/bin/sh

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

# fix these invalid 0.0.0 version requires of k8s.io/kubernetes
# modified the script from https://github.com/kubernetes/kubernetes/issues/79384#issuecomment-521493597

VERSION=${1#"v"}
if [ -z "$VERSION" ]; then
    echo "Must specify version!"
    exit 1
fi
MODS=($(
    curl -sS https://raw.githubusercontent.com/kubernetes/kubernetes/v${VERSION}/go.mod |
    sed -n 's|.*\(k8s.io/.*\) => ./staging/src/k8s.io/.*|\1|p'
))
for MOD in "${MODS[@]}"; do
  echo fixing $MOD
  V=v${VERSION/1/0}
  go mod edit "-replace=${MOD}=${MOD}@${V}"
done
