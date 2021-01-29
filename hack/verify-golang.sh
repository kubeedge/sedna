#!/usr/bin/env bash

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

# The root of the build/dist directory
SEDNA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"

source "${SEDNA_ROOT}/hack/lib/init.sh"

if ! command -v go 2>/dev/null; then
  cat >&2 <<EOF
Can't find 'go' in PATH, please fix and retry.
See http://golang.org/doc/install for installation instructions.
EOF
  exit 1
fi

echo "go detail version: $(go version)"

# output of $(go version):
# go version go1.15.6 linux/amd64
goversion=$(go version|awk '$0=$3'|sed 's/go//g')

X=$(echo $goversion|cut -d. -f1)
Y=$(echo $goversion|cut -d. -f2)

if [ $X -lt 1 ] ; then
  echo "go major version must >= 1, now is $X" >&2
  exit 1
fi

if [ $X -eq 1 -a $Y -lt 12 ] ; then
  echo "go minor version must >= 12 when major version is 1, now is $Y" >&2
  exit 1
fi
