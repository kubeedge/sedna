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

# The root of the sedna
SEDNA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"

SEDNA_OUT_BINPATH="${SEDNA_ROOT}/${OUT_BINPATH:-_output/bin}"
SEDNA_OUT_IMAGEPATH="${SEDNA_ROOT}/${OUT_IMAGESPATH:-_output/images}"

readonly SEDNA_GO_PACKAGE="github.com/kubeedge/sedna"

source "${SEDNA_ROOT}/hack/lib/golang.sh"
source "${SEDNA_ROOT}/hack/lib/util.sh"
source "${SEDNA_ROOT}/hack/lib/buildx.sh"
