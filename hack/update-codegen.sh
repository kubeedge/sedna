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

${SEDNA_ROOT}/hack/generate-groups.sh "deepcopy,client,informer,lister" \
${SEDNA_GO_PACKAGE}/pkg/client ${SEDNA_GO_PACKAGE}/pkg/apis \
"sedna:v1alpha1" \
--go-header-file ${SEDNA_ROOT}/hack/boilerplate/boilerplate.generatego.txt

# Check if Sedna home is different from the standard directory where GO projects are located
if [ "${GOPATH}/src/${SEDNA_GO_PACKAGE}/" != "$${SEDNA_ROOT}/" ]; then
    # Copy generated code into SEDNA_ROOT
    echo "Copying generated code from ${GOPATH}/src/${SEDNA_GO_PACKAGE}/pkg/ to ${SEDNA_ROOT}/ ..."
    cp -Rf ${GOPATH}/src/${SEDNA_GO_PACKAGE}/pkg ${SEDNA_ROOT}/
    if [ $? -eq 0 ]; then
        echo "Copy successful!"
    else
        echo "Error during copy!"
    fi
fi

