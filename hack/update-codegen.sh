#!/bin/bash

set -o errexit
set -o nounset
set -o pipefail

NEPTUNE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"

source ${NEPTUNE_ROOT}/hack/lib/init.sh

${NEPTUNE_ROOT}/hack/generate-groups.sh "deepcopy,client,informer,lister" \
${NEPTUNE_GO_PACKAGE}/pkg/client ${NEPTUNE_GO_PACKAGE}/pkg/apis \
"neptune:v1alpha1" \
--go-header-file ${NEPTUNE_ROOT}/hack/boilerplate/boilerplate.txt
