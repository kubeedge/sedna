#!/bin/bash

set -o errexit
set -o nounset
set -o pipefail

SCRIPT_ROOT=$(unset CDPATH && cd $(dirname "${BASH_SOURCE[0]}")/.. && pwd)

${SCRIPT_ROOT}/hack/generate-groups.sh "deepcopy,client,informer,lister" \
github.com/edgeai-neptune/neptune/pkg/client github.com/edgeai-neptune/neptune/pkg/apis \
"neptune:v1alpha1" \
--go-header-file ${SCRIPT_ROOT}/hack/boilerplate/boilerplate.txt
