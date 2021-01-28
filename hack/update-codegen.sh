#!/bin/bash

set -o errexit
set -o nounset
set -o pipefail

SEDNA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"

source ${SEDNA_ROOT}/hack/lib/init.sh

${SEDNA_ROOT}/hack/generate-groups.sh "deepcopy,client,informer,lister" \
${SEDNA_GO_PACKAGE}/pkg/client ${SEDNA_GO_PACKAGE}/pkg/apis \
"sedna:v1alpha1" \
--go-header-file ${SEDNA_ROOT}/hack/boilerplate/boilerplate.txt
