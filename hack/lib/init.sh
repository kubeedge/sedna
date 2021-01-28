#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail

# The root of the sedna
SEDNA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"

SEDNA_OUT_BINPATH="${SEDNA_ROOT}/${OUT_BINPATH:-_output/bin}"

readonly SEDNA_GO_PACKAGE="github.com/kubeedge/sedna"

source "${SEDNA_ROOT}/hack/lib/golang.sh"
source "${SEDNA_ROOT}/hack/lib/util.sh"
