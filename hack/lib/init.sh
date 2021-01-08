#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail

# The root of the neptune
NEPTUNE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"

NEPTUNE_OUT_BINPATH="${NEPTUNE_ROOT}/${OUT_BINPATH:-_output/bin}"

readonly NEPTUNE_GO_PACKAGE="github.com/edgeai-neptune/neptune"

source "${NEPTUNE_ROOT}/hack/lib/golang.sh"
source "${NEPTUNE_ROOT}/hack/lib/util.sh"
