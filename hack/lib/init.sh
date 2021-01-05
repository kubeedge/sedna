#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail

# The root of the neptune
NEPTUNE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"

NEPTUNE_OUTPUT_SUBPATH="${NEPTUNE_OUTPUT_SUBPATH:-_output/local}"
NEPTUNE_OUTPUT="${NEPTUNE_ROOT}/${NEPTUNE_OUTPUT_SUBPATH}"
NEPTUNE_OUTPUT_BINPATH="${NEPTUNE_OUTPUT}/bin"

readonly NEPTUNE_GO_PACKAGE="github.com/edgeai-neptune/neptune"

source "${NEPTUNE_ROOT}/hack/lib/golang.sh"
source "${NEPTUNE_ROOT}/hack/lib/util.sh"
