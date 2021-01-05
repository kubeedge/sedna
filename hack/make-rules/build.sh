#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail

NEPTUNE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"

source "${NEPTUNE_ROOT}/hack/lib/init.sh"

neptune::golang::build_binaries "$@"
