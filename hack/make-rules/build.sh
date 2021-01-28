#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail

SEDNA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"

source "${SEDNA_ROOT}/hack/lib/init.sh"

sedna::golang::build_binaries "$@"
