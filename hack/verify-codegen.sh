#!/bin/bash

set -o errexit
set -o nounset
set -o pipefail

SEDNA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"

CLIENT_ROOT="${SEDNA_ROOT}/pkg/client"
ZZ_FILE="zz_generated.deepcopy.go"

UPDATE_SCRIPT="hack/update-codegen.sh"
"${SEDNA_ROOT}/$UPDATE_SCRIPT"

if git status --short 2>/dev/null | grep -qE "${CLIENT_ROOT}|${ZZ_FILE}"; then
  echo "FAILED: codegen verify failed." >&2
  echo "Please run the command to update your codegen files: $UPDATE_SCRIPT" >&2
  exit 1
else
  echo "SUCCESS: codegen verified."
fi
