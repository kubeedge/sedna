#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail

SEDNA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"

UPDATE_SCRIPT=hack/update-vendor.sh
${SEDNA_ROOT}/${UPDATE_SCRIPT}
 
if git status --short 2>/dev/null | grep -qE 'go\.mod|go\.sum|vendor/'; then
  echo "FAILED: vendor verify failed." >&2
  echo "Please run the command to update your vendor directories: $UPDATE_SCRIPT" >&2
  exit 1
else
  echo "SUCCESS: vendor verified."
fi
