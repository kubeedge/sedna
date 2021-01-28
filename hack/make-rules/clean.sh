#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail

SEDNA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
source "${SEDNA_ROOT}/hack/lib/init.sh"

sedna::clean::cache(){
  GOARM= go clean -cache
}

sedna::clean::bin(){
  if [ -n "$SEDNA_OUTPUT_BINPATH" ]; then
    rm -rf $SEDNA_OUTPUT_BINPATH/*
  fi
}

sedna::clean::cache
sedna::clean::bin
