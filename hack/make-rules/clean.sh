#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail

NEPTUNE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
source "${NEPTUNE_ROOT}/hack/lib/init.sh"

neptune::clean::cache(){
  GOARM= go clean -cache
}

neptune::clean::bin(){
  if [ -n "$NEPTUNE_OUTPUT_BINPATH" ]; then
    rm -rf $NEPTUNE_OUTPUT_BINPATH/*
  fi
}

neptune::clean::cache
neptune::clean::bin
