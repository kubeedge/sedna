#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail

echo "running 'go mod tidy'"
go mod tidy

echo "running 'go mod vendor'"
go mod vendor
