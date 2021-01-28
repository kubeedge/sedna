#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail

# The root of the build/dist directory
SEDNA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"

source "${SEDNA_ROOT}/hack/lib/init.sh"

if ! command -v go 2>/dev/null; then
  cat >&2 <<EOF
Can't find 'go' in PATH, please fix and retry.
See http://golang.org/doc/install for installation instructions.
EOF
  exit 1
fi

echo "go detail version: $(go version)"

# output of $(go version):
# go version go1.15.6 linux/amd64
goversion=$(go version|awk '$0=$3'|sed 's/go//g')

X=$(echo $goversion|cut -d. -f1)
Y=$(echo $goversion|cut -d. -f2)

if [ $X -lt 1 ] ; then
  echo "go major version must >= 1, now is $X" >&2
  exit 1
fi

if [ $X -eq 1 -a $Y -lt 12 ] ; then
  echo "go minor version must >= 12 when major version is 1, now is $Y" >&2
  exit 1
fi
