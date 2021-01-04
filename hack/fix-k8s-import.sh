#!/bin/sh

# fix these invalid 0.0.0 version requires of k8s.io/kubernetes
# modified the script from https://github.com/kubernetes/kubernetes/issues/79384#issuecomment-521493597

VERSION=${1#"v"}
if [ -z "$VERSION" ]; then
    echo "Must specify version!"
    exit 1
fi
MODS=($(
    curl -sS https://raw.githubusercontent.com/kubernetes/kubernetes/v${VERSION}/go.mod |
    sed -n 's|.*\(k8s.io/.*\) => ./staging/src/k8s.io/.*|\1|p'
))
for MOD in "${MODS[@]}"; do
  echo fixing $MOD
  V=v${VERSION/1/0}
  go mod edit "-replace=${MOD}=${MOD}@${V}"
done
