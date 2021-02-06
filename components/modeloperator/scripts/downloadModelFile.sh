#!/bin/sh

set -euo pipefail
modelname=$1
version=$2
url=$3
sha256sum=$4

modelfilename=$(basename $url)
basedir="/var/lib/sedna/models/$modelname"
mkdir -p $basedir

write_config_file () {
cat >"$basedir/models.config" <<EOF
model_config_list {
  config {
    name: '$modelname'
    base_path: '/models/$modelname'
    model_platform: 'tensorflow'
    model_version_policy {
      specific {
        versions: $1
      }
    }
  }
}
EOF
}

if [ -d "$basedir/$version" ]; then
  echo "version $version file of model is already deployed, writing config file and exiting."
  write_config_file $version
  exit 0
fi

mkdir -p "$basedir/cache"
cd "$basedir/cache"
if [ $(echo "$sha256sum"'  '"$modelfilename" | sha256sum -c -s;echo $?) -eq 0 ]; then
  echo "$modelfilename already exists in cache"
else
  echo "$modelfilename not found in cache, downloading it from $url now"
  rm -f "$modelfilename"
  wget $url
  if [ $(echo "$sha256sum"'  '"$modelfilename" | sha256sum -c -s;echo $?) -ne 0 ]; then
    echo "downloaded file does not have the expected sha256sum, exiting"
    exit 1
  fi
fi

echo "deploying the model file $modelfilename"
targetDir="$basedir/$version"
mkdir -p $targetDir
tar xzf "$modelfilename" -C $targetDir --strip-components=1
rm -f $modelfilename

write_config_file $version
echo "$modelfilename successfully deployed"