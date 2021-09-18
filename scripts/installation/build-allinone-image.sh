#!/bin/bash

# Copyright 2021 The KubeEdge Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This script builds the node image for all-in-one Sedna.

set -o errexit
set -o nounset
set -o pipefail


# just reuse kind image
# https://github.com/kubernetes-sigs/kind/blob/4910c3e221a858e68e29f9494170a38e1c4e8b80/pkg/build/nodeimage/defaults.go#L23
# 
# Note: here use v1.21.1 of kindest/node, because kubeedge-1.8.0 still uses apiextensions.k8s.io/v1beta1 of CRD, which will be removed in k8s 1.22.0
readonly BASE_IMAGE=kindest/node:v1.21.1

readonly BUILD_IMAGE_NAME=sedna-edge-build-$RANDOM

function get_latest_version() {
  # get the latest version of specified gh repo
  local repo=${1}
  # output of this latest page:
  # ...
  # "tag_name": "v1.0.0",
  # ...
  curl -s https://api.github.com/repos/$repo/releases/latest | awk '/"tag_name":/&&$0=$2' | sed 's/[",]//g'
}

function create_build_container() {
  docker run --rm --name $BUILD_IMAGE_NAME -d --entrypoint sleep $BASE_IMAGE inf || true

  if [ -z "$RETAIN_BUILD_CONTAINER" ]; then
    trap clean_build_container EXIT
  fi
}

function clean_build_container() {
  docker stop $BUILD_IMAGE_NAME
}

function run_in_build_container() {
  docker exec -i $BUILD_IMAGE_NAME "$@"
}

function commit_build_container() {
  local commit_args=(
    docker commit
    # put entrypoint back
    # https://github.com/kubernetes-sigs/kind/blob/4910c3e221a858e68e29f9494170a38e1c4e8b80/images/base/Dockerfile#L203
    --change 'ENTRYPOINT [ "/usr/local/bin/entrypoint", "/sbin/init" ]'
    $BUILD_IMAGE_NAME $ALLINONE_NODE_IMAGE
  )
  "${commit_args[@]}"
}

function gen_edgecore_config_template() {
  run_in_build_container mkdir -p /etc/kubeedge
  cat | run_in_build_container mkdir -p /etc/kubeedge
}

function arch() {
  local arch=$(uname -m)
  case "$arch" in
    x86_64) arch=amd64;;
    *);;
  esac
  echo "$arch"
}

function install_keadm() {
  # download the specified keadm binary
  local arch=$(arch) version=${KUBEEDGE_VERSION}
  local tarfile=keadm-$version-linux-$arch.tar.gz 
  local path=keadm-$version-linux-$arch/keadm/keadm
  local configdir=/etc/kubeedge

  run_in_build_container bash -euc "
    # copy kube config file
    # install keadm
    curl --fail -LSO https://github.com/kubeedge/kubeedge/releases/download/$version/$tarfile
    tar -xvf $tarfile $path
    mv $path /usr/local/bin/
    rm $tarfile

    # install dependencies of keadm init/join
    apt update -y
    apt install -y wget sudo mosquitto
    systemctl enable mosquitto

    # install debug tools
    apt install -y less sqlite3

    # add convenient command
    echo 'alias docker=crictl' > ~/.bash_aliases
  "
  
  # download the kubeedge into the docker image in advance
  download_kubeedge
}

function download_kubeedge() {
  # download the specified kubeedge package for keadm
  local arch=$(arch) version=${KUBEEDGE_VERSION}
  local tarfile=kubeedge-$version-linux-$arch.tar.gz 
  local configdir=/etc/kubeedge

  run_in_build_container bash -euc "
    mkdir -p $configdir; cd $configdir
    curl --fail -LSO https://github.com/kubeedge/kubeedge/releases/download/$version/$tarfile
  "
}

function install_edgecore() {
  # download the specified edgecore binary
  local arch=$(arch) version=${KUBEEDGE_VERSION}
  local tarfile=kubeedge-$version-linux-$arch.tar.gz 
  local edgecorepath=kubeedge-$version-linux-$arch/edge/edgecore
  local configdir=/etc/kubeedge

  run_in_build_container bash -euc "
    mkdir -p $configdir; cd $configdir
    curl --fail -LSO https://github.com/kubeedge/kubeedge/releases/download/$version/$tarfile

    tar -xvf $tarfile $edgecorepath
    mv $edgecorepath /usr/local/bin/
    rm $tarfile

    # install service
    curl --fail -LSO https://raw.githubusercontent.com/kubeedge/kubeedge/$version/build/tools/edgecore.service
    systemctl enable $configdir/edgecore.service
  "
}

: ${KUBEEDGE_VERSION:=$(get_latest_version kubeedge/kubeedge)}

: ${NODE_IMAGE:=kubeedge/sedna-allinone-node:v1.21.1}
: ${RETAIN_BUILD_CONTAINER:=}

create_build_container
install_keadm
commit_build_container

