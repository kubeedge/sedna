#!/usr/bin/env bash

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

set -o errexit
set -o nounset
set -o pipefail

sedna::buildx::prepare_env() {
  # Check whether buildx exists.
  if ! docker buildx >/dev/null 2>&1; then
    echo "ERROR: docker buildx not available. Docker 19.03 or higher is required with experimental features enabled" >&2
    exit 1
  fi

  # Use tonistiigi/binfmt that is to enable an execution of different multi-architecture containers
  docker run --privileged --rm tonistiigi/binfmt --install all

  # Create a new builder which gives access to the new multi-architecture features.
  builder_instance="sedna-buildx"
  if ! docker buildx inspect $builder_instance >/dev/null 2>&1; then
    docker buildx create --use --name $builder_instance --driver docker-container
  fi
  docker buildx use $builder_instance
}

sedna::buildx:generate-dockerfile() {
  dockerfile=${1}
  sed "/AS builder/s/FROM/FROM --platform=\$BUILDPLATFORM/g" ${dockerfile}
}

sedna::buildx::push-multi-platform-images() {
  sedna::buildx::prepare_env

  bash scripts/storage-initializer/push_image.sh

  for component in ${COMPONENTS[@]}; do
    echo "pushing ${PLATFORMS} image for $component"

    temp_dockerfile=build/${component}/buildx_dockerfile
    sedna::buildx:generate-dockerfile build/${component}/Dockerfile > ${temp_dockerfile}

    docker buildx build --push \
      --build-arg GO_LDFLAGS=${GO_LDFLAGS} \
      --platform ${PLATFORMS} \
      -t ${IMAGE_REPO}/sedna-${component}:${IMAGE_TAG} \
      -f ${temp_dockerfile} .

    rm ${temp_dockerfile}
  done
}

sedna::buildx::build-multi-platform-images() {
  sedna::buildx::prepare_env

  mkdir -p ${OUT_IMAGESPATH}
  arch_array=(${PLATFORMS//,/ })

  temp_dockerfile=${OUT_IMAGESPATH}/buildx_dockerfile
  for component in ${COMPONENTS[@]}; do
    echo "building ${PLATFORMS} image for ${component}"

    sedna::buildx:generate-dockerfile build/${component}/Dockerfile > ${temp_dockerfile}

    for arch in ${arch_array[@]}; do
      dest_tar=${OUT_IMAGESPATH}/${component}-${IMAGE_TAG}-${arch////-}.tar
      echo "building ${arch} image for ${component} and the image will be saved in ${dest_tar}"

      docker buildx build -o type=docker,dest=${dest_tar} \
        --build-arg GO_LDFLAGS=${GO_LDFLAGS} \
        --platform ${arch} \
        -t ${IMAGE_REPO}/sedna-${component}:${IMAGE_TAG} \
        -f ${temp_dockerfile} .
      done
  done

  rm ${temp_dockerfile}
}
