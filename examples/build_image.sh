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

export GOINSECURE="dmitri.shuralyov.com"
export GOPRIVATE=*

helpFunction()
{
   echo ""
   echo "Usage: $0 -t type"
   echo -e "\t-t The type parameters allows to select which kind of images to build (cm5, cm6, others, all)"
   exit 1 # Exit script after printing help
}

while getopts "t:" opt
do
   case "$opt" in
      t ) type="$OPTARG" ;;
      ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done

if [ -z "$type" ]
then
   echo "Defaulting to building all example images..";
   parameterA="all"
fi

cd "$(dirname "${BASH_SOURCE[0]}")"

IMAGE_REPO=${IMAGE_REPO:-kubeedge}
IMAGE_TAG=${IMAGE_TAG:-v0.3.0}

EXAMPLE_REPO_PREFIX=${IMAGE_REPO}/sedna-example-

dockerfiles_cm5=(
multi-edge-tracking-feature-extraction.Dockerfile
multi-edge-tracking-detection.Dockerfile
multi-edge-tracking-reid.Dockerfile
)

dockerfiles_cm6=(
dnn-partitioning-alex-net-edge.Dockerfile
dnn-partitioning-alex-net-cloud.Dockerfile
)

dockerfiles_others=(
#incremental-learning-helmet-detection.Dockerfile
#lifelong-learning-atcii-classifier.Dockerfile
)

case $type in

  cm5)
    dockerfiles=${dockerfiles_cm5[@]}
    ;;

  cm6)
    dockerfiles=${dockerfiles_cm6[@]}
    ;;

  others)
    dockerfiles=${dockerfiles_others[@]}
    ;;

  all | *)
    dockerfiles+=( "${dockerfiles_cloud[@]}" "${dockerfiles_cm5[@]}" "${dockerfiles_cm6[@]}" "${dockerfiles_others[@]}" )
    ;;
esac

for dockerfile in ${dockerfiles[@]}; do
  echo "Building $dockerfile" 
  example_name=${dockerfile/.Dockerfile}
  docker build -f $dockerfile -t ${EXAMPLE_REPO_PREFIX}${example_name}:${IMAGE_TAG} --label sedna=examples ..
  docker tag ${EXAMPLE_REPO_PREFIX}${example_name}:${IMAGE_TAG} registry-cbu.huawei.com/${EXAMPLE_REPO_PREFIX}${example_name}:${IMAGE_TAG}
  docker push registry-cbu.huawei.com/${EXAMPLE_REPO_PREFIX}${example_name}:${IMAGE_TAG}
done