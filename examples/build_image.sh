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

helpFunction()
{
   echo ""
   echo "Usage: $0 -t type"
   echo -e "\t-t The type parameters allows to select which kind of images to build (edge, cloud, others, all)"
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

dockerfiles_cloud=(
# federated-learning-surface-defect-detection-aggregation.Dockerfile
# joint-inference-helmet-detection-big.Dockerfile
multi-edge-tracking-reid.Dockerfile
)

dockerfiles_edge=(
# federated-learning-surface-defect-detection-train.Dockerfile
# joint-inference-helmet-detection-little.Dockerfile
multi-edge-tracking-mot.Dockerfile
)

dockerfiles_others=(
#incremental-learning-helmet-detection.Dockerfile
#lifelong-learning-atcii-classifier.Dockerfile
)

case $type in

  cloud)
    dockerfiles=$dockerfiles_cloud
    ;;

  edge)
    dockerfiles=$dockerfiles_edge
    ;;

  others)
    dockerfiles=$dockerfiles_others
    ;;

  all | *)
    dockerfiles+=( "${dockerfiles_cloud[@]}" "${dockerfiles_edge[@]}" "${dockerfiles_others[@]}" )
    ;;
esac

for dockerfile in ${dockerfiles[@]}; do
  example_name=${dockerfile/.Dockerfile}
  docker build -f $dockerfile -t ${EXAMPLE_REPO_PREFIX}${example_name}:${IMAGE_TAG} --label sedna=examples ..
done