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

# Can help when behind corporate network
export GOINSECURE="dmitri.shuralyov.com"
export GOPRIVATE=*

helpFunction()
{
   echo ""
   echo "Usage: $0 -t type"
   echo -e "\t-t The type parameters allows to select which Sedna example to build (joint_inference, federated_learning, etc..)"
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

# Uncomment this line if you want to push your images to a private repository
PRIVATE_DOCKER_REPOSITORY="registry-cbu.huawei.com"

dockerfiles_multiedgetracking=(
multi-edge-tracking-feature-extraction.Dockerfile
multi-edge-tracking-detection.Dockerfile
multi-edge-tracking-reid.Dockerfile
)

dockerfiles_dnn_partitioning=(
dnn-partitioning-alex-net-edge.Dockerfile
dnn-partitioning-alex-net-cloud.Dockerfile
)

dockerfiles_federated_learning=(
federated-learning-mistnet-yolo-aggregator.Dockerfile
federated-learning-mistnet-yolo-client.Dockerfile
federated-learning-surface-defect-detection-aggregation.Dockerfile
federated-learning-surface-defect-detection-aggregation-train.Dockerfile
)

dockerfiles_joint_inference=(
joint-inference-helmet-detection-big.Dockerfile
joint-inference-helmet-detection-little.Dockerfile
)

dockerfiles_lifelong_learning=(
lifelong-learning-atcii-classifier.Dockerfile
)

dockerfiles_incremental_learning=(
incremental-learning-helmet-detection.Dockerfile
)

case $type in

  multiedgetracking | cm5)
    dockerfiles=${dockerfiles_multiedgetracking[@]}
    ;;

  dnn_partitioning | cm6)
    dockerfiles=${dockerfiles_dnn_partitioning[@]}
    ;;

  federated_learning)
    dockerfiles=${dockerfiles_federated_learning[@]}
    ;;

  joint_inference)
    dockerfiles=${dockerfiles_joint_inference[@]}
    ;;

  lifelong_learning)
    dockerfiles=${dockerfiles_lifelong_learning[@]}
    ;;

  incremental_learning)
    dockerfiles=${dockerfiles_incremental_learning[@]}
    ;;

  all | *)
    dockerfiles+=( "${dockerfiles_multiedgetracking[@]}"
      "${dockerfiles_dnn_partitioning[@]}"
      "${dockerfiles_federated_learning[@]}"
      "${dockerfiles_joint_inference[@]}"
      "${dockerfiles_lifelong_learning[@]}"
      "${dockerfiles_incremental_learning[@]}")
    ;;
esac

# If no private Docker repo is set, fallback to the default one.
if [ -z ${PRIVATE_DOCKER_REPOSITORY+x} ]; then TARGET_REPO=${EXAMPLE_REPO_PREFIX}; else TARGET_REPO=${PRIVATE_DOCKER_REPOSITORY}/${EXAMPLE_REPO_PREFIX}; fi

for dockerfile in ${dockerfiles[@]}; do
  echo "Building $dockerfile" 
  example_name=${dockerfile/.Dockerfile}
  docker build -f $dockerfile -t ${TARGET_REPO}${example_name}:${IMAGE_TAG} --label sedna=examples ..
  docker push ${TARGET_REPO}${example_name}:${IMAGE_TAG}
done