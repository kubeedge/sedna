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


# Reset in case getopts has been used previously in the shell
OPTIND=1

usage()
{
   echo ""
   echo "Usage: $0 -r repository dir_1 ... dir_n"
   echo -e "\t-r The repository parameters allows to select a private Docker repository to upload the images to."
   echo -e "\tThe script expects a list of Sedna example to build (joint_inference, federated_learning, etc..).
   \tMultiple example can be built at the same time by passing a list of directories such as: dir_1 dir_2 ...
   \tIf no directory is specified, the script will automatically build all available examples."
   exit 1 # Exit script after printing help
}

while getopts "r:" opt
do
   case "$opt" in
      r ) IMAGE_REPO="$OPTARG" ;;
      ? ) usage ;; # Print usage in case parameter is non-existent
   esac
done

shift $((OPTIND-1))

[ "${1:-}" = "--" ] && shift

type=$@

if [ -z "$type" ]
then
   echo "No example directory/s specified, building all example images.."
   type="all"
fi

if [ -z "$IMAGE_REPO" ]
then
   echo "Using default Docker hub"
   IMAGE_REPO="kubeedge"
fi

cd "$(dirname "${BASH_SOURCE[0]}")"

IMAGE_TAG=${IMAGE_TAG:-v0.4.0} 
EXAMPLE_REPO_PREFIX=${IMAGE_REPO}/sedna-example-

dockerfiles_multiedgetracking=(
multi-edge-tracking-feature-extraction.Dockerfile
# multi-edge-tracking-gpu-feature-extraction.Dockerfile
# multi-edge-tracking-gpu-videoanalytics.Dockerfile
multi-edge-tracking-reid.Dockerfile
multi-edge-tracking-videoanalytics.Dockerfile
)

dockerfiles_federated_learning=(
federated-learning-mistnet-yolo-aggregator.Dockerfile
federated-learning-mistnet-yolo-client.Dockerfile
federated-learning-surface-defect-detection-aggregation.Dockerfile
federated-learning-surface-defect-detection-train.Dockerfile
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

# Iterate over the input folders and build them sequentially.
for tp in ${type[@]}; do
   if [[ "$tp" == "all" ]]; then
      dockerfiles+=(
         "${dockerfiles_multiedgetracking[@]}"
         "${dockerfiles_federated_learning[@]}"
         "${dockerfiles_joint_inference[@]}"
         "${dockerfiles_lifelong_learning[@]}"
         "${dockerfiles_incremental_learning[@]}")
   else
      dfiles=dockerfiles_$tp[@]
      dockerfiles+=("${!dfiles}")
   fi
done

# Removing duplicate entries (if any)
dockerfiles=($(echo "${dockerfiles[@]}" | tr ' ' '\n' | sort -u))

for dockerfile in ${dockerfiles[@]}; do
   echo "Building $dockerfile" 
   example_name=${dockerfile/.Dockerfile}
   docker build -f $dockerfile -t ${EXAMPLE_REPO_PREFIX}${example_name}:${IMAGE_TAG} --label sedna=examples ..
done
