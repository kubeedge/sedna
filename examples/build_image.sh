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

cd "$(dirname "${BASH_SOURCE[0]}")"

IMAGE_REPO=${IMAGE_REPO:-kubeedge}
IMAGE_TAG=${IMAGE_TAG:-v0.1.0}

EXAMPLE_REPO_PREFIX=${IMAGE_REPO}/sedna-example-

dockerfiles=(
federated-learning-surface-defect-detection-aggregation.Dockerfile
federated-learning-surface-defect-detection-train.Dockerfile
incremental-learning-helmet-detection.Dockerfile
joint-inference-helmet-detection-big.Dockerfile
joint-inference-helmet-detection-little.Dockerfile
)

for dockerfile in ${dockerfiles[@]}; do
  example_name=${dockerfile/.Dockerfile}
  docker build -f $dockerfile -t ${EXAMPLE_REPO_PREFIX}${example_name}:${IMAGE_TAG} --label sedna=examples ..
done
