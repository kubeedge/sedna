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

export IMAGE_REPO=${IMAGE_REPO:-kubeedge}
export IMAGE_TAG=${IMAGE_TAG:-v0.3.0}

bash build_image.sh

for i in $(
    docker images --filter label=sedna=examples |
    grep -F "$IMAGE_REPO" |
    grep -F "$IMAGE_TAG" |
    awk '$0=$1":"$2'
 ); do
 docker push $i && {
   echo done docker push $i
 } || {
   echo failed to docker push $i
 }
done
