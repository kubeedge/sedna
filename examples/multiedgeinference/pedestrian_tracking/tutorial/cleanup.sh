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


echo "⚪ Removing ALL resources created for pedestrian ReID."
kubectl delete -f ../yaml/feature-extraction-service.yaml
kubectl delete -f ../yaml/reid-job.yaml
kubectl delete -f ../yaml/video-analytics-job.yaml
kubectl delete -f ../yaml/pv/*.yaml
kubectl delete -f ../yaml/pvc/*.yaml
kubectl delete -f ../yaml/kafka/*.yaml
kubectl delete -f ../yaml/models/*.yaml

# exit
echo "🟢 Done!"