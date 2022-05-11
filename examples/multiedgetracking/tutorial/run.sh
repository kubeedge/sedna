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

VA_LABEL="videoanalyticsjob.sedna.io/name=video-analytics"
REID_LABEL="reidjob.sedna.io/name=reid"
FE_LABEL="app.sedna.io=feature-extraction-fe-svc"
sp="/-\|"

function check_pod_status {
  local func_result="$(kubectl get pod -l $1 -o json | jq ".items[0].status.phase")"
  echo "$func_result"
}

### FEATURE EXTRACTION ###

# create feature extraction service 
echo "⚪ Create Feature Extraction service."
kubectl apply -f ../yaml/feature-extraction-service.yaml

# check that FE is running
while [ $(check_pod_status $FE_LABEL) != '"Running"' ]
do
  echo -ne "🟡 Feature Extraction service is not ready ${sp:i++%${#sp}:1} \r"
  sleep 0.2
done

echo -n "" && echo "🟢 Feature Extraction service is ready."

### VIDEO ANALYTICS ###

# create VideoAnalytics Job
echo "⚪ Create VideoAnalytics job."
kubectl apply -f ../yaml/video-analytics-job.yaml

# wait for the job to be ready
while [ $(check_pod_status $VA_LABEL) != '"Running"' ] && [ $(check_pod_status $VA_LABEL) != '"Succeeded"' ]
do
  echo -ne "🟡 VideoAnalytics job is not ready ${sp:i++%${#sp}:1} \r"
  sleep 0.2
done

echo -n "" && echo "🟢 VideoAnalytics job is ready."

# wait for the job to complete
while [ $(check_pod_status $VA_LABEL) != '"Succeeded"' ]
do
  echo -ne "🟡 Waiting for VideoAnalytics job completion (this will take awhile) ${sp:i++%${#sp}:1} \r"
  sleep 0.2
done

echo -n "" && echo "🟢 VideoAnalytics job has completed."
echo "⚪ Clean-up VideoAnalytics job resources."
kubectl delete -f ../yaml/video-analytics-job.yaml

### REID ###

# create ReID job
echo "⚪ Create ReID job."
kubectl apply -f ../yaml/reid-job.yaml

# wait for the job to be ready
while [ $(check_pod_status $REID_LABEL) != '"Running"' ] && [ $(check_pod_status $REID_LABEL) != '"Succeeded"' ]
do
  echo -ne "🟡 ReID job is not ready ${sp:i++%${#sp}:1} \r"
  sleep 0.2
done

echo -n "" && echo "🟢 ReID job is ready."

# wait for the job to complete
while [ $(check_pod_status $REID_LABEL) != '"Succeeded"' ]
do
  echo -ne "🟡 Waiting for ReID job completion (this will take awhile) ${sp:i++%${#sp}:1} \r"
  sleep 0.2
done

echo -n "" &&  echo "🟢 ReID job has completed."
echo "⚪ Clean-up ReID job resources."
kubectl delete -f ../yaml/reid-job.yaml

# remove feature extraction service
echo "⚪ Remove Feature Extraction service."
kubectl delete -f ../yaml/feature-extraction-service.yaml

# restart kafka pod to clean dirty topics
echo "⚪ Clean-up Kafka broker."
kubectl delete pod -l app=kafka

# exit
echo "🟢 Done!"





