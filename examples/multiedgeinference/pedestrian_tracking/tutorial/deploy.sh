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

ZOO="zoo1"
KAFKA="kafka-broker0"
NFS_PATH="/data/network_shared/reid"
sp="/-\|"

function check_deployment_status {
  local deployment_ready_replicas=$(kubectl get deployments.apps $1 -o json | jq ".status.readyReplicas")
  local deployment_total_replicas=$(kubectl get deployments.apps $1 -o json | jq ".status.replicas")
  local result=$(($deployment_ready_replicas-$deployment_total_replicas))
  echo $result
}

usage="$(basename "$0") [-h] [-a MASTER_NODE_IP] [-p NFS_PATH]
Deploy the components required to run the pedestrian ReID example (Kafka, Zookeeper, NFS creation, CRDs, PV and PVC):
    -h  show this help text
    -a  master node ip
    -p  path where you want to store the ReID data and create the NFS (default=/data/network_shared)"

while getopts 'a:p:' flag
do
    case "$flag" in
        a) MASTER_NODE_IP="$OPTARG";;
        p) NFS_PATH="$OPTARG";;
        :) printf "missing argument for -%s\n" "$OPTARG" >&2; echo "$usage" >&2; exit 1;;
        \?) printf "illegal option: -%s\n" "$OPTARG" >&2; echo "$usage" >&2; exit 1;;
    esac
done

# mandatory arguments
if [ ! "$MASTER_NODE_IP" ]; then
  echo "argument -a must be provided"
  echo "$usage" >&2; exit 1
fi

echo "Node IP: $MASTER_NODE_IP";
echo "NFS Path: $NFS_PATH";

# Create a backup of the YAML files in case we need to restore them (only if the directory doesn't exist).
[ ! -d "../backup" ] && mkdir ../backup && cp -r ../yaml/* ../backup && echo "⚪ Created a backup copy of the YAML files in ../backup"

# Deploy Apache Kafka and Zookeper
echo "⚪ Provided master node IP: ${MASTER_NODE_IP}"
sed -i "s/MASTER_NODE_IP/${MASTER_NODE_IP}/" ../yaml/kafka/*.yaml

echo "⚪ Deploy Kafka broker and Zookeeper."
kubectl apply -f ../yaml/kafka/kafkabrk.yaml
kubectl apply -f ../yaml/kafka/kafkasvc.yaml
kubectl apply -f ../yaml/kafka/zoodeploy.yaml
kubectl apply -f ../yaml/kafka/zooservice.yaml

# check that kafka and zookeeper are running
echo "⚪ Verify that the Kafka broker and Zookeeper are running."

while [ $(check_deployment_status $ZOO) -ne 0 ]
do
  echo -ne "🟡 Zookeeper is not ready ${sp:i++%${#sp}:1} \r"
  sleep 0.2
done

echo "" && echo "🟢 Zookeeper is ready."

while [ $(check_deployment_status $KAFKA) -ne 0 ]
do
  echo -ne "🟡 Apache Kafka is not ready ${sp:i++%${#sp}:1} \r"
  sleep 0.2
done

echo "" && echo "🟢 Apache Kafka is ready."

# create CRDs for the models
echo "⚪ Create CRDs for AI models."
kubectl apply -f ../yaml/models/model_detection.yaml
kubectl apply -f ../yaml/models/model_m3l.yaml

echo "⚪ Check NFS setup."

if [ "$(sudo showmount -e localhost | grep -wo ${NFS_PATH})" == "${NFS_PATH}" ]; then
  echo "🟢 NFS directory already created."
else
  echo "🔴 NFS directory not created."
  echo "🟡 Attempting automated creation of NFS directory."
  sudo apt-get install -y nfs-kernel-server
  sudo mkdir -p $NFS_PATH
  sudo mkdir ${NFS_PATH}/processed
  sudo mkdir $NFS_PATH/query
  sudo mkdir $NFS_PATH/images
  sudo chmod 1777 $NFS_PATH
  sudo bash -c "echo '${NFS_PATH} *(rw,sync,no_root_squash,subtree_check)' >> /etc/exports"
  sudo exportfs -ra
  sudo showmount -e localhost
fi

# create CRDs for the PV and PVC
echo "⚪ Create PV and PVC."

sed -i "s/MASTER_NODE_IP/${MASTER_NODE_IP}/" ../yaml/pv/reid_volume.yaml
sed -i "s/NFS_PATH/${NFS_PATH}/" ../yaml/pv/reid_volume.yaml

kubectl apply -f ../yaml/pv/reid_volume.yaml
kubectl apply -f ../yaml/pvc/reid-volume-claim.yaml

# update apps YAML
echo "⚪ Update jobs and service YAML file with ${MASTER_NODE_IP} value."
sed -i "s/MASTER_NODE_IP/${MASTER_NODE_IP}/" ../yaml/video-analytics-job.yaml
sed -i "s/MASTER_NODE_IP/${MASTER_NODE_IP}/" ../yaml/feature-extraction-service.yaml

# exit
echo "🟢 Done!"





