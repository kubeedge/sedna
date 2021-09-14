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

# Influential env vars:
#
# SEDNA_GM_NODE   | required | The node which Sedna gm will be deployed at
# SEDNA_ACTION    | optional | 'create'/'clean', default is 'create'
# SEDNA_VERSION   | optional | The Sedna version to be installed.
#                              if not specified, it will get latest release version.
# SEDNA_ROOT      | optional | The Sedna offline directory

set -o errexit
set -o nounset
set -o pipefail

TMP_DIR=$(mktemp -d --suffix=.sedna)
SEDNA_ROOT=${SEDNA_ROOT:-$TMP_DIR}

GM_NODE_NAME=${SEDNA_GM_NODE:-}
KB_NODE_NAME=${SEDNA_GM_NODE:-}

DEFAULT_SEDNA_VERSION=v0.4.0


trap "rm -rf '$TMP_DIR'" EXIT 

get_latest_version() {
  # get Sedna latest release version
  local repo=kubeedge/sedna
  # output of this latest page:
  # ...
  # "tag_name": "v1.0.0",
  # ...
  {
    curl -s https://api.github.com/repos/$repo/releases/latest |
    awk '/"tag_name":/&&$0=$2' |
    sed 's/[",]//g'
  } || echo $DEFAULT_SEDNA_VERSION # fallback
}

: ${SEDNA_VERSION:=$(get_latest_version)}
: ${KB_VERSION:=v0.3.0}

_download_yamls() {

  yaml_dir=$1
  mkdir -p ${SEDNA_ROOT}/$yaml_dir
  cd ${SEDNA_ROOT}/$yaml_dir
  for yaml in ${yaml_files[@]}; do
    # the yaml file already exists, no need to download
    [ -e "$yaml" ] && continue

    echo downloading $yaml into ${SEDNA_ROOT}/$yaml_dir
    local try_times=30 i=1 timeout=2
    while ! timeout ${timeout}s curl -sSO https://raw.githubusercontent.com/kubeedge/sedna/main/$yaml_dir/$yaml; do
      ((++i>try_times)) && {
        echo timeout to download $yaml
        exit 2
      }
      echo -en "retrying to download $yaml after $[i*timeout] seconds...\r"
    done
  done
}

download_yamls() {
  yaml_files=(
  sedna.io_datasets.yaml
  sedna.io_federatedlearningjobs.yaml
  sedna.io_incrementallearningjobs.yaml
  sedna.io_jointinferenceservices.yaml
  sedna.io_lifelonglearningjobs.yaml
  sedna.io_models.yaml
  )
  _download_yamls build/crds
  yaml_files=(
    gm.yaml
  )
  _download_yamls build/gm/rbac
}

prepare_install(){
  # need to create a namespace
  kubectl create ns sedna

  kubectl label node/$GM_NODE_NAME sedna=control-plane --overwrite
}

prepare() {
  mkdir -p ${SEDNA_ROOT}
  
  # we only need build directory
  # here don't use git clone because of large vendor directory
  download_yamls
}

cleanup(){
  kubectl label node/$SEDNA_GM_NODE sedna- | sed 's/labeled$/un&/' || true
  kubectl delete ns sedna
}

create_crds() {
  cd ${SEDNA_ROOT}
  kubectl create -f build/crds
}

delete_crds() {
  cd ${SEDNA_ROOT}
  kubectl delete -f build/crds --timeout=90s
}

create_kb(){
  cd ${SEDNA_ROOT}

  kubectl $action -f - <<EOF
apiVersion: v1
kind: Service
metadata:
  name: kb
  namespace: sedna
spec:
  selector:
    sedna: kb
  type: NodePort
  ports:
    - protocol: TCP
      port: 9020
      targetPort: 9020
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: kb
  labels:
    sedna: kb
  namespace: sedna
spec:
  replicas: 1
  selector:
    matchLabels:
      sedna: kb
  template:
    metadata:
      labels:
        sedna: kb
    spec:
      nodeSelector:
        sedna: control-plane
      serviceAccountName: sedna
      containers:
      - name: kb
        imagePullPolicy: IfNotPresent
        image: kubeedge/sedna-kb:$KB_VERSION
        env:
          - name: KB_URL
            value: "sqlite:///db/kb.sqlite3"
        volumeMounts:
        - name: kb-url
          mountPath: /db
        resources:
          requests:
            memory: 256Mi
            cpu: 100m
          limits:
            memory: 512Mi
      volumes:
        - name: kb-url
          hostPath:
            path: /opt/kb-data
            type: DirectoryOrCreate
EOF
}

prepare_gm_config_map() {
  kb_node_port=$(kubectl -n sedna get svc kb -ojsonpath='{.spec.ports[0].nodePort}')

  # here try to get node ip by kubectl
  kb_node_ip=$(kubectl get node $KB_NODE_NAME -o jsonpath='{ .status.addresses[?(@.type=="ExternalIP")].address }')
  kb_node_internal_ip=$(kubectl get node $KB_NODE_NAME -o jsonpath='{ .status.addresses[?(@.type=="InternalIP")].address }')

  KB_ADDRESS=${kb_node_ip:-$kb_node_internal_ip}:$kb_node_port

  cm_name=${1:-gm-config}
  config_file=${TMP_DIR}/${2:-gm.yaml}

  if [ -n "${SEDNA_GM_CONFIG:-}" ] && [ -f "${SEDNA_GM_CONFIG}" ] ; then
    cp "$SEDNA_GM_CONFIG" $config_file
  else
    cat > $config_file << EOF
kubeConfig: ""
master: ""
namespace: ""
websocket:
  address: 0.0.0.0
  port: 9000
localController:
  server: http://localhost:${SEDNA_LC_BIND_PORT:-9100}
knowledgeBaseServer:
  server: http://$KB_ADDRESS
EOF
  fi

  kubectl $action -n sedna configmap $cm_name --from-file=$config_file
}

create_gm() {

  cd ${SEDNA_ROOT}

  kubectl create -f build/gm/rbac/

  cm_name=gm-config
  config_file_name=gm.yaml
  prepare_gm_config_map $cm_name $config_file_name


  kubectl $action -f - <<EOF
apiVersion: v1
kind: Service
metadata:
  name: gm
  namespace: sedna
spec:
  selector:
    sedna: gm
  type: NodePort
  ports:
    - protocol: TCP
      port: 9000
      targetPort: 9000
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: gm
  labels:
    sedna: gm
  namespace: sedna
spec:
  replicas: 1
  selector:
    matchLabels:
      sedna: gm
  template:
    metadata:
      labels:
        sedna: gm
    spec:
      nodeSelector:
        sedna: control-plane
      serviceAccountName: sedna
      containers:
      - name: gm
        image: kubeedge/sedna-gm:$SEDNA_VERSION
        command: ["sedna-gm", "--config", "/config/$config_file_name", "-v2"]
        volumeMounts:
        - name: gm-config
          mountPath: /config
        resources:
          requests:
            memory: 32Mi
            cpu: 100m
          limits:
            memory: 128Mi
      volumes:
        - name: gm-config
          configMap:
            name: $cm_name
EOF
}

delete_gm() {
  cd ${SEDNA_ROOT}

  kubectl delete -f build/gm/rbac/


  # no need to clean gm deployment alone
}

create_lc() {
  gm_node_port=$(kubectl -n sedna get svc gm -ojsonpath='{.spec.ports[0].nodePort}')

  # here try to get node ip by kubectl
  gm_node_ip=$(kubectl get node $GM_NODE_NAME -o jsonpath='{ .status.addresses[?(@.type=="ExternalIP")].address }')
  gm_node_internal_ip=$(kubectl get node $GM_NODE_NAME -o jsonpath='{ .status.addresses[?(@.type=="InternalIP")].address }')

  GM_ADDRESS=${gm_node_ip:-$gm_node_internal_ip}:$gm_node_port

  kubectl $action -f- <<EOF
apiVersion: apps/v1
kind: DaemonSet
metadata:
  labels:
    sedna: lc
  name: lc
  namespace: sedna
spec:
  selector:
    matchLabels:
      sedna: lc
  template:
    metadata:
      labels:
        sedna: lc
    spec:
      containers:
        - name: lc
          image: kubeedge/sedna-lc:$SEDNA_VERSION
          env:
            - name: GM_ADDRESS
              value: $GM_ADDRESS
            - name: BIND_PORT
              value: "${SEDNA_LC_BIND_PORT:-9100}"
            - name: NODENAME
              valueFrom:
                fieldRef:
                  fieldPath: spec.nodeName
            - name: ROOTFS_MOUNT_DIR
              # the value of ROOTFS_MOUNT_DIR is same with the mount path of volume
              value: /rootfs
          resources:
            requests:
              memory: 32Mi
              cpu: 100m
            limits:
              memory: 128Mi
          volumeMounts:
            - name: localcontroller
              mountPath: /rootfs
      volumes:
        - name: localcontroller
          hostPath:
            path: /
      restartPolicy: Always
      hostNetwork: true
EOF
}

delete_lc() {
  # ns would be deleted in delete_gm
  # so no need to clean lc alone
  return
}

wait_ok() {
  echo "Waiting control components to be ready..."
  kubectl -n sedna wait --for=condition=available --timeout=600s deployment/gm
  kubectl -n sedna wait pod --for=condition=Ready --selector=sedna
  kubectl -n sedna get pod
}

delete_pods() {
  # in case some nodes are not ready, here delete with a 60s timeout, otherwise force delete these
  kubectl -n sedna delete pod --all --timeout=60s || kubectl -n sedna delete pod --all --force --grace-period=0
}

check_kubectl () {
  kubectl get pod >/dev/null
}

check_action() {
  action=${SEDNA_ACTION:-create}
  support_action_list="create delete"
  if ! echo "$support_action_list" | grep -w -q "$action"; then
    echo "\`$action\` not in support action list: create/delete!" >&2
    echo "You need to specify it by setting $(red_text SEDNA_ACTION) environment variable when running this script!" >&2
    exit 2
  fi
  
}

check_node() {
  if [ -z "$GM_NODE_NAME" ] || ! kubectl get node $GM_NODE_NAME; then 
    echo "ERROR: $(red_text GM node name \`$GM_NODE_NAME\` does not exist in k8s cluster)!" >&2
    echo "You need to specify it by setting $(red_text SEDNA_GM_NODE) environment variable when running this script!" >&2
    exit 1
  fi
}

do_check() {
  check_kubectl
  check_action
  check_node
}

show_debug_infos() {
  cat - <<EOF
Sedna is $(green_text running):
See GM status: kubectl -n sedna get deploy
See LC status: kubectl -n sedna get ds lc
See Pod status: kubectl -n sedna get pod
EOF
}

NO_COLOR='\033[0m'
RED='\033[0;31m'
GREEN='\033[0;32m'
green_text() {
  echo -ne "$GREEN$@$NO_COLOR"
}

red_text() {
  echo -ne "$RED$@$NO_COLOR"
}

do_check

prepare
case "$action" in
  create)
    echo "Installing Sedna $SEDNA_VERSION..."
    prepare_install
    create_crds
    create_kb
    create_gm
    create_lc
    wait_ok
    show_debug_infos
    ;;

  delete)
    delete_pods
    delete_gm
    delete_lc
    delete_crds
    cleanup
    echo "$(green_text Sedna is uninstalled successfully)"
    ;;
esac
