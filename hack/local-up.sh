#!/bin/bash

# Developers can run `hack/local-up.sh` to setup up a local environment:
# 1. a local k8s cluster with a master node.
# 2. a kubeedge node.
# 3. our gm/lc.

# Based on the kubeedge-local-up script which builds a local k8s cluster and kubeedge,
# our local-up script installs our package locally for
# simply developing and preparing for e3e tests.

# It does:
# 1. build the gm/lc/worker images.
# 2. download kubeedge source code and run its localup script.
# 3. prepare our k8s env.
# 4. config gm config and start gm.
# 5. start lc.
# 6. add cleanup.

# For cleanup, it needs to do our cleanups before kubeedge cleanup.
# Otherwise lc cleanup (via kubectl delete) is stuck and lc is kept running.

set -o errexit
set -o nounset
set -o pipefail

SEDNA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"

cd "$SEDNA_ROOT"

NO_CLEANUP=${NO_CLEANUP:-false}

IMAGE_REPO=localhost/kubeedge/sedna
IMAGE_TAG=localup

# local k8s cluster name for local-up-kubeedge.sh
CLUSTER_NAME=sedna
MASTER_NODENAME=${CLUSTER_NAME}-control-plane
EDGE_NODENAME=edge-node
NAMESPACE=sedna

KUBEEDGE_VERSION=master
TMP_DIR="$(realpath local-up-tmp)"

GM_BIND_PORT=9000
LC_BIND_PORT=9100

arch() {
  local arch=$(uname -m)
  case "$arch" in
    x86_64) arch=amd64;;
    *);;
  esac
  echo "$arch"
}

download_and_extract_kubeedge() {

  [ -d kubeedge ] && return
  local version=${1:-$KUBEEDGE_VERSION}

  # master branch can't works with git clone --depth 1
  git clone -b $version https://github.com/kubeedge/kubeedge
  return

  # the archive file can't works since local-up-kubeedge.sh depends git tag
  # https://github.com/kubeedge/kubeedge/archive/${version}.tar.gz
}

get_kubeedge_pid() {
  ps -e -o pid,comm,args |
   grep -F "$TMP_DIR" |
   # match executable name and print the pid
   awk -v bin="${1:-edgecore}" 'NF=$2==bin'
}

localup_kubeedge() {
  pushd $TMP_DIR >/dev/null
  download_and_extract_kubeedge
  # without setsid when hits ctrl-c, edgecore/cloudclore will be terminated
  # before cleanup called.
  # but we need cloudcore/edgecore alive to clean our container(mainly lc),
  # so here new a session to run local-up-kubeedge.sh
  setsid  bash -c "
    cd kubeedge

    # no use ENABLE_DAEMON=true since it has not-fully-cleanup problem.
    TIMEOUT=90 CLUSTER_NAME=$CLUSTER_NAME ENABLE_DAEMON=false
    source hack/local-up-kubeedge.sh
   " &
  KUBEEDGE_ROOT_PID=$!
  add_cleanup '
    # for the case sometimes kube-proxy container in local machine
    # not cleanup.
    kubectl delete ds -n kube-system kube-proxy

    echo "found kubeedge pid, kill it: $KUBEEDGE_ROOT_PID"
    for((i=0;i<60;i++)); do
      ((i%15==0)) && kill "$KUBEEDGE_ROOT_PID"
      kill -0 "$KUBEEDGE_ROOT_PID" || break
      echo "waiting for $KUBEEDGE_ROOT_PID exists"
      sleep 1
    done
    # sometimes cloudcore/edgecore cant be stopped(one kill command
    # local-up-kubeedge.sh is not enough),
    # so to ensure this cleanup we clean it manully.
    for bin in cloudcore edgecore; do
      pid=$(get_kubeedge_pid $bin)
      if [ -n "$pid" ]; then
        echo "found $bin: $pid, kill it"
        kill $pid
        kill $pid
      fi
    done
  '

  # wait ${MASTER_NODENAME} container to be running
  while ! docker ps --filter=name=${MASTER_NODENAME} | grep -q ${MASTER_NODENAME}; do
    # errexit when kubeedge-local pid exited
    kill -0 "$KUBEEDGE_ROOT_PID"
    sleep 3
  done

  # wait edgecore
  while [ -z "$(get_kubeedge_pid edgecore)" ]; do
    # errexit when kubeedge-local pid exited
    kill -0 "$KUBEEDGE_ROOT_PID"
    sleep 3
  done

  local parent=$$
  {
    # healthcheck for kubeedge-local pid
    # if it died, we died.
    while true; do
      if ! kill -0 "$KUBEEDGE_ROOT_PID"; then
        kill -INT $parent
        break
      fi
      sleep 1
    done
  }&
  popd

}

build_component_image() {
  local bin
  for bin; do
    echo "building $bin image"
    make -C "${SEDNA_ROOT}" ${bin}image IMAGE_REPO=$IMAGE_REPO IMAGE_TAG=$IMAGE_TAG
    eval ${bin^^}_IMAGE="'${IMAGE_REPO}/${bin}:${IMAGE_TAG}'"
  done
  # no clean up for images
}

build_worker_base_images() {
  echo "building worker base images"
  # build tensorflow1.15 image
  WORKER_TF1_IMAGE=$IMAGE_REPO/worker-tensorflow:1.15
  docker build -f build/worker/base_images/tensorflow/tensorflow-1.15.Dockerfile -t $WORKER_TF1_IMAGE .

  WORKER_IMAGE_HUB="'tensorflow:1.15': $WORKER_TF1_IMAGE"
  # add more base images
}

load_images_to_master() {
  local image
  for image in $GM_IMAGE; do
    # just use the docker-image command of kind instead of ctr
    # docker save $image | docker exec -i $MASTER_NODENAME ctr --namespace k8s.io image import -
    kind load --name $CLUSTER_NAME docker-image $image
  done
}

prepare_k8s_env() {
  kind get kubeconfig --name $CLUSTER_NAME > $TMP_DIR/kubeconfig
  export KUBECONFIG=$(realpath $TMP_DIR/kubeconfig)
  # prepare our k8s environment
  # create these crds including dataset, model, joint-inference etc.
  kubectl apply -f build/crds/sedna/

  # gm, lc will be created in this namespace
  kubectl create namespace $NAMESPACE

  # create the cluster role for gm
  kubectl apply -f build/gm/rbac/

  add_cleanup "
    kubectl delete -f build/crds/sedna/
    kubectl delete namespace $NAMESPACE --timeout=5s
  "
  load_images_to_master
}

start_gm() {
  # config gm and start as pod

  pushd $TMP_DIR >/dev/null

  local gm_node_name=${MASTER_NODENAME}
  local gm_pod_name=gm-pod

  # prepare gm config
  cat > gmconfig <<EOF
kubeConfig: ""
namespace: ""
imageHub:
  $WORKER_IMAGE_HUB
websocket:
  port: $GM_BIND_PORT
localController:
  server: http://localhost:$LC_BIND_PORT
EOF

  add_cleanup "kubectl delete cm config -n $NAMESPACE"

  # create configmap for gm config
  kubectl create -n $NAMESPACE configmap config --from-file=gmconfig

  add_cleanup "
    kubectl delete deployment gm -n $NAMESPACE
    kubectl delete service gm -n $NAMESPACE
  "

  # start gm as pod with specified node name
  # TODO: create a k8s service, but kubeedge can't support this.
  kubectl create -f - <<EOF
apiVersion: v1
kind: Service
metadata:
  name: gm
  namespace: sedna
spec:
  selector:
    app: gm
  type: NodePort
  ports:
    - protocol: TCP
      port: $GM_BIND_PORT
      targetPort: $GM_BIND_PORT
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: gm
  labels:
    app: gm
  namespace: sedna
spec:
  replicas: 1
  selector:
    matchLabels:
      app: gm
  template:
    metadata:
      labels:
        app: gm
    spec:
      nodeName: $gm_node_name
      serviceAccountName: sedna
      containers:
      - name: gm
        image: $GM_IMAGE
        command: ["sedna-gm", "--config", "/config/gmconfig", "-v2"]
        resources:
          requests:
            memory: 32Mi
            cpu: 100m
          limits:
            memory: 128Mi
        volumeMounts:
        - name: config
          mountPath: /config
      volumes:
        - name: config
          configMap:
            name: config
EOF

  local gm_ip=$(kubectl get node $gm_node_name -o jsonpath='{ .status.addresses[?(@.type=="InternalIP")].address }')
  local gm_port=$(kubectl -n $NAMESPACE get svc gm -ojsonpath='{.spec.ports[0].nodePort}')
  
  GM_ADDRESS=$gm_ip:$gm_port

  add_debug_info "See GM status: kubectl get deploy -n $NAMESPACE gm"
  popd
}

start_lc() {
  local lc_ds_name=lc

  add_cleanup "
  # so here give a timeout in case edgecore is exited unexpectedly
  kubectl delete --timeout=5s ds lc -n sedna

  # if edgecore exited unexpectedly, we need to clean lc manually
  [ -z \"\$(get_kubeedge_pid edgecore)\" ] && {
    # TODO: find a better way to do this
    echo 'try to stop lc and its pause in edgenode manually'
    docker stop \$(
      docker ps |
      # find lc and its pause container id
      # kubeedge/k8s container name rule:
      #   pod: k8s_${lc_ds_name}_{pod_name}_${NAMESPACE}_{pod_uid}_
      #   pause: k8s_POD_{pod_name}_${NAMESPACE}_{pause_uid}_
      #   where pod_name is ${lc_ds_name}-[a-z0-9]{5}
      grep 'k8s_.*_${lc_ds_name}-[a-z0-9]*_${NAMESPACE}_' |
      awk NF=1
    ) 2>/dev/null
  }

  "

  # start lc as daemonset
  kubectl create -f- <<EOF
apiVersion: apps/v1
kind: DaemonSet
metadata:
  labels:
    k8s-app: sedna-lc
  name: $lc_ds_name
  namespace: $NAMESPACE
spec:
  selector:
    matchLabels:
      k8s-app: $lc_ds_name
  template:
    metadata:
      labels:
        k8s-app: $lc_ds_name
    spec:
      nodeSelector:
        # only schedule to edge node
        node-role.kubernetes.io/edge:  ""
      containers:
        - name: $lc_ds_name
          image: $LC_IMAGE
          env:
            - name: GM_ADDRESS
              value: $GM_ADDRESS
            - name: BIND_PORT
              value: "$LC_BIND_PORT"
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
      hostNetwork: true
EOF
  add_debug_info "See LC status: kubectl get ds -n $NAMESPACE $lc_ds_name"

}

declare -a CLEANUP_CMDS=()
add_cleanup() {
  CLEANUP_CMDS+=("$@")
}

cleanup() {
  if [[ "${NO_CLEANUP}" = true ]]; then
    echo "No clean up..."
    return
  fi

  set +o errexit

  echo "Cleaning up sedna..."

  local idx=${#CLEANUP_CMDS[@]} cmd
  # reverse call cleanup
  for((;--idx>=0;)); do
    cmd=${CLEANUP_CMDS[idx]}
    echo "calling $cmd:"
    eval "$cmd"
  done

  set -o errexit
}

check_healthy() {
  # TODO
  true
}

debug_infos=""
add_debug_info() {
  debug_infos+="$@
"
}

check_prerequisites() {
  # TODO
  true
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

trap cleanup EXIT

cleanup

mkdir -p "$TMP_DIR"
add_cleanup 'rm -rf "$TMP_DIR"'

build_component_image gm lc
build_worker_base_images

check_prerequisites

localup_kubeedge

prepare_k8s_env

start_gm
start_lc

echo "Local Sedna cluster is $(green_text running).
Currently local-up script only support foreground running.
Press $(red_text Ctrl-C) to shut it down!

You can use it with: kind export kubeconfig --name ${CLUSTER_NAME}

$debug_infos
"

while check_healthy; do sleep 5; done
