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

# This script installs a all-in-one Sedna environment, including:
# - A Kubernetes v1.30 cluster with multi worker nodes, default none worker node.
# - KubeEdge with multi nodes, default is KubeEdge v1.21.0 and one edge node.
# - Sedna, default is v0.7.0.
#
# It requires you:
# - 2 CPUs or more
# - 2GB+ free memory, depends on node number setting
# - 10GB+ free disk space
# - Internet connection(docker hub, github etc.)
# - Linux platform, such as ubuntu/centos
# - Docker 17.06+
#
# Advanced options, influential env vars:
#
# NUM_CLOUD_WORKER_NODES| optional | The number of cloud worker nodes, default 0
# NUM_EDGE_NODES        | optional | The number of KubeEdge nodes, default 1
# KUBEEDGE_VERSION      | optional | The KubeEdge version to be installed, default v1.21.0,
#                                    the version keadm in the node image is built for.
#                                    'latest' gets the latest release.
# SEDNA_VERSION         | optional | The Sedna version to be installed, default v0.7.0.
#                                    'latest' gets the latest release.
# CLUSTER_NAME          | optional | The all-in-one cluster name, default 'sedna-mini'
# NO_INSTALL_SEDNA      | optional | If 'false', install Sedna, else no install, default false.
# FORCE_INSTALL_SEDNA   | optional | If 'true', force reinstall Sedna, default false.
# NODE_IMAGE            | optional | Custom node image
# REUSE_EDGE_CONTAINER  | optional | Whether reuse edge node containers or not, default is true

set -o errexit
set -o nounset
set -o pipefail


# tested together; the node image (Kubernetes version) bundles keadm of DEFAULT_KUBEEDGE_VERSION
DEFAULT_SEDNA_VERSION=v0.7.0
DEFAULT_KUBEEDGE_VERSION=v1.21.0
DEFAULT_NODE_IMAGE_VERSION=v1.30.0


function prepare_env() {
  : ${CLUSTER_NAME:=sedna-mini}

  # here not use := because it ignore the error of get_latest_version command
  if [ "${KUBEEDGE_VERSION:-}" = latest ]; then
    KUBEEDGE_VERSION=$(get_latest_version kubeedge/kubeedge $DEFAULT_KUBEEDGE_VERSION)
  fi
  : ${KUBEEDGE_VERSION:=$DEFAULT_KUBEEDGE_VERSION}
  # 1.21.0 => v1.21.0
  # v1.21.0 => v1.21.0
  KUBEEDGE_VERSION=v${KUBEEDGE_VERSION#v}

  if [ "${SEDNA_VERSION:-}" = latest ]; then
    SEDNA_VERSION=$(get_latest_version kubeedge/sedna $DEFAULT_SEDNA_VERSION)
  fi
  : ${SEDNA_VERSION:=$DEFAULT_SEDNA_VERSION}
  SEDNA_VERSION=v${SEDNA_VERSION#v}

  : ${NUM_CLOUD_WORKER_NODES:=0}
  : ${NUM_EDGE_NODES:=1}

  : ${ALLINONE_NODE_IMAGE:=kubeedge/sedna-allinone-node:$DEFAULT_NODE_IMAGE_VERSION}

  readonly MAX_CLOUD_WORKER_NODES=2
  readonly MAX_EDGE_WORKER_NODES=3

  # TODO: find a better way to figure this kind control plane
  readonly CONTROL_PLANE_NAME=${CLUSTER_NAME}-control-plane
  readonly CLOUD_WORKER_NODE_NAME=${CLUSTER_NAME}-worker
  # target this cluster explicitly, the current kubectl context may be another cluster
  readonly KUBE_CONTEXT=kind-${CLUSTER_NAME}

  # cloudcore default websocket port
  : ${CLOUDCORE_WS_PORT:=10000}
  # cloudcore default cert port
  : ${CLOUDCORE_CERT_PORT:=10002}

  # for debug purpose
  : ${RETAIN_CONTAINER:=}

  # use existing edge node containers
  # default is true
  : ${REUSE_EDGE_CONTAINER:=true}

  # whether install sedna control plane or not
  # false means install, other values mean no install
  : ${NO_INSTALL_SEDNA:=false}

  # force install sedna control plane
  # default is false
  : ${FORCE_INSTALL_SEDNA:=false}

  # The docker network for edge nodes to separate the network of control plane.
  # Since `kind` CNI doesn't support edge node, here just use the network 'kind'.
  # TODO(llhuii): find a way to use the default docker network 'bridge'.
  : ${EDGE_NODE_NETWORK:=kind}

  # the EdgeMesh helm chart: a URL, downloaded with retries, or a local .tgz file
  # for hosts with a slow or no connection to GitHub. Pinned to a release, so a change
  # on EdgeMesh's main branch cannot break all-in-one; v1.17.0 deploys edgemesh-agent v1.17.0
  : ${EDGEMESH_VERSION:=v1.17.0}
  : ${EDGEMESH_CHART:=https://raw.githubusercontent.com/kubeedge/edgemesh/$EDGEMESH_VERSION/build/helm/edgemesh.tgz}

  # how long to wait for all nodes to be Ready after the edge nodes join
  : ${NODE_READY_TIMEOUT:=300s}


  validate_env
}

function validate_env() {

  ((NUM_CLOUD_WORKER_NODES<=MAX_CLOUD_WORKER_NODES)) || {
    log_fault "Only support NUM_CLOUD_WORKER_NODES at most $MAX_CLOUD_WORKER_NODES"
  }

  ((NUM_EDGE_NODES<=MAX_EDGE_WORKER_NODES)) || {
    log_fault "Only support NUM_EDGE_NODES at most $MAX_EDGE_WORKER_NODES"
  }
}


function _log() {
  local level=$1
  shift
  timestamp=$(date +"[$level%m%d %H:%M:%S.%3N]")
  echo "$timestamp $@"
}

function log_fault() {
  _log E "$@" >&2
  exit 2
}

function log_error() {
  _log E "$@" >&2
}

function log_info() {
  _log I "$@"
}

function gen_kind_config() {
  cat <<EOF
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
name: $CLUSTER_NAME
nodes:
  - role: control-plane
    image: $ALLINONE_NODE_IMAGE
    # expose kubeedge cloudcore
    extraPortMappings:
    - containerPort: $CLOUDCORE_WS_PORT
    - containerPort: $CLOUDCORE_CERT_PORT
EOF

  for((i=0;i<NUM_CLOUD_WORKER_NODES;i++)); do
  cat <<EOF
  - role: worker
    image: $ALLINONE_NODE_IMAGE
EOF
  done
}

function patch_kindnet() {
  # `kindnet`, the CNI of kind, requires `InClusterConfig`, i.e. the KUBERNETES_SERVICE_HOST
  # and KUBERNETES_SERVICE_PORT environment variables, which edgecore does not inject, so it
  # crashes on edge nodes. Edge nodes get their CNI config from this script
  # (gen_cni_config), so keep kindnet off them.
  run_in_control_plane kubectl -n kube-system patch daemonset kindnet --type merge -p '
    {"spec":{"template":{"spec":{"affinity":{"nodeAffinity":{"requiredDuringSchedulingIgnoredDuringExecution":
      {"nodeSelectorTerms":[{"matchExpressions":[{"key":"node-role.kubernetes.io/edge","operator":"DoesNotExist"}]}]}}}}}}}'
}

function create_k8s_cluster() {
  if kind get clusters | grep -qx -F "$CLUSTER_NAME"; then
    log_info "The k8s cluster $CLUSTER_NAME already exists, and just use it!"
    log_info "If you want to recreate one, just run \`$0 clean\`."
    return
  fi

  local extra_options=(--wait 90s)
  [ -n "$RETAIN_CONTAINER" ] && extra_options+=(--retain)
  gen_kind_config | kind create cluster ${extra_options[@]} --config -

}

function clean_k8s_cluster() {
  kind delete cluster --name ${CLUSTER_NAME}
}

function run_in_control_plane() {
  docker exec -i $CONTROL_PLANE_NAME "$@"
}

function get_control_plane_ip() {
  # https://stackoverflow.com/a/20686101
  docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' $CONTROL_PLANE_NAME
}

function get_control_plane_exposed_port() {
  local container_port=$1
  docker inspect -f "{{(index (index .NetworkSettings.Ports \"${container_port}/tcp\") 0).HostPort}}" $CONTROL_PLANE_NAME
}

function setup_control_kubeconfig() {
  run_in_control_plane bash -euc "
    # copy kube config file
    mkdir -p ~/.kube
    cp /etc/kubernetes/admin.conf ~/.kube/config
  "
}

function setup_cloudcore() {
  # keadm already built into control plane

  CLOUDCORE_LOCAL_IP=$(get_control_plane_ip)

  # Use default docker network for edge nodes to separate the network of control plane which uses the defined network 'kind'
  CLOUDCORE_EXPOSED_IP=$(get_docker_network_gw $EDGE_NODE_NETWORK)

  CLOUDCORE_EXPOSED_WS_PORT=$(get_control_plane_exposed_port $CLOUDCORE_WS_PORT)
  CLOUDCORE_EXPOSED_CERT_PORT=$(get_control_plane_exposed_port $CLOUDCORE_CERT_PORT)
  CLOUDCORE_ADVERTISE_ADDRESSES=$CLOUDCORE_LOCAL_IP,$CLOUDCORE_EXPOSED_IP
  CLOUDCORE_EXPOSED_ADDR=$CLOUDCORE_EXPOSED_IP:$CLOUDCORE_EXPOSED_WS_PORT

  # keadm accepts version format: 1.21.0
  local version=${KUBEEDGE_VERSION/v}
  # keadm (since KubeEdge v1.11) deploys cloudcore with helm as a pod on the host network.
  # Pin it to the control plane, whose container exposes the cloudcore ports, and enable
  # the dynamicController, which the edge nodes' metaServer (needed by edgemesh) uses.
  run_in_control_plane bash -euc "
    kubectl -n kubeedge get deploy cloudcore >/dev/null 2>&1 ||
      keadm init --kubeedge-version=$version --advertise-address=$CLOUDCORE_ADVERTISE_ADDRESSES \
        --kube-config=/etc/kubernetes/admin.conf \
        --set cloudCore.nodeSelector.kubernetes\\\\.io/hostname=$CONTROL_PLANE_NAME \
        --set cloudCore.tolerations[0].operator=Exists \
        --set cloudCore.modules.dynamicController.enable=true
    "
  if ! run_in_control_plane kubectl -n kubeedge rollout status deploy/cloudcore --timeout=300s; then
    run_in_control_plane kubectl -n kubeedge logs deploy/cloudcore --tail=50 | sed "s/^/    /" || true
    log_fault "Timeout to wait cloudcore, above are the last logs of cloudcore."
  fi

  # wait token to be created
  local i
  for((i=1;i<=30;i++)); do
    KUBEEDGE_TOKEN=$(run_in_control_plane keadm gettoken --kube-config=/etc/kubernetes/admin.conf 2>/dev/null) && break
    sleep 1
  done
  [ -n "${KUBEEDGE_TOKEN:-}" ] || log_fault "Timeout to wait cloudcore to generate the token."
}

function install_edgemesh() {
  if ((NUM_EDGE_NODES<1)); then
    # no edge node, no edgemesh
    return
  fi

  local server_node_name
  if ((NUM_CLOUD_WORKER_NODES>0)); then
    server_node_name=${CLUSTER_NAME}-worker
    # master node also needs edgemesh agent for apiservice's communication
    kubectl --context "$KUBE_CONTEXT" taint nodes ${CLUSTER_NAME}-control-plane node-role.kubernetes.io/control-plane:NoSchedule- || true
  else
    server_node_name=${CLUSTER_NAME}-control-plane
  fi

  echo Installing edgemesh with server on $server_node_name

  # the dynamicController of cloudcore and the metaServer and DNS of edgecore that
  # edgemesh needs are enabled by keadm init and join

  # helm fetches a chart URL once, with a short timeout, so a slow connection to
  # GitHub fails the whole install; download it with retries first
  local chart=$EDGEMESH_CHART
  if [[ "$chart" == http://* || "$chart" == https://* ]]; then
    local chart_dir=$(mktemp -d)
    local i
    for i in 1 2 3 4 5; do
      curl -fsSL --connect-timeout 30 --max-time 180 -o "$chart_dir/edgemesh.tgz" "$chart" && break
      log_info "downloading the EdgeMesh chart failed (attempt $i/5), retrying"
      sleep 5
    done
    [ -s "$chart_dir/edgemesh.tgz" ] || log_fault "cannot download the EdgeMesh chart from $chart; set EDGEMESH_CHART to a local copy"
    chart=$chart_dir/edgemesh.tgz
  fi

  # no server.publicIP
  # since allinone is in flat network, we just use private ip for edgemesh server
  helm upgrade --install edgemesh --kube-context "$KUBE_CONTEXT" \
    --set server.nodeName=$server_node_name \
    "$chart"

  echo Install edgemesh successfully
}

function gen_cni_config() {
  cat <<EOF
{
  "cniVersion": "0.3.1",
  "name": "edgecni",
  "plugins": [
    {
      "type": "ptp",
      "ipMasq": false,
      "ipam": {
        "type": "host-local",
        "dataDir": "/run/cni-ipam-state",
        "routes": [
          {
            "dst": "0.0.0.0/0"
          }
        ],
        "ranges": [
          [
            {
              "subnet": "10.244.0.0/24"
            }
          ]
        ]
      },
      "mtu": 1500
    },
    {
      "type": "portmap",
      "capabilities": {
        "portMappings": true
      }
    }
  ]
}

EOF
}

function create_and_setup_edgenodes() {

  for((i=0;i<NUM_EDGE_NODES;i++)); do
    log_info "Installing $i-th edge node..."
    # prefix with the cluster name so that several all-in-one clusters can run on one host;
    # the default CLUSTER_NAME keeps the previous name sedna-mini-edge<N>
    local containername=${CLUSTER_NAME}-edge$i
    local hostname=edge$i
    local label=sedna.io=sedna-mini-edge
    # label filters match literally, unlike name filters which are regexes,
    # so a CLUSTER_NAME with dots cannot match another cluster's containers
    local cluster_label=sedna.io/cluster=${CLUSTER_NAME}

    # Many tricky arguments are from the kind code
    # https://github.com/kubernetes-sigs/kind/blob/4910c3e221a858e68e29f9494170a38e1c4e8b80/pkg/cluster/internal/providers/docker/provision.go#L148
    local run_cmds=(
      docker run
      --network "$EDGE_NODE_NETWORK"
      --hostname "$hostname"
      --name "$containername"
      --label $label
      --label $cluster_label
      --privileged
      --security-opt seccomp=unconfined
      --security-opt apparmor=unconfined
      --tmpfs /tmp
      --tmpfs /run
      --volume /var
      # some k8s things want to read /lib/modules
      --volume /lib/modules:/lib/modules:ro
      --restart=on-failure:1
      --tty
      --detach $ALLINONE_NODE_IMAGE
    )

    local existing_id=$(docker ps -qa --filter label=$label --filter label=$cluster_label --filter "name=^/${containername}$")
    if [ -z "$existing_id" ] && docker inspect "$containername" &>/dev/null; then
      # the name is taken by a container without this cluster's label
      if [ -n "$(legacy_edge_containers | grep -x "$containername")" ] &&
          [ "$CLUSTER_NAME" = sedna-mini ] &&
          kubectl --context "$KUBE_CONTEXT" get node "$hostname" &>/dev/null; then
        # created by an older version of this script for this cluster: its edge node
        # is registered here. docker cannot add a label to an existing container, so it
        # stays unlabelled; clean still removes it.
        existing_id=$containername
      else
        log_error "A container named $containername already exists but does not belong to cluster $CLUSTER_NAME."
        log_error "If an older version of this script created it, see 'Edge containers of older versions' in"
        log_error "  https://github.com/kubeedge/sedna/blob/main/docs/setup/all-in-one.md"
        log_fault "Otherwise remove or rename it, or set another CLUSTER_NAME, and rerun this script."
      fi
    fi
    if [ -n "$existing_id" ]; then
      if [ "${REUSE_EDGE_CONTAINER,,}" = true ] ; then
        log_info "Use existing container for ''$containername'"
        log_info "If not your attention, you can do:"
        log_info "  1) set REUSE_EDGE_CONTAINER=false"
        log_info "  Or 2) clean it first."
        log_info "And rerun this script."
        # start in case stopped
        docker start $containername
      else
        log_error "The container named $containername already exists, you can do:"
        log_error "  1) set REUSE_EDGE_CONTAINER=true"
        log_error "  Or 2) clean it first."
        log_fault "And rerun this script."
      fi
    else
      # does not exist, create one container for this edge
      "${run_cmds[@]}"
    fi

    # install edgecore using keadm join; the kind node image runs containerd with the
    # systemd cgroup driver. The edgecore settings:
    # - metaServer, clusterDNS and clusterDomain for edgemesh
    # - imageGCHighThresholdPercent 100 for no image gc
    local version=${KUBEEDGE_VERSION/v}
    docker exec -i $containername bash -uec "
      pgrep edgecore >/dev/null || {
        # containerd starts with the container's systemd, a few seconds after docker run
        for((i=0;i<60;i++)); do [ -S /run/containerd/containerd.sock ] && break; sleep 1; done
        # keadm join needs a clean /etc/kubeedge, e.g. after a failed join
        keadm reset edge --force >/dev/null 2>&1 || true

        keadm join \
          --cloudcore-ipport=${CLOUDCORE_EXPOSED_ADDR} \
          --certport=${CLOUDCORE_EXPOSED_CERT_PORT} \
          --token=$KUBEEDGE_TOKEN \
          --kubeedge-version '$version' \
          --edgenode-name '$hostname' \
          --remote-runtime-endpoint unix:///run/containerd/containerd.sock \
          --cgroupdriver systemd \
          --set modules.metaManager.metaServer.enable=true \
          --set 'modules.edged.tailoredKubeletConfig.clusterDNS={169.254.96.16}' \
          --set modules.edged.tailoredKubeletConfig.clusterDomain=cluster.local \
          --set modules.edged.tailoredKubeletConfig.imageGCHighThresholdPercent=100
     }

    "
    # fix cni config file
    gen_cni_config | docker exec -i $containername tee /etc/cni/net.d/10-edgecni.conflist >/dev/null

    {
      # wait edge node to be created at background
      nwait=20
      for((i=0;i<nwait;i++)); do
        kubectl --context "$KUBE_CONTEXT" get node $hostname &>/dev/null && break
        sleep 3
      done
    } &

  done
  # wait all edge nodes to be created
  wait

}

function legacy_edge_containers() {
  # edge containers of older versions of this script: named sedna-mini-edge<N> whatever
  # the CLUSTER_NAME, and without the sedna.io/cluster label
  docker ps -a --filter label=sedna.io=sedna-mini-edge \
    --format '{{.Names}} {{.Label "sedna.io/cluster"}}' |
    awk '$1 ~ /^sedna-mini-edge[0-9]+$/ && $2 == "" { print $1 }'
}

function clean_edgenodes() {
  # only this cluster's edge nodes, so cleaning one cluster keeps the others running
  local cids=$(docker ps -a --filter label=sedna.io=sedna-mini-edge --filter label=sedna.io/cluster=${CLUSTER_NAME} -q)
  # the default cluster also removes the edge containers of older versions of this script;
  # see 'Edge containers of older versions' in docs/setup/all-in-one.md for other clusters
  if [ "$CLUSTER_NAME" = sedna-mini ]; then
    cids+=" $(legacy_edge_containers)"
  fi
  for cid in $(printf '%s\n' $cids | sort -u); do
    # -v: also remove the container's anonymous volumes (its /var, with the images it
    # pulled), which `docker run --volume /var` creates and a plain `docker rm` leaves behind
    docker stop $cid; docker rm -v $cid
  done
}

function get_docker_network_gw() {
  docker network inspect ${1-bridge} --format='{{(index .IPAM.Config 0).Gateway}}'
}

function setup_cloud() {
  create_k8s_cluster

  patch_kindnet

  setup_control_kubeconfig

  setup_cloudcore
}

function clean_cloud() {
  clean_k8s_cluster
}

function setup_edge() {
  create_and_setup_edgenodes
}

function clean_edge() {
  clean_edgenodes
}

function install_sedna() {
  if [[ "$NO_INSTALL_SEDNA" != "false" ]]; then
    return
  fi

  if run_in_control_plane kubectl get ns sedna; then
    if [ "$FORCE_INSTALL_SEDNA" != true ]; then
      log_info '"sedna" namespace already exists, no install Sedna control components.'
      log_info 'If want to reinstall them, you can remove it by `kubectl delete ns sedna` or set FORCE_INSTALL_SEDNA=true!'
      log_info
      return
    fi
    run_in_control_plane bash -e -o pipefail -c "
    curl -fsSL https://raw.githubusercontent.com/kubeedge/sedna/main/scripts/installation/install.sh | SEDNA_ACTION=delete SEDNA_VERSION=$SEDNA_VERSION bash -
  "
  fi

  log_info "Installing Sedna Control Components..."

  run_in_control_plane bash -e -o pipefail -c "
    curl -fsSL https://raw.githubusercontent.com/kubeedge/sedna/main/scripts/installation/install.sh | SEDNA_ACTION=create SEDNA_VERSION=$SEDNA_VERSION bash -
  "
}

function get_latest_version() {
  # get the latest version of specified gh repo
  local repo=${1} default_version=${2:-}
  # output of this latest page:
  # ...
  # "tag_name": "v1.0.0",
  # ...

  # Sometimes this will reach rate limit
  # https://docs.github.com/rest/overview/resources-in-the-rest-api#rate-limiting
  local url=https://api.github.com/repos/$repo/releases/latest
  if ! curl --fail -s $url | awk '/"tag_name":/&&$0=$2' | sed 's/[",]//g'; then
    log_error "Error to get latest version of $repo: $(curl -s $url | head)"
    [ -n "$default_version" ] && {
      log_error "Fall back to default version: $default_version"
      echo $default_version
    }
  fi
}

function arch() {
  # the architecture names of the kind, kubectl and KubeEdge release assets
  local arch=$(uname -m)
  case "$arch" in
    x86_64) arch=amd64;;
    aarch64) arch=arm64;;
    *);;
  esac
  echo "$arch"
}

function _download_tool() {
  local name=$1 url=$2
  local file=/usr/local/bin/$name
	curl -Lo $file $url
  chmod +x $file
}

function check_command_exists() {
  type $1 >/dev/null 2>&1
}

function ensure_tool() {
  local command=$1 download_url=$2
  if check_command_exists $command; then
    return
  fi

	_download_tool $command $download_url

}

function ensure_kind() {
  # kind v0.23.0 is the first release for Kubernetes v1.30 node images
  local version=${KIND_VERSION:-0.23.0}
	ensure_tool kind https://kind.sigs.k8s.io/dl/v${version/v}/kind-linux-$(arch)
}

function ensure_kubectl() {

  local version=${KUBECTL_VERSION:-1.30.0}
  ensure_tool kubectl https://dl.k8s.io/release/v${version/v}/bin/linux/$(arch)/kubectl
}

function ensure_helm() {
  if check_command_exists helm; then
    return
  fi
  # the helm version installed when the host has none: Helm 3, pinned so that the
  # installer's latest release (now Helm 4) cannot change what this script runs
  local version=${HELM_VERSION:-v3.22.1}
  curl -fsSL https://raw.githubusercontent.com/helm/helm/$version/scripts/get-helm-3 |
    DESIRED_VERSION=$version bash
}

function ensure_tools() {
  ensure_kind
  ensure_kubectl
  ensure_helm
}

function main() {
  ensure_tools
  prepare_env
  action=${1-create}

  case "$action" in
    create)
      setup_cloud
      setup_edge
      # wait all nodes to be ready; kubectl waits only 30s by default, which a busy
      # host can exceed right after the edge nodes join
      kubectl --context "$KUBE_CONTEXT" wait --for=condition=ready node --all \
        --timeout="$NODE_READY_TIMEOUT"

      # edgemesh need to be installed before sedna
      install_edgemesh
      install_sedna
      log_info "Mini Sedna is created successfully"
      ;;

    delete|clean)
      clean_edge
      clean_cloud
      log_info "Mini Sedna is uninstalled successfully"
      ;;

    # As a source file, noop
    __source__)
      ;;

    *)
      log_fault "Unknown action $action"
      ;;
  esac
}

main "$@"
