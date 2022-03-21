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
# - A Kubernetes v1.21 cluster with multi worker nodes, default none worker node.
# - KubeEdge with multi nodes, default is latest KubeEdge and one edge node.
# - Sedna, default is latest release version.
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
# KUBEEDGE_VERSION      | optional | The KubeEdge version to be installed.
#                                    if not specified, it try to get latest version or v1.8.0
# SEDNA_VERSION         | optional | The Sedna version to be installed.
#                                    if not specified, it will get latest release or v0.4.1
# CLUSTER_NAME          | optional | The all-in-one cluster name, default 'sedna-mini'
# NO_INSTALL_SEDNA      | optional | If 'false', install Sedna, else no install, default false.
# FORCE_INSTALL_SEDNA   | optional | If 'true', force reinstall Sedna, default false.
# NODE_IMAGE            | optional | Custom node image
# REUSE_EDGE_CONTAINER  | optional | Whether reuse edge node containers or not, default is true

set -o errexit
set -o nounset
set -o pipefail


DEFAULT_SEDNA_VERSION=v0.4.1
DEFAULT_KUBEEDGE_VERSION=v1.8.0
DEFAULT_NODE_IMAGE_VERSION=v1.21.1


function prepare_env() {
  : ${CLUSTER_NAME:=sedna-mini}

  # here not use := because it ignore the error of get_latest_version command
  if [ -z "${KUBEEDGE_VERSION:-}" ]; then
    KUBEEDGE_VERSION=$(get_latest_version kubeedge/kubeedge $DEFAULT_KUBEEDGE_VERSION)
  fi
  # 1.8.0 => v1.8.0
  # v1.8.0 => v1.8.0
  KUBEEDGE_VERSION=v${KUBEEDGE_VERSION#v}

  if [ -z "${SEDNA_VERSION:-}" ]; then
    SEDNA_VERSION=$(get_latest_version kubeedge/sedna $DEFAULT_SEDNA_VERSION)
  fi
  SEDNA_VERSION=v${SEDNA_VERSION#v}

  : ${NUM_CLOUD_WORKER_NODES:=0}
  : ${NUM_EDGE_NODES:=1}

  : ${ALLINONE_NODE_IMAGE:=kubeedge/sedna-allinone-node:$DEFAULT_NODE_IMAGE_VERSION}

  readonly MAX_CLOUD_WORKER_NODES=2
  readonly MAX_EDGE_WORKER_NODES=3

  # TODO: find a better way to figure this kind control plane
  readonly CONTROL_PLANE_NAME=${CLUSTER_NAME}-control-plane
  readonly CLOUD_WORKER_NODE_NAME=${CLUSTER_NAME}-worker

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
  # Since in edge node, we just use containerd instead of docker, this requires CNI,
  # And `kindnet` is the CNI in kind, requires `InClusterConfig`
  # which would require KUBERNETES_SERVICE_HOST/KUBERNETES_SERVICE_PORT environment variables.
  # But edgecore(up to 1.8.0) does not inject these environments.
  # Here make a patch: can be any value
  run_in_control_plane kubectl set env -n kube-system daemonset/kindnet KUBERNETES_SERVICE_HOST=10.96.0.1 KUBERNETES_SERVICE_PORT=443
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

  # keadm accepts version format: 1.8.0
  local version=${KUBEEDGE_VERSION/v}
  run_in_control_plane bash -euc "
    # install cloudcore
    pgrep cloudcore >/dev/null || {
      # keadm 1.8.1 is incompatible with 1.9.1 since crds' upgrade
      rm -rf /etc/kubeedge/crds

      keadm init --kubeedge-version=$version --advertise-address=$CLOUDCORE_ADVERTISE_ADDRESSES"'
    }

    # wait token to be created
    exit_code=1
    TIMEOUT=30 # in seconds
    for((i=1;i<=TIMEOUT; i++)); do
      keadm gettoken >/dev/null 2>&1 && exit_code=0 && break
      echo -ne "Waiting cloudcore to generate token, $i seconds...\r"
      sleep 1
    done
    echo
    if [ $exit_code -gt 0 ]; then
      log_lines=50
      tail -$log_lines /var/log/kubeedge/cloudcore.log | sed "s/^/    /"
      echo "Timeout to wait cloudcore, above are the last $log_lines log of cloudcore."
    fi
    exit $exit_code
  '
  KUBEEDGE_TOKEN=$(run_in_control_plane keadm gettoken)
}

_change_detect_yaml_change() {
  # execute the specified yq commands on stdin
  # if same, output nothing
  # else output the updated yaml
  local yq_cmds="${1:-.}"
  docker run -i --rm --entrypoint sh mikefarah/yq -c "
   yq e . - > a
   yq e '$yq_cmds' a > b
   cmp -s a b || cat b
   "
}

reconfigure_edgecore() {
  # update edgecore.yaml for every edge node
  local script_name=reconfigure-edgecore

  if ((NUM_EDGE_NODES<1)); then
    return
  fi

  local yq_cmds="$1"

  # I want to leverage kubectl but k8s has no ways to run job on each node once
  # see https://github.com/kubernetes/kubernetes/issues/64623 for more detais
  # So I use Daemonset temporarily

  kubectl apply -f - <<EOF
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: $script_name
  namespace: kubeedge
spec:
  selector:
    matchLabels:
      edgecore: script
  template:
    metadata:
      labels:
        edgecore: script
    spec:
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
              - matchExpressions:
                  - key: node-role.kubernetes.io/edge
                    operator: Exists
      hostPID: true
      volumes:
      - name: config
        hostPath:
          path: /etc/kubeedge/config
      containers:
      - name: $script_name
        env:
        - name: NODE_NAME
          valueFrom:
            fieldRef:
              fieldPath: spec.nodeName

        securityContext:
          runAsUser: 0
        volumeMounts:
        - name: config
          mountPath: /config
        image: mikefarah/yq
        command:
        - sh
        - -c
        - |
         # inject random cmd to reapply when reconfigure
         : $$
         yq e . /config/edgecore.yaml > a && yq e '$(echo $yq_cmds)' a > b || exit 1
         cmp -s a b && echo No need to reconfigure \$NODE_NAME edgecore || {
           # backup and overwrite config, kill edgecore and wait systemd restart it
           cp /config/edgecore.yaml /config/edgecore.yaml.reconfigure_bk
           cp b /config/edgecore.yaml

           pkill edgecore

           # check to edgecore process status
           > pids
           for i in 0 1 2 3; do
             sleep 10
             { pidof edgecore || echo =\$i; } >> pids
           done
           [ \$(sort -u pids | wc -l) -le 2 ] && echo Reconfigure \$NODE_NAME edgecore successfully || {
             echo Failed to reconfigure \$NODE_NAME edgecore >&2
             echo And recovery edgecore config yaml >&2
             cp a /config/edgecore.yaml

             # prevent daemonset execute this script too frequently
             sleep 1800
             exit 1
           }
         }
         sleep inf
EOF

  # wait this script been executed
  kubectl -n kubeedge rollout status --timeout=5m ds $script_name
  # wait all edge nodes to be ready if restarted
  kubectl wait --for=condition=ready node -l node-role.kubernetes.io/edge=

  # keep this daemonset script for debugging
  # kubectl -n kubeedge delete ds $script_name

}

reconfigure_cloudcore() {

  local config_file=/etc/kubeedge/config/cloudcore.yaml
  local yq_cmds=$1

  run_in_control_plane cat $config_file |
    _change_detect_yaml_change "$yq_cmds" |
  run_in_control_plane bash -euc "
   cat > cc.yaml
   ! grep -q . cc.yaml || {
     echo reconfigure and restart cloudcore
     cp $config_file ${config_file}.reconfigure_bk
     cp cc.yaml $config_file
     pkill cloudcore || true
     # TODO: use a systemd service
     (cloudcore &>> /var/log/kubeedge/cloudcore.log &)
   }

  "
  echo Reconfigure cloudcore successfully
}

function install_edgemesh() {
  if ((NUM_EDGE_NODES<1)); then
    # no edge node, no edgemesh
    return
  fi

  local server_node_name
  if ((NUM_CLOUD_WORKER_NODES>0)); then
    server_node_name=${CLUSTER_NAME}-worker
  else
    server_node_name=${CLUSTER_NAME}-control-plane
  fi

  echo Installing edgemesh with server on $server_node_name
  # enable Local APIServer
  reconfigure_cloudcore '.modules.dynamicController.enable=true'

  reconfigure_edgecore '
    .modules.edged.clusterDNS="169.254.96.16"
    | .modules.edged.clusterDomain="cluster.local"
    | .modules.metaManager.metaServer.enable=true
  '

  # no server.publicIP
  # since allinone is in flat network, we just use private ip for edgemesh server
 helm upgrade --install edgemesh \
    --set server.nodeName=$server_node_name \
    https://raw.githubusercontent.com/kubeedge/edgemesh/main/build/helm/edgemesh.tgz

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
    local containername=sedna-mini-edge$i
    local hostname=edge$i
    local label=sedna.io=sedna-mini-edge

    # Many tricky arguments are from the kind code
    # https://github.com/kubernetes-sigs/kind/blob/4910c3e221a858e68e29f9494170a38e1c4e8b80/pkg/cluster/internal/providers/docker/provision.go#L148
    local run_cmds=(
      docker run
      --network "$EDGE_NODE_NETWORK"
      --hostname "$hostname"
      --name "$containername"
      --label $label
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

    local existing_id=$(docker ps -qa --filter name=$containername --filter label=$label)
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

    # install edgecore using keadm join
    local version=${KUBEEDGE_VERSION/v}
    docker exec -i $containername bash -uec "
      pgrep edgecore >/dev/null || {
        keadm join \
          --cloudcore-ipport=${CLOUDCORE_EXPOSED_ADDR} \
          --certport=${CLOUDCORE_EXPOSED_CERT_PORT} \
          --token=$KUBEEDGE_TOKEN \
          --kubeedge-version '$version' \
          --edgenode-name '$hostname' \
          --remote-runtime-endpoint unix:///var/run/containerd/containerd.sock \
          --runtimetype remote

        # set imageGCHighThreshold to 100% for no image gc
        sed -i 's/imageGCHighThreshold:.*/imageGCHighThreshold: 100/' /etc/kubeedge/config/edgecore.yaml &&
          systemctl restart edgecore ||
          true  # ignore the error
     }

    "
    # fix cni config file
    gen_cni_config | docker exec -i $containername tee /etc/cni/net.d/10-edgecni.conflist >/dev/null

    {
      # wait edge node to be created at background
      nwait=20
      for((i=0;i<nwait;i++)); do
        kubectl get node $hostname &>/dev/null && break
        sleep 3
      done
    } &

  done
  # wait all edge nodes to be created
  wait

}

function clean_edgenodes() {
  for cid in $(docker ps -a --filter label=sedna.io=sedna-mini-edge -q); do
    docker stop $cid; docker rm $cid
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
    run_in_control_plane bash -ec "
    curl https://raw.githubusercontent.com/kubeedge/sedna/main/scripts/installation/install.sh | SEDNA_ACTION=clean SEDNA_VERSION=$SEDNA_VERSION bash -
  "
  fi

  log_info "Installing Sedna Control Components..."

  run_in_control_plane bash -ec "
    curl https://raw.githubusercontent.com/kubeedge/sedna/main/scripts/installation/install.sh | SEDNA_ACTION=create SEDNA_VERSION=$SEDNA_VERSION bash -
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
  local arch=$(uname -m)
  case "$arch" in
    x86_64) arch=amd64;;
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
  local version=${KIND_VERSION:-0.11.1}
	ensure_tool kind https://kind.sigs.k8s.io/dl/v${version/v}/kind-linux-$(arch)
}

function ensure_kubectl() {

  local version=${KUBECTL_VERSION:-1.21.0}
  ensure_tool kubectl https://dl.k8s.io/release/v${version/v}/bin/linux/$(arch)/kubectl
}

function ensure_helm() {
  if check_command_exists helm; then
    return
  fi
  curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
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
      # wait all nodes to be ready
      kubectl wait --for=condition=ready node --all

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
