### Deploy All In One Sedna
The [all-in-one script](/scripts/installation/all-in-one.sh) is used to install Sedna along with a mini Kubernetes environment locally, including:
  - A Kubernetes v1.30 cluster with multi worker nodes, default zero worker node.
  - KubeEdge with multi edge nodes, default is KubeEdge v1.21.0 and one edge node.
  - Sedna, default is v0.7.0.

These versions are tested together. The node image `kubeedge/sedna-allinone-node:v1.30.0` is built from `kindest/node:v1.30.0` with `keadm` v1.21.0 by [build-allinone-image.sh](/scripts/installation/build-allinone-image.sh). If the image is not available from Docker Hub, build it locally first (about 2 minutes):

  ```bash
  bash scripts/installation/build-allinone-image.sh
  ```

It requires you:
  - 2 CPUs or more
  - 2GB+ free memory, depends on node number setting
  - 10GB+ free disk space
  - Internet connection(docker hub, github etc.)
  - Linux platform, such as ubuntu/centos
  - Docker 17.06+

For example: 

  ```bash
  curl https://raw.githubusercontent.com/kubeedge/sedna/main/scripts/installation/all-in-one.sh | NUM_EDGE_NODES=2 bash -
  ```

Above command installs a mini Sedna environment, including:
  - A Kubernetes v1.30 cluster with only one master node.
  - KubeEdge v1.21.0 with two edge nodes.
  - Sedna v0.7.0.

You can play it online on [katacoda](https://www.katacoda.com/kubeedge-sedna/scenarios/all-in-one).

Advanced options:
| Env Variable |  Description| Default Value|
| --- |  --- | --- |
|NUM_EDGE_NODES     | Number of KubeEdge nodes| 1 |
|NUM_CLOUD_WORKER_NODES    | Number of cloud **worker** nodes, not master node| 0|
|SEDNA_VERSION    | The Sedna version to be installed; `latest` for the latest release |v0.7.0|
|KUBEEDGE_VERSION    | The KubeEdge version to be installed; `latest` for the latest release. Other versions need a node image with a matching `keadm` |v1.21.0|
|CLUSTER_NAME       | The all-in-one cluster name| sedna-mini|
|FORCE_INSTALL_SEDNA       | If 'true', force to reinstall Sedna|false|
|NODE_IMAGE       | Custom node image| kubeedge/sedna-allinone-node:v1.30.0|
|REUSE_EDGE_CONTAINER      | Whether reuse edge node containers or not|true|

Clean all-in-one Sedna:  
  ```bash
  curl https://raw.githubusercontent.com/kubeedge/sedna/main/scripts/installation/all-in-one.sh | bash /dev/stdin clean
  ```

A cluster created with the former node image (`v1.21.1`, KubeEdge up to v1.14) cannot be reused: clean it before creating one with the current defaults.

#### Edge containers of older versions

Edge containers are labelled with their cluster (`sedna.io/cluster=<CLUSTER_NAME>`) and named `<CLUSTER_NAME>-edge<N>`, so creating or cleaning one cluster leaves the others alone. Older versions of the script named them `sedna-mini-edge<N>` for every `CLUSTER_NAME`, without that label.

- For the default cluster (`CLUSTER_NAME` unset), nothing is needed: the script reuses an older edge container whose node is registered in the cluster, and `clean` removes it.
- For a custom `CLUSTER_NAME`, clean the cluster with the version of the script that created it. If its edge containers are left behind, list the unlabelled ones:

  ```bash
  docker ps -a --filter label=sedna.io=sedna-mini-edge \
    --format '{{.Names}} {{.Label "sedna.io/cluster"}}' | awk '$2 == "" { print $1 }'
  ```

  Check that each listed container belongs to the cluster you are removing (`docker inspect <name>`), then remove it with `docker rm -f <name>`.