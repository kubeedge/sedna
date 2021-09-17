### Deploy All In One Sedna
The [all-in-one script](/scripts/installation/all-in-one.sh) is used to install Sedna along with a mini Kubernetes environment locally, including:
  - A Kubernetes v1.21 cluster with multi worker nodes, default zero worker node.
  - KubeEdge with multi nodes, default is latest KubeEdge and one edge node.
  - Sedna, latest release version.

It requires you:
  - 2 CPUs or more
  - 2GB+ free memory, depends on node number setting
  - 10GB+ free disk space
  - Internet connection(docker hub, github etc.)
  - Linux platform, such as ubuntu/centos
  - Docker 17.06+

For example: 

  ```bash
  curl https://raw.githubusercontent.com/kubeedge/sedna/master/scripts/installation/all-in-one.sh | NUM_EDGE_WORKERS=2 bash -
  ```

Above command installs a mini Sedna environment, including:
  - A Kubernetes v1.21 cluster with multi worker nodes, default none worker node.
  - KubeEdge with multi nodes, default is latest KubeEdge and one edge node.
  - Sedna, latest release version.

You can play it online on [katacoda](https://www.katacoda.com/kubeedge-sedna/scenarios/all-in-one).

Advanced options:
| Env Variable |  Description| Default Value|
| --- |  --- | --- |
|NUM_CLOUD_WORKERS    | The cloud workers| 0|
|NUM_EDGE_WORKERS     | The KubeEdge workers| 1|
|KUBEEDGE_VERSION    | The KubeEdge version to be installed. |The latest KubeEdge release version|
|CLUSTER_NAME       | The all-in-one cluster name| sedna-mini|
|FORCE_INSTALL_SEDNA       | If 'true', force to reinstall Sedna|false|
|NODE_IMAGE       | Custom node image| kubeedge/sedna-allinone-node:v1.21.1|
|REUSE_EDGE_CONTAINER      | Whether reuse edge node containers or not|true|
