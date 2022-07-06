### Deploy Local Sedna Cluster

The [local-up script](/hack/local-up.sh) boots a local Kubernetes cluster, installs latest KubeEdge, and deploys Sedna based on the Sedna local repository.

#### Use Case
When one is contributing new features for Sedna, codes like AI algorithms under testing can be frequently changed before final deployment.
When coding in that case, s/he would suffer from tortured re-installations and frequent failures of the whole complicated system.
To get rid of the torments, one can use the local-up installation, embraced the single-machine simulation for agiler development and testing.

#### Setup

It requires:
  - 2 CPUs or more
  - 1GB+ free memory
  - 5GB+ free disk space
  - Internet connection(docker hub, github etc.)
  - Linux platform, such as ubuntu/centos
  - Docker 17.06+
  - A local Sedna code repository


Then you can enter Sedna local code repository, and create a local Sedna cluster with:
```
bash hack/local-up.sh
```

In more details, this local-up script uses [kind](https://kind.sigs.k8s.io/docs/user/quick-start/) to create a 
local K8S cluster with one master node, and joins the K8S cluster by running KubeEdge.

In another terminal, you can see them by using `kubectl get nodes -o wide`:
```shell
NAME                  STATUS   ROLES                  AGE     VERSION                   INTERNAL-IP     EXTERNAL-IP   OS-IMAGE             KERNEL-VERSION       CONTAINER-RUNTIME
edge-node             Ready    agent,edge             3d21h   v1.19.3-kubeedge-v1.6.1   192.168.0.233   <none>        Ubuntu 18.04.5 LTS   4.15.0-128-generic   docker://20.10.2
sedna-control-plane   Ready    control-plane,master   3d21h   v1.20.2                   172.18.0.2      <none>        Ubuntu 20.10         4.15.0-128-generic   containerd://1.5.0-beta.3-24-g95513021e
```

You can login the master node with:
```
docker exec -it --detach-keys=ctrl-@ sedna-control-plane bash
# since the master node just uses containerd CRI runtime, you can alias the CRI cli 'crictl' as 'docker'
alias docker=crictl
```

After you have done developing, built worker image and want to run your worker into master node, your worker image should be loaded into the cluster nodes with:
```
kind load docker-image --name sedna <your-custom-worker-image>
```

