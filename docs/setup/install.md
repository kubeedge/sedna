This guide covers how to install Sedna on an existing Kubernetes environment.

For interested readers, Sedna also has two important components that would be mentioned below, i.e., [GM(GlobalManager)](/README.md#globalmanager) and [LC(LocalController)](/README.md#localcontroller) for workerload generation and maintenance.

If you don't have an existing Kubernetes, you can:
1) Install Kubernetes by following the [Kubernetes website](https://kubernetes.io/docs/setup/).
2) Or follow [quick start](quick-start.md) for other options.

### Prerequisites
- [Kubectl][kubectl] with right kubeconfig
- [Kubernetes][kubernetes] 1.16+ cluster running
- [KubeEdge][kubeedge] v1.5+ running


#### Deploy Sedna

Currently GM is deployed as a [`deployment`][deployment], and LC is deployed as a [`daemonset`][daemonset].

Because LC needs to connnect to GM, and the limit connection between cloud `Pod` and edge `Pod`, GM will be deployed to a cloud node where the edge node can access to.

More specifically, the GM node has a INTERNAL-IP or EXTERNAL-IP where edge node can access to directly.

For example, in a kind cluster `kubectl get node -o wide`:
```shell
NAME                  STATUS   ROLES                  AGE     VERSION                   INTERNAL-IP     EXTERNAL-IP   OS-IMAGE             KERNEL-VERSION       CONTAINER-RUNTIME
edge-node             Ready    agent,edge             3d21h   v1.19.3-kubeedge-v1.6.1   192.168.0.233   <none>        Ubuntu 18.04.5 LTS   4.15.0-128-generic   docker://20.10.2
sedna-control-plane   Ready    control-plane,master   3d21h   v1.20.2                   172.18.0.2      <none>        Ubuntu 20.10         4.15.0-128-generic   containerd://1.5.0-beta.3-24-g95513021e
```
In this example the node `sedna-control-plane` has a internal-ip 172.18.0.2, and `edge-node` can access it.

So we can set `SEDNA_GM_NODE=sedna-control-plane` in below instructions:

```shell
# set the right node where edge node can be access
# SEDNA_GM_NODE=sedna-control-plane
SEDNA_GM_NODE=CHANGE-ME-HERE

curl https://raw.githubusercontent.com/kubeedge/sedna/main/scripts/installation/install.sh | SEDNA_GM_NODE=$SEDNA_GM_NODE SEDNA_ACTION=create bash -

```

The way above will require the network to access github since it will download the sedna [crd yamls](/build/crds).
If you have unstable network to access github or existing sedna source, you can try the way:
```shell
# SEDNA_ROOT is the sedna git source directory or cached directory
export SEDNA_ROOT=/opt/sedna
curl https://raw.githubusercontent.com/kubeedge/sedna/main/scripts/installation/install.sh | SEDNA_GM_NODE=$SEDNA_GM_NODE SEDNA_ACTION=create bash -
```

#### Debug
1\. Check the GM status:
```shell
kubectl get deploy -n sedna gm
```

2\. Check the LC status:
```shell
kubectl get ds lc -n sedna
```

3\. Check the pod status:
```shell
kubectl get pod -n sedna
```

#### Uninstall Sedna
```shell
# set the right node where edge node can be access
# SEDNA_GM_NODE=sedna-control-plane
SEDNA_GM_NODE=CHANGE-ME-HERE

curl https://raw.githubusercontent.com/kubeedge/sedna/main/scripts/installation/install.sh | SEDNA_GM_NODE=$SEDNA_GM_NODE SEDNA_ACTION=delete bash -
```

[kubectl]:https://kubernetes.io/docs/tasks/tools/install-kubectl-linux/#install-kubectl-binary-with-curl-on-linux
[kubeedge]:https://github.com/kubeedge/kubeedge
[kubernetes]:https://kubernetes.io/
[deployment]: https://kubernetes.io/docs/concepts/workloads/controllers/deployment/
[daemonset]: https://kubernetes.io/docs/concepts/workloads/controllers/daemonset/
