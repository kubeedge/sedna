### Deploy local Sedna clusters

Once you have docker running, you can create a local Sedna cluster with:
```
bash scripts/local-up.sh
```
This script uses [kind](https://kind.sigs.k8s.io/docs/user/quick-start/) to create a 
local k8s cluster with one master node, and boots one edge node by running KubeEdge.
You can see them by using `kubectl get nodes -o wide`:
```shell
NAME                  STATUS   ROLES                  AGE     VERSION                   INTERNAL-IP     EXTERNAL-IP   OS-IMAGE             KERNEL-VERSION       CONTAINER-RUNTIME
edge-node             Ready    agent,edge             3d21h   v1.19.3-kubeedge-v1.6.1   192.168.0.233   <none>        Ubuntu 18.04.5 LTS   4.15.0-128-generic   docker://20.10.2
sedna-control-plane   Ready    control-plane,master   3d21h   v1.20.2                   172.18.0.2      <none>        Ubuntu 20.10         4.15.0-128-generic   containerd://1.5.0-beta.3-24-g95513021e
```

You can access master node with:
```
docker exec -it --detach-keys=ctrl-@ sedna-control-plane bash
alias docker=crictl
```

Docker images can be loaded into the cluster nodes with:
```
kind load docker-image my-custom-image --name sedna
```

