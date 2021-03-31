* [Prerequisites](#prerequisites)
* [Download project source](#download-source)
* [Create CRDs](#create-crds)
* [Deploy GM](#deploy-gm)
 * [Prepare GM config](#prepare-gm-config)
 * [Run GM as k8s deployment](#run-gm-as-a-k8s-deployment)
* [Deploy LC](#deploy-lc)

## Deploy Sedna

### Prerequisites

- [GIT][git_tool]
- [GO][go_tool] version v1.15+.
- [Kubernetes][kubernetes] 1.16+.
- [KubeEdge][kubeedge] version v.15+.

GM will be deployed to a node which has satisfied these requirements:
1. Has a IP address which the edge can access to.

Simply you can use the node which `cloudcore` of `kubeedge` is deployed at.

The shell commands below should to be executed in this node and **one terminal session** in case keeping the shell variables.

### Download source
```shell
git clone http://github.com/kubeedge/sedna.git
cd sedna
git checkout main
```

### Create CRDs

```shell
# create these crds including dataset, model, joint-inference etc.
kubectl create -f build/crds/
```

### Deploy GM

#### Prepare GM config
The content of `build/gm/gm-config.yaml`:
```yaml
kubeConfig: ""
master: ""
namespace: ""
localController:
  server: http://localhost:9100
```
1. `kubeConfig`: config to connect k8s, default `""`
1. `master`: k8s master addr, default `""`
1. `namespace`: the namespace GM watches, `""` means that gm watches all namespaces, default `""`.
1. `localController`:
   - `server`: to be injected into the worker to connect LC.

Edit the config file if you wish.

Note: if you just want to use the default values, don't need to run the below commands.
```shell
# edit build/gm/gm-config.yaml, here using sed command.
# alternative you can edit the config file manully.
GM_CONFIG_FILE=build/gm/gm-config.yaml

# here edit it with another LC bind ports if you wish or it's conflict with your node environment since LC is deployed in host namespace.
LC_BIND_PORT=9100

LC_SERVER="http://localhost:$LC_BIND_PORT"

# setting lc server
sed -i "s@http://localhost:9100@$LC_SERVER@" $GM_CONFIG_FILE
```

#### Run GM as a K8S Deployment:

We don't need to config the kubeconfig in this method said by [accessing the API from a Pod](https://kubernetes.io/docs/tasks/access-application-cluster/access-cluster/#accessing-the-api-from-a-pod).

1\. Create the cluster roles in order to GM can access/write the CRDs:
```shell
# create the cluster role
kubectl create -f build/gm/rbac/
```

2\. Deploy GM as deployment:

Currently we need to deploy GM to a k8s node which edge node can access to.

More specifically, the k8s node has a INTERNAL-IP or EXTERNAL-IP where edge node can access to.

For example, in a kind cluster `kubectl get node -o wide`:
```shell
NAME                  STATUS   ROLES                  AGE     VERSION                   INTERNAL-IP     EXTERNAL-IP   OS-IMAGE             KERNEL-VERSION       CONTAINER-RUNTIME
edge-node             Ready    agent,edge             3d21h   v1.19.3-kubeedge-v1.6.1   192.168.0.233   <none>        Ubuntu 18.04.5 LTS   4.15.0-128-generic   docker://20.10.2
sedna-control-plane   Ready    control-plane,master   3d21h   v1.20.2                   172.18.0.2      <none>        Ubuntu 20.10         4.15.0-128-generic   containerd://1.5.0-beta.3-24-g95513021e
```
In this example the node `sedna-control-plane` has a internal-ip 172.18.0.2, and `edge-node` can access it.

So we can set `GM_NODE_NAME=sedna-control-plane` in below instructions:

```shell
# set the right node where edge node can be access
# GM_NODE_NAME=sedna-control-plane
GM_NODE_NAME=CHANGE-ME-HERE

# create configmap from $GM_CONFIG_FILE
GM_CONFIG_FILE=${GM_CONFIG_FILE:-build/gm/gm-config.yaml}

GM_CONFIG_FILE_NAME=$(basename $GM_CONFIG_FILE)
kubectl create -n sedna configmap gm-config --from-file=$GM_CONFIG_FILE


kubectl apply -f - <<EOF
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
      port: 9000
      targetPort: 9000
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
      nodeName: $GM_NODE_NAME
      serviceAccountName: sedna
      containers:
      - name: gm
        image: kubeedge/sedna-gm:v0.1.0
        command: ["sedna-gm", "--config", "/config/$GM_CONFIG_FILE_NAME", "-v2"]
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
            name: gm-config
EOF
```

4\. Check the GM status:
```shell
kubectl get deploy -n sedna gm
```

### Deploy LC
Prerequisites:
1. Run GM successfully.

1\. Deploy LC as k8s daemonset:
```shell
gm_node_port=$(kubectl -n sedna get svc gm -ojsonpath='{.spec.ports[0].nodePort}')

# fill the GM_NODE_NAME's ip which edge node can access to.
# such as gm_node_ip=192.168.0.9
# gm_node_ip=<GM_NODE_NAME_IP_ADDRESS>

# Here is the automatical way: try to get node ip by kubectl
gm_node_ip=$(kubectl get node $GM_NODE_NAME -o jsonpath='{ .status.addresses[?(@.type=="ExternalIP")].address }')
gm_node_internal_ip=$(kubectl get node $GM_NODE_NAME -o jsonpath='{ .status.addresses[?(@.type=="InternalIP")].address }')

GM_ADDRESS=${gm_node_ip:-$gm_node_internal_ip}:$gm_node_port

kubectl create -f- <<EOF
apiVersion: apps/v1
kind: DaemonSet
metadata:
  labels:
    k8s-app: sedna-lc
  name: lc
  namespace: sedna
spec:
  selector:
    matchLabels:
      k8s-app: lc
  template:
    metadata:
      labels:
        k8s-app: lc
    spec:
      containers:
        - name: lc
          image: kubeedge/sedna-lc:v0.1.0
          env:
            - name: GM_ADDRESS
              value: $GM_ADDRESS
            - name: BIND_PORT
              value: "${LC_BIND_PORT:-9100}"
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
```

2\. Check the LC status:
```shell
kubectl get ds lc -n sedna

kubectl get pod -n sedna
```

[git_tool]:https://git-scm.com/downloads
[go_tool]:https://golang.org/dl/
[kubeedge]:https://github.com/kubeedge/kubeedge
[kubernetes]:https://kubernetes.io/
