* [Prerequisites](#prerequisites)
* [Download project source](#download-source)
* [Create CRDs](#create-crds)
* [Deploy GM](#deploy-gm)
 * [Prepare GM config](#prepare-gm-config)
 * [Run GM as k8s deployment](#run-gm-as-k8s-deployment)
* [Deploy LC](#deploy-lc)

## Deploy Sedna

### Prerequisites

- [GIT][git_tool]
- [GO][go_tool] version v1.15+.
- [Kubernetes][kubernetes] 1.16+.
- [KubeEdge][kubeedge] version v.15+.

GM will be deployed to a node which has satisfied these requirements:
 1. Has a public IP address which the edge can access to.
 1. Can access the k8s master.

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
# create these crds including dataset, model, joint-inference
kubectl apply -f build/crds/
```

### Deploy GM

#### Prepare GM config
Get `build/gm/gm-config.yaml` for a copy
```yaml
kubeConfig: ""
master: ""
namespace: ""
websocket:
  address: 0.0.0.0
  port: 9000
localController:
  server: http://localhost:9100
```
1. `kubeConfig`: config to connect k8s, default `""`
1. `master`: k8s master addr, default `""`
1. `namespace`: the namespace GM watches, `""` means that gm watches all namespaces, default `""`.
1. `websocket`: since the current limit of kubeedge(1.5), GM needs to build the websocket channel for communicating between GM and LCs.
1. `localController`:
   - `server`: to be injected into the worker to connect LC.

#### Run GM as k8s deployment:

We don't need to config the kubeconfig in this method said by [accessing the API from a Pod](https://kubernetes.io/docs/tasks/access-application-cluster/access-cluster/#accessing-the-api-from-a-pod).

1\. Create the cluster role in case that gm can access/write the CRDs:
```shell
# create the cluster role
kubectl create -f build/gm/rbac/
```

2\. Prepare the config:
```shell
# edit it with another number if you wish
GM_PORT=9000
LC_PORT=9100

# here using github container registry for example
# edit it with the truly container registry by your choice.
IMAGE_REPO=kubeedge
IMAGE_TAG=v0.1.0

LC_SERVER="http://localhost:$LC_PORT"

```

```shell
# copy and edit CONFIG_FILE.
CONFIG_FILE=gm-config.yaml
cp build/gm/gm-config.yaml $CONFIG_FILE

# prepare the config with empty kubeconfig and empty master url meaning accessing k8s by rest.InClusterConfig().
# here using sed command, alternative you can edit the config file manully.
sed -i 's@kubeConfig:.*@kubeConfig: ""@' $CONFIG_FILE
sed -i 's@master:.*@master: ""@' $CONFIG_FILE

sed -i "s@port:.*@port: $GM_PORT@" $CONFIG_FILE

# setting lc server
sed -i "s@http://localhost:9100@$LC_SERVER@" $CONFIG_FILE

```

3\. Build the GM image:
```shell
# build image from source OR use the gm image previous built.

# edit it with the truly base repo by your choice.
GM_IMAGE=$IMAGE_REPO/sedna-gm:$IMAGE_TAG

make gmimage IMAGE_REPO=$IMAGE_REPO IMAGE_TAG=$IMAGE_TAG

# push image to registry, login to registry first if needed
docker push $GM_IMAGE
```

4\. Create gm configmap:
```shell
# create configmap from $CONFIG_FILE
CONFIG_NAME=gm-config   # customize this configmap name
kubectl create -n sedna configmap $CONFIG_NAME --from-file=$CONFIG_FILE
```

5\. Deploy GM as deployment:
```shell
# we assign gm to the node which edge node can access to.
# here current terminal node name, i.e. the k8s master node.
# remember the GM_IP
GM_NODE_NAME=$(hostname)

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
      port: $GM_PORT
      targetPort: $GM_PORT
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
        image: $GM_IMAGE
        command: ["sedna-gm", "--config", "/config/$CONFIG_FILE", "-v2"]
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
            name: $CONFIG_NAME
EOF
```

6\. Check the GM status:
```shell
kubectl get deploy -n sedna gm
```

### Deploy LC
Prerequisites:
1. Run GM successfully.
2. Get the bind address/port of GM.

Steps:

1\. Build LC image:
```shell
LC_IMAGE=$IMAGE_REPO/sedna-lc:$IMAGE_TAG

make lcimage IMAGE_REPO=$IMAGE_REPO IMAGE_TAG=$IMAGE_TAG

# push image to registry, login to registry first if needed
docker push $LC_IMAGE
```

2\. Deploy LC as k8s daemonset:
```shell
gm_node_port=$(kubectl -n sedna get svc gm -ojsonpath='{.spec.ports[0].nodePort}')

# fill the GM_NODE_NAME's ip which edge node can access to.
# such as gm_node_ip=192.168.0.9
# gm_node_ip=<GM_NODE_NAME_IP_ADDRESS>
# here try to get node ip by kubectl
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
          image: $LC_IMAGE
          env:
            - name: GM_ADDRESS
              value: $GM_ADDRESS
            - name: BIND_PORT
              value: "$LC_PORT"
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

3\. Check the LC status:
```shell
kubectl get ds lc -n sedna

kubectl get pod -n sedna
```

[git_tool]:https://git-scm.com/downloads
[go_tool]:https://golang.org/dl/
[kubeedge]:https://github.com/kubeedge/kubeedge
[kubernetes]:https://kubernetes.io/
