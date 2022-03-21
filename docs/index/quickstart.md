
# Quick Start


The following is showing how to run a joint inference job by sedna.
## Quick Start 

#### 0. Check the Environment

For Sedna all-in-one installation, it requires you:
  - 1 VM **(one machine is OK, cluster is not required)**
  - 2 CPUs or more
  - 2GB+ free memory, depends on node number setting
  - 10GB+ free disk space
  - Internet connection(docker hub, github etc.)
  - Linux platform, such as ubuntu/centos
  - Docker 17.06+

you can check the docker version by the following command, 
```bash
docker -v
```
after doing that, the output will be like this, that means your version fits the bill.
```
Docker version 19.03.6, build 369ce74a3c
```


#### 1. Deploy Sedna
Sedna provides three deployment methods, which can be selected according to your actual situation:

- [Install Sedna AllinOne](setup/all-in-one.md). (used for development, here we use it)
- [Install Sedna local up](setup/local-up.md).  
- [Install Sedna on a cluster](setup/install.md).

The [all-in-one script](/scripts/installation/all-in-one.sh) is used to install Sedna along with a mini Kubernetes environment locally, including:
  - A Kubernetes v1.21 cluster with multi worker nodes, default zero worker node.
  - KubeEdge with multi edge nodes, default is latest KubeEdge and one edge node.
  - Sedna, default is the latest version.

  ```bash
  curl https://raw.githubusercontent.com/kubeedge/sedna/master/scripts/installation/all-in-one.sh | NUM_EDGE_NODES=1 bash -
  ```

Then you get two nodes `sedna-mini-control-plane` and `sedna-mini-edge0`,you can get into each node by following command:

```bash
# get into cloud node
docker exec -it sedna-mini-control-plane bash
```

```bash
# get into edge node
docker exec -it sedna-mini-edge0 bash
```
 
#### 1. Prepare Data and Model File

* step1: download [little model](https://kubeedge.obs.cn-north-1.myhuaweicloud.com/examples/helmet-detection-inference/little-model.tar.gz) to your edge node.

```
mkdir -p /data/little-model
cd /data/little-model
wget https://kubeedge.obs.cn-north-1.myhuaweicloud.com/examples/helmet-detection-inference/little-model.tar.gz
tar -zxvf little-model.tar.gz
```

* step2: download [big model](https://kubeedge.obs.cn-north-1.myhuaweicloud.com/examples/helmet-detection-inference/big-model.tar.gz) to your cloud node.

```
mkdir -p /data/big-model
cd /data/big-model
wget https://kubeedge.obs.cn-north-1.myhuaweicloud.com/examples/helmet-detection-inference/big-model.tar.gz
tar -zxvf big-model.tar.gz
```

#### 2. Create Big Model Resource Object for Cloud
In cloud node:
```
kubectl create -f - <<EOF
apiVersion: sedna.io/v1alpha1
kind:  Model
metadata:
  name: helmet-detection-inference-big-model
  namespace: default
spec:
  url: "/data/big-model/yolov3_darknet.pb"
  format: "pb"
EOF
```

#### 3. Create Little Model Resource Object for Edge
In cloud node:
```
kubectl create -f - <<EOF
apiVersion: sedna.io/v1alpha1
kind: Model
metadata:
  name: helmet-detection-inference-little-model
  namespace: default
spec:
  url: "/data/little-model/yolov3_resnet18.pb"
  format: "pb"
EOF
```

#### 4. Create JointInferenceService 

Note the setting of the following parameters, which have to same as the script [little_model.py](/examples/joint_inference/helmet_detection_inference/little_model/little_model.py):
- hardExampleMining: set hard example algorithm from {IBT, CrossEntropy} for inferring in edge side.
- video_url: set the url for video streaming. 
- all_examples_inference_output: set your output path for the inference results.
- hard_example_edge_inference_output: set your output path for results of inferring hard examples in edge side.
- hard_example_cloud_inference_output: set your output path for results of inferring hard examples in cloud side.

Make preparation in edge node
```
mkdir -p /joint_inference/output
```

Create joint inference service
```
CLOUD_NODE="sedna-mini-control-plane"
EDGE_NODE="sedna-mini-edge0"

kubectl create -f https://raw.githubusercontent.com/jaypume/sedna/main/examples/joint_inference/helmet_detection_inference/helmet_detection_inference.yaml 
```


#### 5. Check Joint Inference Status

```
kubectl get jointinferenceservices.sedna.io
```

#### 6. Mock Video Stream for Inference in Edge Side

* step1: install the open source video streaming server [EasyDarwin](https://github.com/EasyDarwin/EasyDarwin/tree/dev).
* step2: start EasyDarwin server.
* step3: download [video](https://kubeedge.obs.cn-north-1.myhuaweicloud.com/examples/helmet-detection-inference/video.tar.gz).
* step4: push a video stream to the url (e.g., `rtsp://localhost/video`) that the inference service can connect.

```
wget https://github.com/EasyDarwin/EasyDarwin/releases/download/v8.1.0/EasyDarwin-linux-8.1.0-1901141151.tar.gz
tar -zxvf EasyDarwin-linux-8.1.0-1901141151.tar.gz
cd EasyDarwin-linux-8.1.0-1901141151
./start.sh

mkdir -p /data/video
cd /data/video
wget https://kubeedge.obs.cn-north-1.myhuaweicloud.com/examples/helmet-detection-inference/video.tar.gz
tar -zxvf video.tar.gz

ffmpeg -re -i /data/video/video.mp4 -vcodec libx264 -f rtsp rtsp://localhost/video
```

### Check Inference Result

You can check the inference results in the output path (e.g. `/joint_inference/output`) defined in the JointInferenceService config.
* the result of edge inference vs the result of joint inference
![](../../ /images/inference-result.png)
![](../../examples/joint_inference/helmet_detection_inference/images/inference-result.png)

## API

- control-plane: Please refer to this [link](api/crd).
- Lib: Please refer to this [link](api/lib).

## Contributing

Contributions are very welcome!

- control-plane: Please refer to this [link](contributing/control-plane/development.md).
- Lib: Please refer to this [link](contributing/lib/development.md).

## Community

Sedna is an open source project and in the spirit of openness and freedom, we welcome new contributors to join us. 
You can get in touch with the community according to the ways:
* [Github Issues](https://github.com/kubeedge/sedna/issues)
* [Regular Community Meeting](https://zoom.us/j/4167237304)
* [slack channel](https://app.slack.com/client/TDZ5TGXQW/C01EG84REVB/details)

