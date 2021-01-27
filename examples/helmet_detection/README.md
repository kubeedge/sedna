# Using Incremental Learning Job in Helmet Detection Scenario

This document introduces how to use incremental learning job in helmet detectioni scenario. Using the incremental learning job, our application can automatically retrains, evaluates, and updates models based on the data generated at the edge.

## Helmet Detection Experiment

### Install Neptune

Follow the [Neptune installation document](/docs/setup/install.md) to install Neptune.

### Prepare Data and Model

Download dataset and model to your node:
* step 1: download [dataset](https://edgeai-neptune.obs.cn-north-1.myhuaweicloud.com/examples/helmet-detection/dataset.tar.gz)
```
mkdir -p /data/helmet_detection
cd /data/helmet_detection
tar -zxvf dataset.tar.gz
```
* step 2: download [base model](https://edgeai-neptune.obs.cn-north-1.myhuaweicloud.com/examples/helmet-detection/model.tar.gz)
```
mkdir /model
cd /model
tar -zxvf model.tar.gz
```
### Prepare Script
Download the [scripts](/examples/helmet_detection/training) to the path `code` of your node


### Create Incremental Job

Create Dataset

```
kubectl create -f - <<EOF
apiVersion: neptune.io/v1alpha1
kind: Dataset
metadata:
  name: incremental-dataset
  namespace: neptune-test
spec:
  url: "/data/helmet_detection/dataset/data.txt"
  format: "txt"
  nodeName: "cloud0"
EOF
```

Create Initial Model

```
kubectl create -f - <<EOF
apiVersion: neptune.io/v1alpha1
kind: Model
metadata:
  name: initial-model
  namespace: neptune-test
spec:
  url : "/model/base_model"
  format: "ckpt"
EOF
```

Create Deploy Model

```
kubectl create -f - <<EOF
apiVersion: neptune.io/v1alpha1
kind: Model
metadata:
  name: deploy-model
  namespace: neptune-test
spec:
  url : "/deploy/model.pb"
  format: "pb"
EOF
```

Start The Incremental Learning Job

```
kubectl create -f - <<EOF
apiVersion: neptune.io/v1alpha1
kind: IncrementalLearningJob
metadata:
  name: helmet-detection-demo
  namespace: neptune-test
spec:
  initialModel:
    name: "initial-model"
  dataset:
    name: "incremental-dataset"
    trainProb: 0.8
  trainSpec:
    workerSpec:
      scriptDir: "/code"
      scriptBootFile: "train.py"
      frameworkType: "tensorflow"
      frameworkVersion: "1.15"
      parameters:
        - key: "batch_size"
          value: "32"
        - key: "epochs"
          value: "1"
        - key: "input_shape"
          value: "352,640"
        - key: "class_names"
          value: "person,helmet,helmet-on,helmet-off"
        - key: "nms_threshold"
          value: "0.4"
        - key: "obj_threshold"
          value: "0.3"
    trigger:
      checkPeriodSeconds: 60
      timer:
        start: 02:00
        end: 04:00
      condition:
        operator: ">"
        threshold: 500
        metric: num_of_samples
  evalSpec:
    workerSpec:
      scriptDir: "/code"
      scriptBootFile: "eval.py"
      frameworkType: "tensorflow"
      frameworkVersion: "1.15"
      parameters:
        - key: "input_shape"
          value: "352,640"
        - key: "class_names"
          value: "person,helmet,helmet-on,helmet-off"
  deploySpec:
    model:
      name: "deploy-model"
    trigger:
      condition:
        operator: ">"
        threshold: 0.1
        metric: precision_delta
    nodeName: "cloud0"
    hardExampleMining:
      name: "IBT"
    workerSpec:
      scriptDir: "/code"
      scriptBootFile: "inference.py"
      frameworkType: "tensorflow"
      frameworkVersion: "1.15"
      parameters:
        - key: "input_shape"
          value: "352,640"
        - key: "video_url"
          value: "rtsp://localhost/video"
        - key: "HE_SAVED_URL" 
          value: "/he_saved_url"
  nodeName: "cloud0"
  outputDir: "/output"
EOF
```
1. The `Dataset` describes data with labels and `HE_SAVED_URL` indicates the address of the deploy container for uploading hard examples. Users will mark label for the hard examples in the address.
2. Ensure that the path of outputDir in the YAML file exists on your node. This path will be directly mounted to the container.



### Mock Video Stream for Inference in Edge Side

* step1: install the open source video streaming server [EasyDarwin](https://github.com/EasyDarwin/EasyDarwin/tree/dev).
* step2: start EasyDarwin server.
* step3: download [video](https://edgeai-neptune.obs.cn-north-1.myhuaweicloud.com/examples/helmet-detection/video.tar.gz).
* step4: push a video stream to the url (e.g., `rtsp://localhost/video`) that the inference service can connect.

```
wget https://github.com/EasyDarwin/EasyDarwin/releases/download/v8.1.0/EasyDarwin-linux-8.1.0-1901141151.tar.gz --no-check-certificate
tar -zxvf EasyDarwin-linux-8.1.0-1901141151.tar.gz
cd EasyDarwin-linux-8.1.0-1901141151
./start.sh

mkdir -p /data/video
cd /data/video
tar -zxvf video.tar.gz
ffmpeg -re -i /data/video/helmet-detection.mp4 -vcodec libx264 -f rtsp rtsp://localhost/video
```


### Check Incremental Job Result

query the service status
```
kubectl get incrementallearningjob helmet-detection-demo -n neptune-test
```

after the job completed, we can view the updated model in the /output directory in cloud0 node

