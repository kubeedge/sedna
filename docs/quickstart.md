
# Quick Start

## Introduction

Sedna provide some examples of running Sedna jobs in [here](/examples/README.md)

Here is a general guide to quick start an incremental learning job.

### Get Sedna

You can find the latest Sedna release [here](https://github.com/kubeedge/sedna/releases).

### Deploying Sedna

Sedna provides two deployment methods, which can be selected according to your actual situation:

- Install Sedna on a cluster Step By Step: [guide here](setup/install.md).
- Install Sedna AllinOne : [guide here](setup/local-up.md).

### Component
Sedna consists of the following components：

![Architecture](./proposals/images/framework.png)

#### GlobalManager
* Unified edge-cloud synergy AI task management
* Cross edge-cloud synergy management and collaboration
* Central Configuration Management

#### LocalController
* Local process control of edge-cloud synergy AI tasks
* Local general management: model, dataset, and status synchronization


#### Worker
* Do inference or training, based on existing ML framework.
* Launch on demand, imagine they are docker containers.
* Different workers for different features.
* Could run on edge or cloud.


#### Lib
* Expose the Edge AI features to applications, i.e. training or inference programs.


### System Design

There are three stages in a [incremental learning job](./proposals/incremental-learning.md): train/eval/deploy.

![](./proposals/images/incremental-learning-state-machine.png)

## Deployment Guide

### 1. Prepare

#### 1.1 Deployment Planning

In this example, there is only one host with two nodes, which had creating a Kubernetes cluster with `kind`.

| NAME  | ROLES | Ip Address                    | Operating System        | Host Configuration | Storage | Deployment Module                                            |
| ----- | ------- | ----------------------------- | ----------------------- | ------------------ | ------- | ------------------------------------------------------------ |
| edge-node  | agent,edge   | 192.168.0.233 | Ubuntu 18.04.5 LTS  | 8C16G              | 500G    | LC，lib, inference worker |
| sedna-control-plane | control-plane,master    | 172.18.0.2                   | Ubuntu 20.10  | 8C16G              | 500G    | GM，LC，lib，training worker，evaluate worker |

#### 1.2 Network Requirements

In this example the node **sedna-control-plane** has a internal-ip `172.18.0.2`, and **edge-node** can access it.

### 2. Project Deployment

#### 2.1 (optional) create virtual env

```bash
python3.6 -m venv venv
source venv/bin/activate
pip3 install -U pip
```

#### 2.2 install sedna SDK

```bash
cd $SENDA_ROOT/lib
python3.6 setup.py bdist_wheel
pip3 install dist/sedna*.whl
``` 

#### 2.3 Prepare your machine learning model and datasets

##### 2.3.1 Encapsulate an Estimators 

Sedna implements several pre-made Estimators in [example](/examples), your can find them from the python scripts called `interface`.
Sedna supports the Estimators build from popular AI frameworks, such as TensorFlow, Pytorch, PaddlePaddle, MindSpore. Also Custom estimators can be used according to our interface document. 
All Estimators—pre-made or custom ones—are classes should encapsulate the following actions:
- Training
- Evaluation
- Prediction
- Export/load

Follow [here](/lib/sedna/README.md) for more details, a [toy_example](/examples/incremental_learning/helmet_detection/training/interface.py) like:


```python

os.environ['BACKEND_TYPE'] = 'TENSORFLOW'

class Estimator:

    def __init__(self, **kwargs):
        ...
    
    def train(self, train_data, valid_data=None, **kwargs):     
        ...
    
    def evaluate(self, data, **kwargs):
       ...

    def predict(self, data, **kwargs):
        ...

    def load(self, model_url, **kwargs):
        ...

    def save(self, model_path, **kwargs):
        ...

    def get_weights(self):
        ...

    def set_weights(self, weights):
        ...
```

##### 2.3.2 Dataset prepare

In incremental_learning jobs, the following files will be indispensable:

- base model: tensorflow object detection Fine-tuning a model from an existing checkpoint.
- deploy model:  tensorflow object detection model, for inference.
- train data: images with label use for Fine-tuning model.
- test data: video stream use for model inference.

```bash
# download models, including base model and deploy model.

cd /
wget https://kubeedge.obs.cn-north-1.myhuaweicloud.com/examples/helmet-detection/models.tar.gz
tar -zxvf models.tar.gz

# download train data

cd /data/helmet_detection  # notes: files here will be monitored and used to trigger the incremental training
wget  https://kubeedge.obs.cn-north-1.myhuaweicloud.com/examples/helmet-detection/dataset.tar.gz
tar -zxvf dataset.tar.gz

# download test data

cd /incremental_learning/video/
wget https://kubeedge.obs.cn-north-1.myhuaweicloud.com/examples/helmet-detection/video.tar.gz
tar -zxvf video.tar.gz
 
```

#### 2.3.3 Scripts prepare

In incremental_learning jobs, the following scripts will be indispensable:

- train.py: script for model fine-tuning/training.
- eval.py: script for model evaluate.
- inference.py: script for data inference.

You can also find demos [here](/examples/incremental_learning/helmet_detection).

Some interfaces should be learn in job pipeline:

- `BaseConfig` provides the capability of obtaining the config from env

```python

from sedna.common.config import BaseConfig

train_dataset_url = BaseConfig.train_dataset_url
model_url = BaseConfig.model_url

```

- `Context` provides the capability of obtaining the context from CRD

```python
from sedna.common.config import Context

obj_threshold = Context.get_parameters("obj_threshold")
nms_threshold = Context.get_parameters("nms_threshold")
input_shape = Context.get_parameters("input_shape")
epochs = Context.get_parameters('epochs')
batch_size = Context.get_parameters('batch_size')

```

- `datasources` base class, as that core feature of sedna require identifying the features and labels from data input, we specify that the first parameter for train/evaluate of the ML framework

```python
from sedna.datasources import BaseDataSource


train_data = BaseDataSource(data_type="train")
train_data.x = []
train_data.y = []
for item in mnist_ds.create_dict_iterator():
     train_data.x.append(item["image"].asnumpy())
     train_data.y.append(item["label"].asnumpy())
```

-  `sedna.core` contain all edge-cloud features,  Please note that each feature has its own parameters.
- **Hard Example Mining Algorithms** in IncrementalLearning named `hard_example_mining`  

```python
from sedna.core.incremental_learning import IncrementalLearning

hard_example_mining = IncrementalLearning.get_hem_algorithm_from_config(
    threshold_img=0.9
)

# initial an incremental instance
incremental_instance = IncrementalLearning(
    estimator=Estimator,
    hard_example_mining=hem_dict
)

# Call the interface according to the job state

# train.py
incremental_instance.train(train_data=train_data, epochs=epochs,
                          batch_size=batch_size,
                          class_names=class_names,
                          input_shape=input_shape,
                          obj_threshold=obj_threshold,
                          nms_threshold=nms_threshold)

# inference
results, _, is_hard_example = incremental_instance.inference(
            data, input_shape=input_shape)
            

```


### 3. Configuration

##### 3.1 Prepare Image
This example uses the image:  
```
kubeedge/sedna-example-incremental-learning-helmet-detection:v0.4.0
```

This image is generated by the script [build_images.sh](/examples/build_image.sh), used for creating training, eval and inference worker.  

##### 3.2 Create Incremental Job
In this example, `$WORKER_NODE` is a custom node, you can fill it which you actually run.  

```
WORKER_NODE="edge-node" 
```

- Create Dataset  

```
kubectl create -f - <<EOF
apiVersion: sedna.io/v1alpha1
kind: Dataset
metadata:
  name: incremental-dataset
spec:
  url: "/data/helmet_detection/train_data/train_data.txt"
  format: "txt"
  nodeName: $WORKER_NODE
EOF
```

- Create Initial Model to simulate the initial model in incremental learning scenario.  

```
kubectl create -f - <<EOF
apiVersion: sedna.io/v1alpha1
kind: Model
metadata:
  name: initial-model
spec:
  url : "/models/base_model"
  format: "ckpt"
EOF
```

- Create Deploy Model  

```
kubectl create -f - <<EOF
apiVersion: sedna.io/v1alpha1
kind: Model
metadata:
  name: deploy-model
spec:
  url : "/models/deploy_model/saved_model.pb"
  format: "pb"
EOF
```


### 4. Run

* incremental learning supports hot model updates and cold model updates. Job support 
cold model updates default. If you want to use hot model updates, please to add the following fields:  

```yaml
deploySpec:
  model:
    hotUpdateEnabled: true
    pollPeriodSeconds: 60  # default value is 60
```

* create the job:

```
IMAGE=kubeedge/sedna-example-incremental-learning-helmet-detection:v0.4.0

kubectl create -f - <<EOF
apiVersion: sedna.io/v1alpha1
kind: IncrementalLearningJob
metadata:
  name: helmet-detection-demo
spec:
  initialModel:
    name: "initial-model"
  dataset:
    name: "incremental-dataset"
    trainProb: 0.8
  trainSpec:
    template:
      spec:
        nodeName: $WORKER_NODE
        containers:
          - image: $IMAGE
            name:  train-worker
            imagePullPolicy: IfNotPresent
            args: ["train.py"]
            env:
              - name: "batch_size"
                value: "32"
              - name: "epochs"
                value: "1"
              - name: "input_shape"
                value: "352,640"
              - name: "class_names"
                value: "person,helmet,helmet-on,helmet-off"
              - name: "nms_threshold"
                value: "0.4"
              - name: "obj_threshold"
                value: "0.3"
    trigger:
      checkPeriodSeconds: 60
      timer:
        start: 02:00
        end: 20:00
      condition:
        operator: ">"
        threshold: 500
        metric: num_of_samples
  evalSpec:
    template:
      spec:
        nodeName: $WORKER_NODE
        containers:
          - image: $IMAGE
            name:  eval-worker
            imagePullPolicy: IfNotPresent
            args: ["eval.py"]
            env:
              - name: "input_shape"
                value: "352,640"
              - name: "class_names"
                value: "person,helmet,helmet-on,helmet-off"                    
  deploySpec:
    model:
      name: "deploy-model"
      hotUpdateEnabled: true
      pollPeriodSeconds: 60 
    trigger:
      condition:
        operator: ">"
        threshold: 0.1
        metric: precision_delta
    hardExampleMining:
      name: "IBT"
      parameters:
        - key: "threshold_img"
          value: "0.9"
        - key: "threshold_box"
          value: "0.9"
    template:
      spec:
        nodeName: $WORKER_NODE
        containers:
        - image: $IMAGE
          name:  infer-worker
          imagePullPolicy: IfNotPresent
          args: ["inference.py"]
          env:
            - name: "input_shape"
              value: "352,640"
            - name: "video_url"
              value: "file://video/video.mp4"
            - name: "HE_SAVED_URL" 
              value: "/he_saved_url"
          volumeMounts:
          - name: localvideo
            mountPath: /video/
          - name: hedir
            mountPath: /he_saved_url
          resources:  # user defined resources
            limits:
              memory: 2Gi
        volumes:   # user defined volumes
          - name: localvideo
            hostPath:
              path: /incremental_learning/video/
              type: DirectoryorCreate
          - name: hedir
            hostPath:
              path:  /incremental_learning/he/
              type: DirectoryorCreate
  outputDir: "/output"
EOF
```

1. The `Dataset` describes data with labels and `HE_SAVED_URL` indicates the address of the deploy container for uploading hard examples. Users will mark label for the hard examples in the address.
2. Ensure that the path of outputDir in the YAML file exists on your node. This path will be directly mounted to the container.


### 5. Monitor

### Check Incremental Learning Job
Query the service status:  

```
kubectl get incrementallearningjob helmet-detection-demo
```

In the `IncrementalLearningJob` resource helmet-detection-demo, the following trigger is configured:  

```
trigger:
  checkPeriodSeconds: 60
  timer:
    start: 02:00
    end: 20:00
  condition:
    operator: ">"
    threshold: 500
    metric: num_of_samples
```

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

