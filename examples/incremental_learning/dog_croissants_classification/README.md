# Dog-Croissants-classification Demo
## Prepare Model
```shell
cd /
#download ckpt file here:https://drive.google.com/file/d/1DdIFn1uz9Z4cwvbcf4QFRoxvhqcrPWpo/view?usp=sharing
tar -xvf models.tar.gz
```

## Prepare for inference worker
```shell
cd /
#download dataset here: https://drive.google.com/file/d/1zGrQ8Qr3qCT01PXINdm2PxN2C-x0qvmy/view?usp=sharing
mkdir /output
```
We provide images to inference, train, evaluate in this dataset

## build docker image
```shell
$  docker build -f incremental-learning-dog-croissants-classification.Dockerfile -t test/dog:v0.1 .
```
You can build your own image by referring to this dockerfile. 

## Create Incremental Job
```shell
WORKER_NODE="edge-node" 
```
Create Dataset CRD
```shell
kubectl create -f - <<EOF
apiVersion: sedna.io/v1alpha1
kind: Dataset
metadata:
  name: incremental-dataset
spec:
  url: "/data/dog_croissants/train_data.txt"
  format: "txt"
  nodeName: $WORKER_NODE
EOF
```
Create initial Model to simulate the initial model in incremental learning scenario.
```shell
kubectl create -f - <<EOF
apiVersion: sedna.io/v1alpha1
kind: Model
metadata:
  name: initial-model
spec:
  url : "/models/base_model/base_model.ckpt"
  format: "ckpt"
EOF
```
Create Deploy Model CRD
```shell
kubectl create -f - <<EOF
apiVersion: sedna.io/v1alpha1
kind: Model
metadata:
  name: deploy-model
spec:
  url : "/models/deploy_model/deploy_model.ckpt"
  format: "ckpt"
EOF
```
Create the job CRD
```shell
IMAGE=lj1ang/dog:v0.40
kubectl create -f - <<EOF
apiVersion: sedna.io/v1alpha1
kind: IncrementalLearningJob
metadata:
  name: dog-croissants-classification-demo
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
            name: train-worker
            imagePullPolicy: IfNotPresent
            args: [ "train.py" ]
            env:
              - name: "batch_size"
                value: "2"
              - name: "epochs"
                value: "2"
              - name: "input_shape"
                value: "224"
              - name: "class_names"
                value: "Croissants, Dog"
              - name: "num_parallel_workers"
                value: "2"
    trigger:
      checkPeriodSeconds: 60
      timer:
        start: 02:00
        end: 20:00
      condition:
        operator: ">"
        threshold: 50
        metric: num_of_samples
  evalSpec:
    template:
      spec:
        nodeName: $WORKER_NODE
        containers:
          - image: $IMAGE
            name: eval-worker
            imagePullPolicy: IfNotPresent
            args: [ "eval.py" ]
            env:
              - name: "input_shape"
                value: "224"
              - name: "batch_size"
                value: "2"
              - name: "num_parallel_workers"
                value: "2"              
              - name: "class_names"
                value: "Croissants, Dog"
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
      name: "Random"
      parameters:
        - key: "random_ratio"
          value: "0.3"
    template:
      spec:
        nodeName: $WORKER_NODE
        containers:
          - image: $IMAGE
            name: infer-worker
            imagePullPolicy: IfNotPresent
            args: [ "inference.py" ]
            env:
              - name: "input_shape"
                value: "224"
              - name: "infer_url"
                value: "/infer"
              - name: "HE_SAVED_URL"
                value: "/he_saved_url"
            volumeMounts:
              - name: localinferdir
                mountPath: /infer
              - name: hedir
                mountPath: /he_saved_url
            resources: # user defined resources
              limits:
                memory: 3Gi
        volumes: # user defined volumes
          - name: localinferdir
            hostPath:
              path: /incremental_learning/infer/
              type: DirectoryOrCreate
          - name: hedir
            hostPath:
              path: /incremental_learning/he/
              type: DirectoryOrCreate
  outputDir: "/output"
EOF
```
Here we use Random as hard example algorithm, which will select infer images randomly as hard example. 
This is used in models with very high accuracy, but we need hard examples from edge node in real scenario. 

## trigger
```shell
cd /data/dog_croissants
mv train_data.txt.full train_data.txt
```
Then we will get into train stage. After that, evaluate stage is the next.
For more information, you can see what happened by check the logs of lc in your `$WORKER_NODE` and gm in your cloud node.
Also, you can find more information by reading this doc:https://sedna.readthedocs.io/en/latest/proposals/incremental-learning.html
