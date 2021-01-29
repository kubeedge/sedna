# Using Federated Learning Job in Surface Defect Detection Scenario
This case introduces how to use federated learning job in surface defect detection scenario.
In the safety surface defect detection, data is scattered in different places (such as server node, camera or others) and cannot be aggregated due to data privacy and bandwidth. As a result, we cannot use all the data for training.
Using Federated Learning, we can solve the problem. Each place uses its own data for model training ,uploads the weight to the cloud for aggregation, and obtains the aggregation result for model update.


## Surface Defect Detection Experiment
> Assume that there are two edge nodes (edge1 and edge2) and a cloud node. Data on the edge nodes cannot be migrated to the cloud due to privacy issues.
> Base on this scenario, we will demonstrate the surface inspection.

### Install Sedna

Follow the [Sedna installation document](/docs/setup/install.md) to install Sedna.
 
### Prepare Dataset

Download [dataset](https://github.com/abin24/Magnetic-tile-defect-datasets.) and the [label file](/examples/federated_learning/surface_defect_detection/data/1.txt) to `/data` of edge1.  
Download [dataset](https://github.com/abin24/Magnetic-tile-defect-datasets.) and the [label file](/examples/federated_learning/surface_defect_detection/data/2.txt) to `/data` of edge2.

### Prepare Script

Download the script [aggregate.py](/examples/federated_learning/surface_defect_detection/aggregation_worker/aggregate.py) to the `/code` of cloud node.

Download the script [training_worker](/examples/federated_learning/surface_defect_detection/training_worker/train.py) to the `/code` of edge1 and edge2.


### Create Federated Learning Job 

#### Create Dataset

```
# create dataset for edge1
kubectl create -f - <<EOF
apiVersion: sedna.io/v1alpha1
kind: Dataset
metadata:
  name: "edge1-surface-defect-detection-dataset"
spec:
  url: "/data/1.txt"
  format: "txt"
  nodeName: "edge1"
EOF

# create dataset for edge2
kubectl create -f - <<EOF
apiVersion: sedna.io/v1alpha1
kind: Dataset
metadata:
  name: "edge2-surface-defect-detection-dataset"
spec:
  url: "/data/2.txt"
  format: "txt"
  nodeName: "edge2"
EOF
```

#### Create Model

```
kubectl create -f - <<EOF
apiVersion: sedna.io/v1alpha1
kind: Model
metadata:
  name: "surface-defect-detection-model"
spec:
  url: "/model"
  format: "ckpt"
EOF
```

#### Start Federated Learning Job

```
kubectl create -f - <<EOF
apiVersion: sedna.io/v1alpha1
kind: FederatedLearningJob
metadata:
  name: surface-defect-detection
spec:
  aggregationWorker:
    model:
      name: "surface-defect-detection-model"
    nodeName: "cloud0"
    workerSpec:
      scriptDir: "/code"
      scriptBootFile: "aggregate.py"
      frameworkType: "tensorflow"
      frameworkVersion: "2.3"
      parameters:
        - key: "exit_round"
          value: "3"
  trainingWorkers:
    - nodeName: "edge1"
      dataset:
        name: "edge-1-surface-defect-detection-dataset"
      workerSpec:
        scriptDir: "/code"
        scriptBootFile: "train.py"
        frameworkType: "tensorflow"
        frameworkVersion: "2.3"
        parameters:
          - key: "batch_size"
            value: "32"
          - key: "learning_rate"
            value: "0.001"
          - key: "epochs"
            value: "1"
    - nodeName: "edge2"
      dataset:
          name: "edge-2-surface-defect-detection-dataset"
      workerSpec:
        scriptDir: "/code"
        scriptBootFile: "train.py"
        frameworkType: "tensorflow"
        frameworkVersion: "2.3"
        parameters:
          - key: "batch_size"
            value: "32"
          - key: "learning_rate"
            value: "0.001"
          - key: "epochs"
            value: "1"
EOF
```

### Check Federated Learning Status

```
kubectl get federatedlearningjob surface-defect-detection
```

### Check Federated Learning Train Result
After the job completed, you will find the model generated on the path `/model` in edge1 and edge2.