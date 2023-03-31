This tutorial targets at [lifelong learning job in thermal comfort prediction scenario](https://github.com/kubeedge/sedna/blob/main/examples/lifelong_learning/atcii/README.md), and includes how to run the default example with customized configurations, as well as how to develop and integrate user-defined modules.
# 1 Configure Default Example
With Custom Resource Definitions (CRDs) of Kubernetes, developers are able to configure the default lifelong process using the following configurations.
## 1.1 Install Sedna
Follow the [Sedna installation document](https://sedna.readthedocs.io/en/v0.5.0/setup/install.html) to install Sedna.
## 1.2 Prepare Dataset
In the default example, [ASHRAE Global Thermal Comfort Database II (ATCII)](https://datadryad.org/stash/dataset/doi:10.6078/D1F671) is used to initiate lifelong learning job.

We provide a well-processed [dataset](https://kubeedge.obs.cn-north-1.myhuaweicloud.com/examples/atcii-classifier/dataset.tar.gz), including train (trainData.csv), evaluation (testData.csv) and incremental (trainData2.csv) dataset.

```
cd /data
wget https://kubeedge.obs.cn-north-1.myhuaweicloud.com/examples/atcii-classifier/dataset.tar.gz
tar -zxvf dataset.tar.gz
```
## 1.3 Create Dataset
After preparing specific dataset and index file, users can configure them as the following example for training and evaluation.

Data will be automatically downloaded from where the index file indicates to the corresponding pods.

| Property | Required | Description |
|----------|----------|-------------|
|name|yes|Dataset name defined in metadata|
|url|yes|Url of dataset index file, which is generally stored in data node|
|format|yes|Format of dataset index file|
|nodeName|yes|Name of data node that stores data and dataset index file|

```
DATA_NODE = "cloud-node" 
```
```
kubectl create -f - <<EOF
apiVersion: sedna.io/v1alpha1
kind: Dataset
metadata:
  name: lifelong-atcii-dataset
spec:
  url: "/data/trainData.csv"
  format: "csv"
  nodeName: $DATA_NODE
EOF
```

## 1.4 Start Lifelong Learning Job
To run lifelong learning jobs, users need to configure their own lifelong learning CRDs in training, evaluation, and inference phases. The configuration process for these three phases is similar.

| Property | Required | Description |
|----------|----------|-------------|
|nodeName|yes|Name of the node where worker runs|
|dnsPolicy|yes|DNS policy set at pod level|
|imagePullPolicy|yes|Image pulling policy when local image does not exist|
|args|yes|Arguments to run images. In this example, it is the startup file of each stage| 
|env|no|Environment variables passed to each stage |
|trigger|yes|Configuration for when training begins|
|resourcs|yes|Limited or required resources of CPU and memory|
|volumeMounts|no|Specified path to be mounted to the host|
|volumes|no|Directory in the node which file systems in a worker are mounted to|

```
TRAIN_NODE = "cloud-node" 
EVAL_NODE = "cloud-node" 
INFER_NODE = "edge-node"
CLOUD_IMAGE = kubeedge/sedna-example-lifelong-learning-atcii-classifier:v0.5.0
```
```
kubectl create -f - <<EOF
apiVersion: sedna.io/v1alpha1
kind: LifelongLearningJob
metadata:
  name: atcii-classifier-demo
spec:
  dataset:
    name: "lifelong-atcii-dataset"
    trainProb: 0.8
  trainSpec:
    template:
      spec:
        nodeName: $TRAIN_NODE
        dnsPolicy: ClusterFirstWithHostNet
        containers:
          - image: $CLOUD_IMAGE
            name:  train-worker
            imagePullPolicy: IfNotPresent
            args: ["train.py"]  # training script
            env:  # Hyperparameters required for training
              - name: "early_stopping_rounds"
                value: "100"
              - name: "metric_name"
                value: "mlogloss"
    trigger:
      checkPeriodSeconds: 60
      timer:
        start: 02:00
        end: 24:00
      condition:
        operator: ">"
        threshold: 500
        metric: num_of_samples
  evalSpec:
    template:
      spec:
        nodeName: $EVAL_NODE
        dnsPolicy: ClusterFirstWithHostNet
        containers:
          - image: $CLOUD_IMAGE
            name:  eval-worker
            imagePullPolicy: IfNotPresent
            args: ["eval.py"]
            env:
              - name: "metrics"
                value: "precision_score"
              - name: "metric_param"
                value: "{'average': 'micro'}"
              - name: "model_threshold"  # Threshold for filtering deploy models
                value: "0.5"
  deploySpec:
    template:
      spec:
        nodeName: $INFER_NODE
        dnsPolicy: ClusterFirstWithHostNet
        containers:
        - image: $CLOUD_IMAGE
          name:  infer-worker
          imagePullPolicy: IfNotPresent
          args: ["inference.py"]
          env:
          - name: "UT_SAVED_URL"  # unseen tasks save path
            value: "/ut_saved_url"
          - name: "infer_dataset_url"  # simulation of the inference samples 
            value: "/data/testData.csv"
          volumeMounts:
          - name: utdir
            mountPath: /ut_saved_url
          - name: inferdata
            mountPath: /data/
          resources:  # user defined resources
            limits:
              memory: 2Gi
        volumes:   # user defined volumes
          - name: utdir
            hostPath:
              path: /lifelong/unseen_task/
              type: DirectoryOrCreate
          - name: inferdata
            hostPath:
              path:  /data/
              type: DirectoryOrCreate
  outputDir: "/output"
EOF
```
## 1.5 Check Lifelong Learning Job
**(1). Query lifelong learning service status**

```
kubectl get lifelonglearningjob atcii-classifier-demo
```

**(2). View pods related to lifelong learning job**

```
kubectl get pod
```

**(3). Process unseen tasks samples**

In a real word, we need to label the hard examples in our unseen tasks which storage in `UT_SAVED_URL` with annotation tools and then put the examples to `Dataset`'s url. By this way, we can realize the function of updating models based on the data generated at the edge.

**(4). View result files**

Artifacts including multi-task learning models, partitioned sample sets, etc. can be found in `outputDir`, and the inference result is stored in the `Dataset`'s url.

# 2 Develop and Integrate Customized Modules

## 2.1 Before Development

Before you start development, you should prepare the [development environment](https://github.com/kubeedge/sedna/blob/main/docs/contributing/prepare-environment.md) and learn about the [interface design of Sedna](https://sedna.readthedocs.io/en/latest/autoapi/lib/sedna/index.html).

## 2.2 Develop Sedna AI Module

The Sedna framework components are decoupled and the registration mechanism is used to combine functional components to facilitate function and algorithm expansion. For details about the Sedna architecture and main mechanisms, see [Lib README](https://github.com/kubeedge/sedna/blob/51219027a0ec915bf3afb266dc5f9a7fb3880074/lib/sedna/README.md).

The following contents explains how to develop customized AI modules of a Sedna project, including **dataset**, **base model**, **algorithm**, etc.

### 2.2.1 Import Service Datasets

During Sedna application development, the first problem encountered is how to import service data sets to Sedna. Sedna provides interfaces and public methods related to data conversion and sampling in the [Dataset class](https://github.com/kubeedge/sedna/blob/c763c1a90e74b4ff1ab0afa06fb976fbb5efa512/lib/sedna/datasources/__init__.py). 

All dataset classes of Sedna are inherited from the base class `sedna.datasources.BaseDataSource`. This base class defines the interfaces required by the dataset, provides attributes such as data_parse_func, save, and concat, and provides default implementation. The derived class can reload these default implementations as required.

We take `txt-format contain sets of images` as an example.

**(1). Inherite from BaseDataSource**

```python
class BaseDataSource:
    """
    An abstract class representing a :class:`BaseDataSource`.

    All datasets that represent a map from keys to data samples should subclass
    it. All subclasses should overwrite parse`, supporting get train/eval/infer
    data by a function. Subclasses could also optionally overwrite `__len__`,
    which is expected to return the size of the dataset.overwrite `x` for the
    feature-embedding, `y` for the target label.

    Parameters
    ----------
    data_type : str
        define the datasource is train/eval/test
    func: function
        function use to parse an iter object batch by batch
    """

    def __init__(self, data_type="train", func=None):
        self.data_type = data_type  # sample type: train/eval/test
        self.process_func = None
        if callable(func):
            self.process_func = func
        elif func:
            self.process_func = ClassFactory.get_cls(
                ClassType.CALLBACK, func)()
        self.x = None  # sample feature
        self.y = None  # sample label
        self.meta_attr = None  # special in lifelong learning

    def num_examples(self) -> int:
        return len(self.x)

    def __len__(self):
        return self.num_examples()

    def parse(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def is_test_data(self):
        return self.data_type == "test"

    def save(self, output=""):
        return FileOps.dump(self, output)
        
class TxtDataParse(BaseDataSource, ABC):
    """
    txt file which contain image list parser
    """

    def __init__(self, data_type, func=None):
        super(TxtDataParse, self).__init__(data_type=data_type, func=func)

    def parse(self, *args, **kwargs):
        pass
```

**(2). Define dataset parse function**

```python
def parse(self, *args, **kwargs):
    x_data = []
    y_data = []
    use_raw = kwargs.get("use_raw")
    for f in args:
        with open(f) as fin:
            if self.process_func:
                res = list(map(self.process_func, [
                           line.strip() for line in fin.readlines()]))
            else:
                res = [line.strip().split() for line in fin.readlines()]
        for tup in res:
            if not len(tup):
                continue
            if use_raw:
                x_data.append(tup)
            else:
                x_data.append(tup[0])
                if not self.is_test_data:
                    if len(tup) > 1:
                        y_data.append(tup[1])
                    else:
                        y_data.append(0)
    self.x = np.array(x_data)
    self.y = np.array(y_data)
```
**(3). Commission**

The preceding implementation can be directly used in the PipeStep in Sedna or independently invoked. The code for independently invoking is as follows:

```python
import os
import unittest


def _load_txt_dataset(dataset_url):
    # use original dataset url,
    # see https://github.com/kubeedge/sedna/issues/35
    return os.path.abspath(dataset_url)


class TestDataset(unittest.TestCase):

    def test_txtdata(self):
        train_data = TxtDataParse(data_type="train", func=_load_txt_dataset)
        train_data.parse(train_dataset_url, use_raw=True)
        self.assertEqual(len(train_data), 1)


if __name__ == "__main__":
    unittest.main()
```

### 2.2.2 Modify Base Model

Estimator is a high-level API that greatly simplifies machine learning programming. Estimators encapsulate training, evaluation, prediction, and exporting for your model.

**(1). Define an Estimator**

In lifelong learning ATCII case, Estimator is defined in [interface.py](https://github.com/kubeedge/sedna/blob/c763c1a90e74b4ff1ab0afa06fb976fbb5efa512/examples/lifelong_learning/atcii/interface.py), and users can replace the existing XGBoost model with the model that best suits their purpose.

 ```python
# XGBOOST
    
import os
import xgboost
 
os.environ['BACKEND_TYPE'] = 'SKLEARN'
 
XGBEstimator = xgboost.XGBClassifier(
        learning_rate=0.1,
        n_estimators=600,
        max_depth=2,
        min_child_weight=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softmax",
        num_class=3,
        nthread=4,
        seed=27
)
```

```python
# Customize
 
class Estimator:

    def __init__(self, **kwargs):
        ...
            
    def load(self, model_url=""):
        ...
            
    def save(self, model_path=None):
        ...

    def predict(self, data, **kwargs):
        ...
            
    def evaluate(self, valid_data, **kwargs):
        ...
            
    def train(self, train_data, valid_data=None, **kwargs):
        ...
```

**(2). Initialize a lifelong learning job**

```python
ll_job = LifelongLearning(
    estimator=Estimator,
    task_definition=task_definition,
)
```

Noted that `Estimator` is the base model for your lifelong learning job.

### 2.2.3 Develop Customized Algorithms

Users may need to develop new algorithms based on the basic classes provided by Sedna, such as `unseen task detect` in lifelong learning example.

Sedna provides a class called `class_factory.py` in `common` package, in which only a few lines of changes are required to integrate existing algorithms into Sedna.

The following content takes a hard example mining algorithm as an example to explain how to add an HEM algorithm to the Sedna hard example mining algorithm library.

**(1). Start from the `class_factory.py`**

First, let's start from the `class_factory.py`. Two classes are defined in `class_factory.py`, namely `ClassType` and `ClassFactory`.

`ClassFactory` can register the modules you want to reuse through decorators. For the new `ClassType.HEM` algorithm, the code is as follows:

```python

@ClassFactory.register(ClassType.HEM, alias="Threshold")
class ThresholdFilter(BaseFilter, abc.ABC):
    def __init__(self, threshold=0.5, **kwargs):
        self.threshold = float(threshold)

    def __call__(self, infer_result=None):
        # if invalid input, return False
        if not (infer_result
                and all(map(lambda x: len(x) > 4, infer_result))):
            return False

        image_score = 0

        for bbox in infer_result:
            image_score += bbox[4]

        average_score = image_score / (len(infer_result) or 1)
        return average_score < self.threshold
```

In this step, you have customized an **hard_example_mining algorithm** named `Threshold`, and the line of `ClassFactory.register(ClassType.HEM)` is to complete the registration.

**(2). Configure CRD yaml**

After registration, you only need to change the name of the hem and parameters in the yaml file, and then the corresponding class will be automatically called according to the name.

```yaml
deploySpec:
    hardExampleMining:
      name: "Threshold"
      parameters:
        - key: "threshold"
          value: "0.9"
```

## 2.3 Run Customized Example

**(1). Build worker images**

First, you need to modify [lifelong-learning-atcii-classifier.Dockerfile](https://github.com/kubeedge/sedna/blob/c763c1a90e74b4ff1ab0afa06fb976fbb5efa512/examples/lifelong-learning-atcii-classifier.Dockerfile) based on your development.

Then generate Images by the script [build_images.sh](https://github.com/kubeedge/sedna/blob/main/examples/build_image.sh).

**(2). Start customized lifelong job**

This process is similar to the process described in section `1.4`, but remember to modify the dataset (explained in `1.3`) and `CLOUD_IMAGE` to match your development.

## 2.4 Further Development

In addition to developing on the lifelong learning case, users can also [develop the control plane](https://github.com/kubeedge/sedna/blob/main/docs/contributing/control-plane/development.md) of the Sedna project, as well as [adding a new synergy feature](https://github.com/kubeedge/sedna/blob/51219027a0ec915bf3afb266dc5f9a7fb3880074/docs/contributing/control-plane/add-a-new-synergy-feature.md).
