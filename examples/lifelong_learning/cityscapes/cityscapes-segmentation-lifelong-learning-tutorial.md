This tutorial targets at lifelong learning job in smart environment perception scenario, and includes how to run the default example with customized configurations, as well as how to develop and integrate user-defined modules.

# 1 Configure Default Example
With Custom Resource Definitions (CRDs) of Kubernetes, developers are able to configure the default lifelong process using the following configurations.
## 1.1 Install Sedna
Follow the [Sedna installation document](https://sedna.readthedocs.io/en/v0.5.0/setup/install.html) to install Sedna.

## 1.2 Prepare Dataset
Users can use semantic segmentation datasets from [CITYSCAPES](https://www.cityscapes-dataset.com/). While we also provide a re-organized [dataset segmentation_data.zip](https://kubeedge.obs.cn-north-1.myhuaweicloud.com/examples/robo_dog_delivery/segmentation_data.zip) of CITYSCAPES as an example for training and evaluation. 

Download and unzip segmentation_data.zip by executing the following commands. 
```
mkdir /data
cd /data
wget https://kubeedge.obs.cn-north-1.myhuaweicloud.com/examples/robo_dog_delivery/segmentation_data.zip
unzip segmentation_data.zip
```

## 1.3 Create Dataset CRD
| Property | Required | Description |
|----------|----------|-------------|
|name|yes|Dataset name defined in metadata|
|url|yes|Url of dataset index file, which is generally stored in data node|
|format|yes|Format of dataset index file|
|nodeName|yes|Name of data node that stores data and dataset index file|

After preparing specific dataset and index file, users can configure them as in the following example for training and evaluation. So data will be automatically downloaded from where the index file indicates to the corresponding pods.
```
DATA_NODE = "cloud-node" 
```

```
kubectl create -f - << EOF
apiVersion: sedna.io/v1alpha1
kind: Dataset
metadata:
  name: lifelong-robo-dataset
spec:
  url: "$data_url"
  format: "txt"
  nodeName: "$DATA_NODE"
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

First, configure parameters for lifelong learning job as follows.

```
local_prefix=/data
cloud_image=docker.io/luosiqi/sedna-robo:v0.1.2
edge_image=docker.io/luosiqi/sedna-robo:v0.1.2
data_url=$local_prefix/segmentation_data/data.txt

WORKER_NODE=sedna-mini-control-plane

DATA_NODE=$WORKER_NODE 
TRAIN_NODE=$WORKER_NODE 
EVAL_NODE=$WORKER_NODE 
INFER_NODE=$WORKER_NODE 
OUTPUT=$local_prefix/lifelonglearningjob/output
job_name=robo-demo
```

Second, use the following yaml configuration to create and run lifelong learning job.

```
kubectl create -f - <<EOF
apiVersion: sedna.io/v1alpha1
kind: LifelongLearningJob
metadata:
  name: $job_name
spec:
  dataset:
    name: "lifelong-robo-dataset"
    trainProb: 0.8
  trainSpec:
    template:
      spec:
        nodeName: $TRAIN_NODE
        dnsPolicy: ClusterFirstWithHostNet
        containers:
          - image: $cloud_image
            name:  train-worker
            imagePullPolicy: IfNotPresent       
            args: ["train.py"]
            env:
              - name: "num_class"
                value: "24"
              - name: "epoches"
                value: "1"
              - name: "attribute"
                value: "real, sim"
              - name: "city"
                value: "berlin"
              - name: "BACKEND_TYPE"
                value: "PYTORCH"
            resources:
              limits:
                cpu: 6
                memory: 12Gi
              requests:
                cpu: 4
                memory: 12Gi
            volumeMounts:
            - mountPath: /dev/shm
              name: cache-volume
        volumes:
        - emptyDir:
            medium: Memory
            sizeLimit: 256Mi
          name: cache-volume
    trigger:
      checkPeriodSeconds: 30
      timer:
        start: 00:00
        end: 24:00
      condition:
        operator: ">"
        threshold: 100
        metric: num_of_samples
  evalSpec:
    template:
      spec:
        nodeName: $EVAL_NODE
        dnsPolicy: ClusterFirstWithHostNet
        containers:
          - image: $cloud_image
            name:  eval-worker
            imagePullPolicy: IfNotPresent
            args: ["evaluate.py"]
            env:
              - name: "operator"
                value: "<"
              - name: "model_threshold"
                value: "0"
              - name: "num_class"
                value: "24"
              - name: "BACKEND_TYPE"
                value: "PYTORCH"
            resources:
              limits:
                cpu: 6
                memory: 12Gi
              requests:
                cpu: 4
                memory: 12Gi
  deploySpec:
    template:
      spec:
        nodeName: $INFER_NODE
        dnsPolicy: ClusterFirstWithHostNet
        hostNetwork: true
        containers:
        - image: $edge_image
          name:  infer-worker
          imagePullPolicy: IfNotPresent
          args: ["predict.py"]
          env:
            - name: "test_data"
              value: "/data/test_data"
            - name: "num_class"
              value: "24"
            - name: "unseen_save_url"
              value: "/data/unseen_samples"
            - name: "INFERENCE_RESULT_DIR"
              value: "/data/infer_results"
            - name: "BACKEND_TYPE"
              value: "PYTORCH"
          volumeMounts:
          - name: unseenurl
            mountPath: /data/unseen_samples
          - name: inferdata
            mountPath: /data/infer_results
          - name: testdata
            mountPath: /data/test_data
          resources:
            limits:
              cpu: 6
              memory: 12Gi
            requests:
              cpu: 4
              memory: 12Gi
        volumes:
          - name: unseenurl
            hostPath:
              path: /data/unseen_samples
              type: DirectoryOrCreate
          - name: inferdata
            hostPath:
              path: /data/infer_results
              type: DirectoryOrCreate
          - name: testdata
            hostPath:
              path: /data/test_data
              type: DirectoryOrCreate
  outputDir: $OUTPUT/$job_name
EOF
```

## 1.5 Check Lifelong Learning Job
**(1). Query lifelong learning service status**

```
kubectl get lifelonglearningjob robo-demo
```

**(2). View pods related to lifelong learning job**

```
kubectl get pod
```

**(3). View result files**

Knowledgebase contents including multi-task learning models, dataset, etc., can be found in `outputDir`. While inference results are stored in `/data/infer_results` of `$INFER_NODE`.

# 2 Develop and Integrate Customized Modules

Before starting development, you should prepare the [development environment](https://github.com/kubeedge/sedna/blob/main/docs/contributing/prepare-environment.md) and learn about the [interface design of Sedna](https://sedna.readthedocs.io/en/latest/autoapi/lib/sedna/index.html).

## 2.1 Develop Sedna AI Module

The Sedna framework components are decoupled and the registration mechanism is used to combine functional components to facilitate function and algorithm expansion. For details about the Sedna architecture and main mechanisms, see [Lib README](https://github.com/kubeedge/sedna/blob/51219027a0ec915bf3afb266dc5f9a7fb3880074/lib/sedna/README.md).

The following contents explains how to develop customized AI modules of a Sedna project, including **dataset**, **base model**, **algorithms**, etc.

### 2.1.1 Import Service Datasets

During Sedna application development, the first problem users encounter is how to import service datasets to Sedna. Sedna provides interfaces and public methods related to data conversion and sampling in the [Dataset class](https://github.com/kubeedge/sedna/blob/c763c1a90e74b4ff1ab0afa06fb976fbb5efa512/lib/sedna/datasources/__init__.py). 

All dataset classes of Sedna are inherited from the base class `sedna.datasources.BaseDataSource`. This base class defines the interfaces and attributes to process datasets customizedly and provides default implementation. The derived class can reload these default implementations as required.

We take `txt format` dataset index file which contains sets of images as an example.

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
        x_data = []
        y_data = []
        use_raw = kwargs.get("use_raw")
        for f in args:
            if not (f and FileOps.exists(f)):
                continue
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

Then users can load and preprocess the dataset url file utilizing `sedna.datasources.TxtDataParse` as follows. Particularly, the attribute `func` of `TxtDataParse` defines the customized preprocessing function for each row's data index in dataset url file.

```python
def _load_txt_dataset(dataset_url):
    # use original dataset url
    original_dataset_url = Context.get_parameters('original_dataset_url', "")
    dataset_urls = dataset_url.split()
    dataset_urls = [
        os.path.join(
            os.path.dirname(original_dataset_url),
            dataset_url) for dataset_url in dataset_urls]
    return dataset_urls[:-1], dataset_urls[-1]

def run():
    estimator = Estimator(num_class=int(Context.get_parameters("num_class", 24)),
                      epochs=int(Context.get_parameters("epoches", 1)))
    train_dataset_url = BaseConfig.train_dataset_url
    train_data = TxtDataParse(data_type="train", func=_load_txt_dataset)
    train_data.parse(train_dataset_url, use_raw=False)

    train(estimator, train_data)


if __name__ == '__main__':
    run()
```

### 2.1.2 Modify Base Model

Estimator is a high-level API that greatly simplifies machine learning programming. Estimators encapsulate `train`, `evaluate`, `predict`, `load` and `save` functions which users should customizedly realize.

**(1). Define an Estimator**

In lifelong learning robotics case, Estimator is defined in interface.py, and users can replace the existing base model with the models that best suits their purposes.

 ```python
class Estimator:
    def __init__(self, **kwargs):
        self.train_args = TrainingArguments(**kwargs)
        self.val_args = EvaluationArguments(**kwargs)

        self.train_args.resume = Context.get_parameters(
            "PRETRAINED_MODEL_URL", None)
        self.trainer = None
        self.train_model_url = None

        label_save_dir = Context.get_parameters(
            "INFERENCE_RESULT_DIR",
            os.path.join(BaseConfig.data_path_prefix,
                         "inference_results"))
        self.val_args.color_label_save_path = os.path.join(
            label_save_dir, "color")
        self.val_args.merge_label_save_path = os.path.join(
            label_save_dir, "merge")
        self.val_args.label_save_path = os.path.join(label_save_dir, "label")
        self.val_args.weight_path = kwargs.get("weight_path")
        self.validator = Validator(self.val_args)

    def train(self, train_data, valid_data=None, **kwargs):
        self.trainer = Trainer(
            self.train_args, train_data=train_data, valid_data=valid_data)
        LOGGER.info("Total epoches: {}".format(self.trainer.args.epochs))
        for epoch in range(
                self.trainer.args.start_epoch,
                self.trainer.args.epochs):
            if epoch == 0 and self.trainer.val_loader:
                self.trainer.validation(epoch)
            self.trainer.training(epoch)

            if self.trainer.args.no_val and \
                (epoch % self.trainer.args.eval_interval ==
                    (self.trainer.args.eval_interval - 1) or
                 epoch == self.trainer.args.epochs - 1):
                # save checkpoint when it meets eval_interval
                # or the training finishes
                is_best = False
                train_model_url = self.trainer.saver.save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': self.trainer.model.state_dict(),
                    'optimizer': self.trainer.optimizer.state_dict(),
                    'best_pred': self.trainer.best_pred,
                }, is_best)

        self.trainer.writer.close()
        self.train_model_url = train_model_url

        return {"mIoU": 0 if not valid_data
                else self.trainer.validation(epoch)}

    def predict(self, data, **kwargs):
        if isinstance(data[0], dict):
            data = preprocess_frames(data)

        if isinstance(data[0], np.ndarray):
            data = preprocess_url(data)

        self.validator.test_loader = DataLoader(
            data,
            batch_size=self.val_args.test_batch_size,
            shuffle=False,
            pin_memory=False)

        return self.validator.validate()

    def evaluate(self, data, **kwargs):
        predictions = self.predict(data.x)
        return accuracy(data.y, predictions)

    def load(self, model_url, **kwargs):
        if model_url:
            self.validator.new_state_dict = torch.load(model_url)
            self.validator.model = load_my_state_dict(
                self.validator.model,
                self.validator.new_state_dict['state_dict'])

            self.train_args.resume = model_url
        else:
            raise Exception("model url does not exist.")

    def save(self, model_path=None):
        if not model_path:
            LOGGER.warning(f"Not specify model path.")
            return self.train_model_url

        return FileOps.upload(self.train_model_url, model_path)
```

**(2). Initialize a lifelong learning job**

```python
import Estimator from interface


ll_job = LifelongLearning(
    estimator=Estimator
)
```

Noted that `Estimator` is the base model for your lifelong learning job.

### 2.1.3 Develop Customized Algorithms

Users may need to develop new algorithms based on the basic classes provided by Sedna, such as `unseen task detection` in lifelong learning example.

Sedna provides a class called `class_factory.py` in `common` package, in which only a few lines of changes are required to integrate existing algorithms into Sedna.

The following content takes a hard example mining algorithm as an example to explain how to add an HEM algorithm to the Sedna hard example mining algorithm library.

**(1). Start from the `class_factory.py`**

First, let's start from the `class_factory.py`. Two classes are defined in `class_factory.py`, namely `ClassType` and `ClassFactory`.

`ClassFactory` can register the modules you want to reuse through decorators. For the new `ClassType.STP` algorithm of task definition in lifelong learning, the code is as follows:

```python
@ClassFactory.register(ClassType.STP)
class TaskDefinitionByOrigin(BaseTaskDefinition):
    """
    Dividing datasets based on the their origins.
    Parameters
    ----------
    attr_filed Tuple[Metadata]
        metadata is usually a class feature label with a finite values.
    """

    def __init__(self, **kwargs):
        super(TaskDefinitionByOrigin, self).__init__()
        self.attribute = kwargs.get("attribute").split(", ")
        self.city = kwargs.get("city")

    def __call__(self,
                 samples: BaseDataSource, **kwargs) -> Tuple[List[Task],
                                                             Any,
                                                             BaseDataSource]:

        tasks = []
        d_type = samples.data_type

        task_index = dict(zip(self.attribute, range(len(self.attribute))))
        sample_index = range(samples.num_examples())

        _idx = [i for i in sample_index if self.city in samples.y[i]]
        _y = samples.y[_idx]
        _x = samples.x[_idx]
        _sample = BaseDataSource(data_type=d_type)
        _sample.x, _sample.y = _x, _y

        g_attr = f"{self.attribute[0]}.model"
        task_obj = Task(entry=g_attr, samples=_sample,
                        meta_attr=self.attribute[0])
        tasks.append(task_obj)

        _idx = list(set(sample_index) - set(_idx))
        _y = samples.y[_idx]
        _x = samples.x[_idx]
        _sample = BaseDataSource(data_type=d_type)
        _sample.x, _sample.y = _x, _y

        g_attr = f"{self.attribute[-1]}.model"
        task_obj = Task(entry=g_attr, samples=_sample,
                        meta_attr=self.attribute[-1])
        tasks.append(task_obj)

        return tasks, task_index, samples
```

In this step, you have customized an **task definition algorithm**, and the line of `ClassFactory.register(ClassType.STP)` is to complete the registration.

**(2). Configure algorithm module in Sedna**

After registration, you only need to configure task definition algorithim in corresponding script. Take the following codes in `train.py` as an example.

```python
def train(estimator, train_data):
    task_definition = {
        "method": "TaskDefinitionByOrigin",
        "param": {
            "attribute": Context.get_parameters("attribute"),
            "city": Context.get_parameters("city")
        }
    }

    task_allocation = {
        "method": "TaskAllocationByOrigin"
    }

    ll_job = LifelongLearning(estimator,
                              task_definition=task_definition,
                              task_relationship_discovery=None,
                              task_allocation=task_allocation,
                              task_remodeling=None,
                              inference_integrate=None,
                              task_update_decision=None,
                              unseen_task_allocation=None,
                              unseen_sample_recognition=None,
                              unseen_sample_re_recognition=None
                              )

    ll_job.train(train_data)
```
Users can configure task definition algorithm and its parameters by dictionary and then pass the dictionary to LifelongLearning class when creating lifelong learning job.


## 2.2 Run Customized Example

**(1). Build worker images**

First, you need to modify lifelong-learning-cityscapes-segmentation.Dockerfile based on your development.

Then generate Images by the script [build_images.sh](https://github.com/kubeedge/sedna/blob/main/examples/build_image.sh).

**(2). Start customized lifelong job**

This process is similar to that in section `1.4`. But remember to modify the dataset (explained in `1.3`) and configure the base model and parameters in yaml like section `1.4`.

## 2.3 Further Development

In addition to developing on the lifelong learning case, users can also [develop the control plane](https://github.com/kubeedge/sedna/blob/main/docs/contributing/control-plane/development.md) of the Sedna project, as well as [adding a new synergy feature](https://github.com/kubeedge/sedna/blob/51219027a0ec915bf3afb266dc5f9a7fb3880074/docs/contributing/control-plane/add-a-new-synergy-feature.md).


