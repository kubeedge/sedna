
# Sedna Python SDK

The Sedna Python Software Development Kit (SDK) aims to provide developers with a convenient yet flexible tool to write the Sedna applications.

This document introduces how to obtain and call Sedna Python SDK.


## Introduction

Expose the Edge AI features to applications, i.e. training or inference programs.

## Requirements and Installation
The build process is tested with Python 3.6, Ubuntu 18.04.5 LTS

```bash
# Clone the repo
git clone --recursive git@github.com:kubeedge/sedna.git
cd sedna/lib

# Build the pip package
python setup.py bdist_wheel

# Install the pip package 
pip install dist/sedna*.whl

```

Install via Setuptools

```bash
python setup.py install --user
```

## Use Python SDK

1. Import the required modules as follows:

    ```python
   from sedna.core.joint_inference import JointInference, TSBigModelService 
   from sedna.core.federated_learning import FederatedLearning
   from sedna.core.incremental_learning import IncrementalLearning
   from sedna.core.lifelong_learning import LifelongLearning

    ```

2. Define an `Estimator`:

	```python
	
    import os
    
    # Keras
    import keras
    from keras.layers import Dense, MaxPooling2D, Conv2D, Flatten, Dropout
    from keras.models import Sequential
    
    os.environ['BACKEND_TYPE'] = 'KERAS'
    
    def KerasEstimator():
        model = Sequential()
        model.add(Conv2D(64, kernel_size=(3, 3),
                         activation="relu", strides=(2, 2),
                         input_shape=(128, 128, 3)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(32, kernel_size=(3, 3), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dropout(0.25))
        model.add(Dense(64, activation="relu"))
        model.add(Dense(32, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(2, activation="softmax"))
        
        model.compile(loss="categorical_crossentropy",
                      optimizer="adam",
                      metrics=["accuracy"])
        loss = keras.losses.CategoricalCrossentropy(from_logits=True)
        metrics = [keras.metrics.categorical_accuracy]
        optimizer = keras.optimizers.Adam(learning_rate=0.1)
        model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        return model    
     ```

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
    > **Notes**:  Estimator is a high-level API that greatly simplifies machine learning programming. Estimators encapsulate training, evaluation, prediction, and exporting for your model. 


3. Initialize a Job:

    ```python
   
    ll_job = LifelongLearning(
        estimator=Estimator,
        task_definition="TaskDefinitionByDataAttr",
    )
    ```

	where:

	- `LifelongLearning` is the Cloud-edge job you want to access.
	- `Estimator` is the base model for your ML job.
	- `TaskDefinitionByDataAttr` is the parameters of lifelong learning job.

	> **Note:** The `job parameters` of each feature are different.

4. Running Job - training / inference / evaluation.

	```python
 
	train_instance = ll_job.train(
        train_data=train_data,
        metric_name=metric_name,
        early_stopping_rounds=early_stopping_rounds
    )
 
	```
 	
	where:

	- `ll_job` is the Sedna Job pre-initialize.
	- `train_data` is the dataset use to train.
	- `metric_name`/`early_stopping_rounds` is the parameters for base model training.


## Code Structure

```StyleText
.
|-- algorithms                  # All algorithms build in the sedna framework
|   |-- __init__.py
|   |-- aggregation             # Aggregation algorithms for federated learning
|   |   `-- __init__.py
|   |-- hard_example_mining     # Hard example mining algorithms for incremental learning and joint_inferencejoint inference
|   |   `-- __init__.py
|   |-- multi_task_learning     # Multitask transfer learning algorithms
|   |   |-- __init__.py
|   |   |-- multi_task_learning.py
|   |   `-- task_jobs			# Components in MTL Pipelines
|   |       |-- __init__.py
|   |       |-- artifact.py     # Artifacts instance used in componetnts
|   |       |-- inference_integrate.py  # Integrating algorithm for the output geted by multitask inference
|   |       |-- task_definition.py      # Multitask definition base on given traning samples
|   |       |-- task_mining.py  # Mining target tasks of inference samples
|   |       |-- task_relation_discover.py  # Discover the relation of tasks which generated from task_definition
|   |       `-- task_remodeling.py  # Remodeling tasks
|   `-- unseen_task_detect      # Unseen task detect algorithms for lifelong learning
|       `-- __init__.py
|-- backend                     # Encapsulated the general ML frameworks, decoupled the core algorithms of sedna from the framework
|   |-- __init__.py
|   |-- base.py
|   `-- tensorflow
|       `-- __init__.py
|-- common                     # Contains the common methods, configurations and class factory
|   |-- __init__.py
|   |-- class_factory.py       # It defines two importantt classes, ClassType and ClassFactory. You can register the modules which want to be reuse through the decorator.
|   |-- config.py              # Globally used configuration information
|   |-- constant.py            # 
|   |-- file_ops.py            # File system processing 
|   |-- log.py                 # Global log definition 
|   `-- utils.py               # Common funtions
|-- core                       # All features build in the sedna framework
|   |-- __init__.py
|   |-- base.py                # Abstract Base class of the features
|   |-- federated_learning     # Federated learning
|   |   |-- __init__.py
|   |   `-- federated_learning.py
|   |-- incremental_learning   # Incremental learning
|   |   |-- __init__.py
|   |   `-- incremental_learning.py
|   |-- joint_inference        # Joint inference
|   |   |-- __init__.py
|   |   `-- joint_inference.py
|   `-- lifelong_learning      # Lifelong learning
|       |-- __init__.py
|       `-- lifelong_learning.py
|-- datasources                # Abastract Base class of the dataset (train/eval/test)
|   `-- __init__.py
`-- service                    # Communication module, include clinet/server
    |-- __init__.py
    |-- client.py              # Send http/ws requests to the server 
    |-- run_kb.py              # Run Knowledgebase Service as rest api
    `-- server                 # Servers for each feature
        |-- __init__.py 
        |-- aggregation.py     # Aggregator service for federated learning
        |-- base.py            # Base module
        |-- inference.py       # Inference Service for Joint inference
        `-- knowledgeBase      # Knoledgebase Service
            |-- __init__.py
            |-- database.py
            |-- model.py       # ORM
            `-- server.py
```

