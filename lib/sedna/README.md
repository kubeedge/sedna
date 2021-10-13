
# Sedna Python SDK

The Sedna Python Software Development Kit (SDK) aims to provide developers with a convenient yet flexible tool to write the Sedna applications.

This document introduces how to obtain and call Sedna Python SDK.


## Introduction

Expose the Edge AI features to applications, i.e. training or inference programs.

## Requirements and Installation
The build process is tested with Python 3.6, Ubuntu 18.04.5 LTS

```bash
# Clone the repo
git clone --recursive https://github.com/kubeedge/sedna.git
cd sedna/lib

# Build the pip package
python setup.py bdist_wheel

# Install the pip package 
pip install dist/sedna*.whl

```

Install via Setuptools

```bash
# Install dependence
pip install -r requirements.txt

# Install sedna
python setup.py install --user
```

## Use Python SDK

0. (optional) Check `Sedna` version
    ```bash
    $ python -c "import sedna; print(sedna.__version__)"
    ```

1. Import the required modules as follows:

    ```python
   from sedna.core.joint_inference import JointInference, BigModelService 
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


3. Initialize a Incremental Learning Job:

    ```python
   
    # get hard exmaple mining algorithm from config
    hard_example_mining = IncrementalLearning.get_hem_algorithm_from_config(
        threshold_img=0.9
    )
   
    # create Incremental Learning infernece instance
    il_job = IncrementalLearning(
        estimator=Estimator,
        hard_example_mining=hard_example_mining
    )
   
    ```

	where:
   
	- `IncrementalLearning` is the Cloud-edge job you want to access.
	- `Estimator` is the base model for your ML job.
	- `hard_example_mining` is the parameters of incremental learning job.
    
    Inference
    ---------
    
	> **Note:** The `job parameters` of each feature are different.

4. Running Job - training / inference / evaluation.

	```python
	results, final_res, is_hard_example = il_job.inference(
            img_rgb, 
            post_process=deal_infer_rsl, 
            input_shape=input_shape
    )
 
	```
 	
	where:

	- `img_rgb` is the sample used to inference
    - `deal_infer_rsl` is a function used to process result after model predict
    - `input_shape` is the parameters of `Estimator` in inference
    - `results` is the result predicted by model
    - `final_res` is the result after process by `deal_infer_rsl`
    - `is_hard_example` tells if the sample is hard sample or not

## Customize algorithm

Sedna provides a class called `class_factory.py` in `common` package, in which only a few lines of changes are required to become a module of sedna.

Two classes are defined in `class_factory.py`, namely `ClassType` and `ClassFactory`.

`ClassFactory` can register the modules you want to reuse through decorators. For example, in the following code example, you have customized an **hard_example_mining algorithm**, you only need to add a line of `ClassFactory.register(ClassType.HEM)` to complete the registration.

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

After registration, you only need to change the name of the hem and parameters in the yaml file, and then the corresponding class will be automatically called according to the name.

```yaml
deploySpec:
    hardExampleMining:
      name: "Threshold"
      parameters:
        - key: "threshold"
          value: "0.9"
```
