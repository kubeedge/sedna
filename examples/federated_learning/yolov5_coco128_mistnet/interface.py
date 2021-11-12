# Copyright 2021 The KubeEdge Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
from sedna.algorithms.aggregation import MistNet
from sedna.algorithms.client_choose import SimpleClientChoose
from sedna.common.config import Context
from sedna.core.federated_learning import FederatedLearningV2

from examples.nnrt.nnrt_trainer_yolo import Trainer
from examples.nnrt.nnrt_algorithms.mistnet import Algorithm
from examples.nnrt.nnrt_models.acl_inference import Inference
from examples.nnrt.nnrt_datasource_yolo import DataSource
simple_chooser = SimpleClientChoose(per_round=1)

# It has been determined that mistnet is required here.
mistnet = MistNet(cut_layer=Context.get_parameters("cut_layer"),
                  epsilon=Context.get_parameters("epsilon"))

# The function `get_transmitter_from_config()` returns an object instance.
s3_transmitter = FederatedLearningV2.get_transmitter_from_config()


class Dataset:
    def __init__(self) -> None:
        self.parameters = {
            "datasource": "YOLO",
            "data_params": "./coco128.yaml",
            # Where the dataset is located
            "data_path": "./data/COCO",
            "train_path": "./data/COCO/coco128/images/train2017/",
            "test_path": "./data/COCO/coco128/images/train2017/",
            # number of training examples
            "num_train_examples": 128,
            # number of testing examples
            "num_test_examples": 128,
            # number of classes
            "num_classes": 80,
            # image size
            "image_size": 640,
            "download_urls": ["https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128.zip",],
            "classes":
                [
                    "person",
                    "bicycle",
                    "car",
                    "motorcycle",
                    "airplane",
                    "bus",
                    "train",
                    "truck",
                    "boat",
                    "traffic light",
                    "fire hydrant",
                    "stop sign",
                    "parking meter",
                    "bench",
                    "bird",
                    "cat",
                    "dog",
                    "horse",
                    "sheep",
                    "cow",
                    "elephant",
                    "bear",
                    "zebra",
                    "giraffe",
                    "backpack",
                    "umbrella",
                    "handbag",
                    "tie",
                    "suitcase",
                    "frisbee",
                    "skis",
                    "snowboard",
                    "sports ball",
                    "kite",
                    "baseball bat",
                    "baseball glove",
                    "skateboard",
                    "surfboard",
                    "tennis racket",
                    "bottle",
                    "wine glass",
                    "cup",
                    "fork",
                    "knife",
                    "spoon",
                    "bowl",
                    "banana",
                    "apple",
                    "sandwich",
                    "orange",
                    "broccoli",
                    "carrot",
                    "hot dog",
                    "pizza",
                    "donut",
                    "cake",
                    "chair",
                    "couch",
                    "potted plant",
                    "bed",
                    "dining table",
                    "toilet",
                    "tv",
                    "laptop",
                    "mouse",
                    "remote",
                    "keyboard",
                    "cell phone",
                    "microwave",
                    "oven",
                    "toaster",
                    "sink",
                    "refrigerator",
                    "book",
                    "clock",
                    "vase",
                    "scissors",
                    "teddy bear",
                    "hair drier",
                    "toothbrush",
                ],
            "partition_size": 128,
        }

class Estimator:
    def __init__(self) -> None:
        # initialize inference object with deviceID, om path, image height and width
        self.model = Inference(0, "./yolov5x_cutlayer4.om", 640, 640)
        self.pretrained = None
        self.trainer = Trainer(model=self.model)
        self.algorithm = Algorithm(self.trainer)
        self.datasource = DataSource()
        self.saved = None
        self.hyperparameters = {
            "type": "yolov5",
            "rounds": 1,
            "target_accuracy": 0.99,
            "epochs": int(Context.get_parameters("EPOCHS", 500)),
            "batch_size": int(Context.get_parameters("BATCH_SIZE", 16)),
            "optimizer": "SGD",
            "linear_lr": False,
            # The machine learning model
            "model_name": "yolov5",
            "model_config": "./yolov5x.yaml",
            "train_params": "./hyp.scratch.yaml"
        }
