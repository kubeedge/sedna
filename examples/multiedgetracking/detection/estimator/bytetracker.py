import datetime
import os
from typing import List

import torch
import numpy as np
import cv2

from sedna.common.config import Context
from sedna.common.benchmark import FTimer, FluentdHelper
from sedna.common.log import LOGGER

# YOLOX imports
from yolox.yolox.exp import get_exp
from yolox.yolox.utils import pre_process, fuse_model, postprocess

os.environ['BACKEND_TYPE'] = 'TORCH'

confidence_thr = Context.get_parameters('confidence_thr', 0.25)
nms_thr = Context.get_parameters('nms_thr', 0.45)
num_classes = Context.get_parameters('num_classes', 1)
image_size = Context.get_parameters('input_shape')

class ExperimentPath():
    @staticmethod
    def get_experiment_dir() -> str:
        return "exps/"

    @staticmethod
    def get_experiment_file_dict() -> dict:
        return {
        "bytetrack_s_mot17": "yolox_s_mix_det.py",
        "bytetrack_m_mot17": "yolox_m_mix_det.py",
        "bytetrack_l_mot17": "yolox_l_mix_det.py",
        "bytetrack_x_mot17": "yolox_x_mix_det.py",
        "bytetrack_x_mot20": "yolox_x_mix_mot20_ch.py"}

class ByteTracker(FluentdHelper):

    def __init__(self, **kwargs) -> None:
        """
        Initializes the YOLOX pedestrian detection model with the default detection parameters (e.g., minimum detection
        confidence, NMS threshold) to be used during inference.
        """
        super(ByteTracker, self).__init__()

        self.num_classes = num_classes
        self.confidence_thr = float(confidence_thr)
        self.nms_thr = float(nms_thr)
        self.set_to_eval = True
        self.model = None
        self.camera_code = kwargs.get('camera_code', 0)
        self.yolox_type = "bytetrack_s_mot17" # This should come from the model CRD

        self.test_size = int(image_size)
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        self.swap = (2, 0, 1)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def write_to_fluentd(self, data):
        try:
            for elem in data:
                bts = np.sum(list(map(lambda x: np.asarray(x).nbytes, elem[0])))
                
                msg = {
                    "worker": "l2-object-detector",
                    "outbound_data": int(bts),
                    "confidence": elem[1].item() # Transforms single-valued tensor into a number.
                }

                self.send_json_msg(msg)
        except Exception as ex:
            LOGGER.error(f"Error while transmitting data to fluentd. Details: [{ex}]")

    def load(self, model_url: str = "") -> None:
        """
        Load the pre-trained weights.
        """
        # Current experiment settings
        self._exp_path = ExperimentPath()
        self._exp_file = os.path.join(
            self._exp_path.get_experiment_dir(),
            self._exp_path.get_experiment_file_dict()[self.yolox_type]
        )
        self.exp = get_exp(self._exp_file, exp_name="")

        # Initialize YOLOX model
        self.model = self.exp.get_model()

        LOGGER.info(f"Loading checkpoint from {model_url}")
        checkpoint = torch.load(model_url, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model"])
        
        self.model = fuse_model(self.model)

    def evaluate(self, **kwargs):
        LOGGER.debug(f"Evaluating model")
        self.model.eval()

    def predict(self, data, **kwargs):
        """
        Run the pedestrian detector on the input image available at img_path
        :param img_path: path to image.
        :return:
            img_with_bbox List[np.ndarray]
        """

        # Image information
        img_info = {"id": 0}

        LOGGER.debug("Loading image to device")
        img = data.copy()
        height, width = img.shape[:2]

        # Store image information, will be used during post processing
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        # Resize and normalize image
        img, ratio = pre_process(img, [self.test_size, self.test_size], self.rgb_means, self.std, self.swap)
        img_info["ratio"] = ratio

        # Convert numpy image to tensor
        img = torch.from_numpy(img).unsqueeze(0).to(self.device)
        img = img.float()

        # Run Inference
        with torch.no_grad():
            with FTimer("detection"):
                # Image forward pass, output now contains
                # outputs is a [batch_size, num_box, 5 + num_classes] tensor
                # outputs[:, :, 5: 5 + num_classes] contains the confidence score for each class
                outputs = self.model(img)

        # Filter detection based on confidence threshold and nms
        outputs = outputs.cpu()

        with FTimer("nms"):
            outputs = postprocess(
                outputs,
                self.num_classes,
                self.confidence_thr,
                self.nms_thr)

        object_crops = []

        # Prepare image with boxes overlaid
        try:
            bboxes = outputs[0][:, 0:4]
            a0 = outputs[0][:, 4]
            a1 = outputs[0][:, 5]
            bboxes /= img_info["ratio"]

            # Crop to person detections
            # person crops is a list of numpy arrays containing the image cropped to that person only
            det_time = datetime.datetime.now().strftime("%a, %d %B %Y %H:%M:%S")

            for i in range(len(bboxes)):
                box = bboxes[i]
                x0, y0, x1, y1 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                score = a0[i]*a1[i]
                _img = data[y0: y1, x0: x1]

                crop_encoded = np.array(cv2.imencode('.jpg', _img)[1])

                object_crops.append([crop_encoded.tolist(), score, self.camera_code, det_time]) 

            self.write_to_fluentd(object_crops)

            LOGGER.info(f"Found {len(object_crops)} objects/s in camera {self.camera_code}")
        except Exception as ex:
            LOGGER.error(f"No objects identified [{ex}].")

        return object_crops