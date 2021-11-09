import os
from typing import List

import torch
import numpy as np
import cv2

from sedna.common.config import Context
from sedna.common.benchmark import FTimer, FluentdHelper
from sedna.common.log import LOGGER
from utils.utils import *

confidence_thr = Context.get_parameters('confidence_thr', 0.25)
nms_thr = Context.get_parameters('nms_thr', 0.45)
num_classes = Context.get_parameters('num_classes', 1)
image_size = Context.get_parameters('input_shape') # in pixels!

class Yolox(FluentdHelper):

    def __init__(self, **kwargs) -> None:
        """
        Initializes the YOLOX pedestrian detection model with the default detection parameters (e.g., minimum detection
        confidence, NMS threshold) to be used during inference.
        """
        super(Yolox, self).__init__()

        self.num_classes = num_classes
        self.confidence_thr = confidence_thr
        self.nms_thr = nms_thr
        self.set_to_eval = True
        self.camera_code = kwargs.get('camera_code', 0)

        self.test_size = image_size
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

        self.device = "cuda" if torch.cuda.is_available() else "cpu",

        # Initialize YOLOX model
        self.model = self.exp.get_model()

    def write_to_fluentd(self, data: List[np.ndarray]) -> None:
        try:
            for elem in data:
                bts = np.asarray(elem).nbytes
                msg = {
                    "worker": "l2-object-detector",
                    "outbound_data": int(bts)
                }
                self.send_json_msg(msg)

        except Exception as ex:
            LOGGER.error(f"Error while transmitting data to fluentd. Details: [{ex}]")

    def load(self, model_url: str = "") -> None:
        """
        Load the pre-trained weights.
        """
        LOGGER.info(f"Loading checkpoint from {model_url}")
        checkpoint = torch.load(model_url, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])

        if self.set_to_eval:
            self.model.eval()

        self.model = fuse_model(self.model)

    def predict(self,
                img_path: str
                ) -> List[np.ndarray]:
        """
        Run the pedestrian detector on the input image available at img_path
        :param img_path: path to image.
        :return:
            img_with_bbox List[np.ndarray]
        """

        # Image information
        img_info = {"id": 0, "file_name": os.path.basename(img_path)}

        LOGGER.debug("Loading image to device")
        img = cv2.imread(img_path)
        height, width = img.shape[:2]

        # Store image information, will be used during post processing
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        # Resize and normalize image
        img, ratio = preproc(img, self.test_size, self.rgb_means, self.std)
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

        # Prepare image with boxes overlaid
        bboxes = outputs[:, 0:4]
        bboxes /= img_info["ratio"]

        cls = outputs[:, 6]
        scores = outputs[:, 4] * outputs[:, 5]

        # Img with bboxes overlaid
        # Used for visualization and check correctness
        img_with_bbox = vis(
            img_info["raw_img"].copy(),
            bboxes,
            scores,
            cls,
            conf=self.confidence_thr,
            class_names=["person"])

        # Crop to person detections
        # person crops is a list of numpy arrays containing the image cropped to that person only
        person_crops = []
        for i in range(len(bboxes)):
            box = bboxes[i]
            x0, y0, x1, y1 = int(box[0]), int(box[1]), int(box[2]), int(box[3])

            _img = img[y0: y1, x0: x1]
            person_crops.append(_img)

        self.write_to_fluentd(person_crops)

        LOGGER.info(f"Found {len(person_crops)} person/s in camera {self.camera_code}")

        return person_crops