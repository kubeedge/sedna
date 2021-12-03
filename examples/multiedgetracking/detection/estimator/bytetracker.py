import datetime
import os
import torch
import numpy as np
import cv2
import pickle

from sedna.core.multi_edge_tracking.data_classes import DetTrackResult
from sedna.common.config import Context
from sedna.common.benchmark import FTimer, FluentdHelper
from sedna.common.log import LOGGER

# YOLOX imports
from yolox.yolox.exp import get_exp
from yolox.yolox.utils import pre_process, fuse_model, postprocess

# Tracking imports
from yolox.yolox.tracker.byte_tracker import BYTETracker

os.environ['BACKEND_TYPE'] = 'TORCH'

# Tracking parameters
frame_rate = int(Context.get_parameters('frame_rate', 30))
track_thresh = float(Context.get_parameters('track_thresh', 0.5))
track_buffer = int(Context.get_parameters('track_buffer', 600))
match_thresh = float(Context.get_parameters('match_thresh', 0.8))
min_box_area = int(Context.get_parameters('min_box_area', 500))

# Detection parameters
confidence_thr = Context.get_parameters('confidence_thr', 0.25)
nms_thr = Context.get_parameters('nms_thr', 0.45)
num_classes = Context.get_parameters('num_classes', 1)
image_size = Context.get_parameters('input_shape')

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

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

        self.original_size = None

        # Tracking parameters
        self.tracker_args = Namespace(
            track_thresh=track_thresh,
            mot20=False,
            track_buffer=track_buffer,
            match_thresh=match_thresh,
            min_box_area=min_box_area,
            frame_rate=frame_rate
        )
        self.tracker = BYTETracker(args=self.tracker_args, frame_rate=frame_rate)

    def write_to_fluentd(self, result : DetTrackResult):
        try:
            msg = {
                    "worker": "l2-object-detector",
                    "outbound_data": len(pickle.dumps(result)),
                    "confidence": np.median(result.confidence).item()
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

    def forward(self, data):
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

        return outputs, img_info

    def detect_only(self, data, outputs, img_info):
        object_crops = []
        result = None

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

                object_crops.append([crop_encoded.tolist(), score.item(), self.camera_code, det_time]) 
            
            if len(object_crops) > 0:
                scene = np.array(cv2.imencode('.jpg', data)[1])
                result = DetTrackResult([item[0] for item in object_crops], scene, [item[1] for item in object_crops], [item[3] for item in object_crops], [item[2] for item in object_crops])
                self.write_to_fluentd(object_crops)
                LOGGER.info(f"Found {len(object_crops)} objects/s in camera {self.camera_code}")
            else:
                return None

        except Exception as ex:
            LOGGER.error(f"No objects identified [{ex}].")
            return None

        return result


    def track(self, data, outputs, img_info):
        # initialize placeholders for the tracking data
        online_tlwhs = []
        online_ids = []
        online_scores = []

        # update tracker
        with FTimer("tracking"):
            online_targets = self.tracker.update(outputs[0], self.original_size, (image_size, image_size))

        # if no detections after tracker update
        # if online_targets is None or (online_targets[0] is None):
        #     return None

        for t in online_targets:
            # bounding box and tracking id
            # tlwh - top left width height
            tlwh = t.tlwh
            tid = t.track_id

            # prior about human aspect ratio
            f_vertical = tlwh[2] / tlwh[3] > 1.6

            if tlwh[2] * tlwh[3] > self.tracker_args.min_box_area and not f_vertical:
                online_tlwhs.append(tuple(map(int, tlwh)))
                online_ids.append(int(tid))
                online_scores.append(t.score)

        online_bboxes = [None] * len(online_tlwhs)
        for i, t in enumerate(online_tlwhs):
            x1, y1, w, h = t
            x2 = x1 + w
            y2 = y1 + h
            online_bboxes[i] = [x1, y1, x2, y2]

        # prepare data for transmission
        object_crops = []
        result = None

        det_time = datetime.datetime.now().strftime("%a, %d %B %Y %H:%M:%S")

        for i in range(len(online_bboxes)):
            # generate the object crop
            box = online_bboxes[i]
            x0, y0, x1, y1 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            _img = data[y0: y1, x0: x1]

            # encode the object crop
            crop_encoded = np.array(cv2.imencode('.jpg', _img)[1])

            # append
            object_crops.append([
                crop_encoded,
                box,
                online_scores[i],
                online_ids[i],
                self.camera_code,
                det_time
            ])
        
        if len(object_crops) > 0:
            scene = np.array(cv2.imencode('.jpg', data)[1])
            result = DetTrackResult(
                bbox=[item[0] for item in object_crops],
                scene=scene,
                confidence=[item[2] for item in object_crops],
                detection_time=[item[5] for item in object_crops],
                camera=[item[4] for item in object_crops],
                bbox_coord=[item[1] for item in object_crops],
                tracking_ids=[item[3] for item in object_crops]
            )
            
            self.write_to_fluentd(result)
            
            LOGGER.info(f"Tracked {len(object_crops)} objects/s in camera {self.camera_code} with IDs {result.tracklets}")
        else:
            return None

        return result


    def predict(self, data, **kwargs):
        """
        Run the pedestrian detector/tracker on the input image available at img_path
        :param img_path: path to image.
        :return:
            img_with_bbox List[np.ndarray]
        """
        
        self.original_size = (data.shape[0], data.shape[1])

        # get detections from the image forward pass
        outputs, img_info = self.forward(data)

        if outputs is None or (outputs[0] is None):
            return None

        if kwargs['op_mode'] == "detection":
            return self.detect_only(data, outputs, img_info)
        else:
            return self.track(data, outputs, img_info)
     