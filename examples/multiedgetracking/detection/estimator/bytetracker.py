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

        if torch.cuda.is_available():
            self.device = "cuda"
            LOGGER.info("Using GPU")
        else:
            self.device = "cpu"
            LOGGER.info("Using CPU")

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
        self.model.to(self.device)

    def evaluate(self, **kwargs):
        LOGGER.debug(f"Evaluating model")
        self.model.eval()

    def forward_batched(self, data):
        """ Data is of List images type"""
        
        img_info = {}

        # Image information
        t_ = data[0].copy()
        height, width = t_.shape[:2]

        # Store image information, will be used during post processing
        img_info["height"] = height
        img_info["width"] = width
        
        LOGGER.debug("Batching images")
        num_imgs = len(data)

        # Image info now stores the raw images
        img_info["raw_img"] = []   
        
        # Input batched
        input_batched = torch.zeros(num_imgs, 3, self.test_size, self.test_size)
        
        # Iterate over all images and concat
        with FTimer("pre_process"):
            for i, img in enumerate(data):
                # img_info["raw_image"].append(img)
                imgp, ratio = pre_process(img, [self.test_size, self.test_size], self.rgb_means, self.std, self.swap)
                
                if i == 0:
                    img_info["ratio"] = ratio
                    
                # Convert numpy image to tensor
                input_batched[i, :, :, :] = torch.from_numpy(imgp).unsqueeze(0).to(self.device)
            
        if self.device == "cpu":
            input_batched = input_batched.float()
                
        input_batched = input_batched.to(self.device)
        
        # Forward pass 
        with torch.no_grad():
            with FTimer("detection"):
                # Image forward pass, output now contains
                # outputs is a [batch_size, num_box, 5 + num_classes] tensor
                # outputs[:, :, 5: 5 + num_classes] contains the confidence score for each class
                
                # in batching mode, outputs will be of shape [b, num_boxes, 5 + num_classes] instead of [1, num_boxes, 5 + num_classes]                
                outputs = self.model(input_batched)

                # Filter detection based on confidence threshold and nms
                outputs = outputs.cpu()

        with FTimer("nms"):
            outputs = postprocess(
                    outputs,
                    self.num_classes,
                    self.confidence_thr,
                    self.nms_thr)
            
        return outputs, img_info

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
        with FTimer("pre_process"):
            img, ratio = pre_process(img, [self.test_size, self.test_size], self.rgb_means, self.std, self.swap)
        img_info["ratio"] = ratio

        # Convert numpy image to tensor
        img = torch.from_numpy(img).unsqueeze(0).to(self.device)
        if self.device == "cpu":
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

    def detection(self, data, output, img_info, det_time):
        object_crops = []
        result = None

        # Prepare image with boxes overlaid
        if output is not None:
            try:
                bboxes = output[:, 0:4]
                a0 = output[:, 4]
                a1 = output[:, 5]
                bboxes /= img_info["ratio"]

                # Crop to person detections
                # person crops is a list of numpy arrays containing the image cropped to that person only

                for i in range(len(bboxes)):
                    box = bboxes[i]
                    x0, y0, x1, y1 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                    score = a0[i]*a1[i]
                    _img = data[y0: y1, x0: x1]

                    crop_encoded = cv2.imencode('.jpg', _img)[1]

                    object_crops.append([
                        crop_encoded,
                        box,
                        score.item(),
                        self.camera_code,
                        det_time
                    ])

                if len(object_crops) > 0:
                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
                    scene = cv2.imencode('.jpg', data, encode_param)[1]

                    result = DetTrackResult(
                        bbox=[item[0] for item in object_crops],
                        scene=scene,
                        confidence=[item[2] for item in object_crops],
                        detection_time=det_time,
                        camera=self.camera_code,
                        bbox_coord=[item[1] for item in object_crops],
                        tracking_ids=[-1]*len(object_crops)
                    )
                
                    self.write_to_fluentd(result)
                    LOGGER.info(f"Found {len(object_crops)} objects/s in camera {self.camera_code}")

            except Exception as ex:
                LOGGER.error(f"No objects identified [{ex}].")

        return result


    def tracking(self, data, output, img_info, det_time):
        # initialize placeholders for the tracking data
        online_tlwhs = []
        online_ids = []
        online_scores = []
        result = None

        original_size = (data.shape[0], data.shape[1])

        try:
            # update tracker
            with FTimer("tracking"):
                online_targets = self.tracker.update(output, original_size, (image_size, image_size))

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

            for i in range(len(online_bboxes)):
                # generate the object crop
                box = online_bboxes[i]
                x0, y0, x1, y1 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                _img = data[y0: y1, x0: x1]

                # encode the object crop
                crop_encoded = cv2.imencode('.jpg', _img)[1]

                # append
                object_crops.append([
                    crop_encoded,
                    box,
                    online_scores[i],
                    online_ids[i],
                ])
            
            if len(object_crops) > 0:
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
                scene = cv2.imencode('.jpg', data, encode_param)[1]

                result = DetTrackResult(
                    bbox=[item[0] for item in object_crops],
                    scene=scene,
                    confidence=[item[2] for item in object_crops],
                    detection_time=det_time,
                    camera=self.camera_code,
                    bbox_coord=[item[1] for item in object_crops],
                    tracking_ids=[item[3] for item in object_crops]
                )
                
                self.write_to_fluentd(result)
                LOGGER.info(f"Tracked {len(object_crops)} objects/s in camera {self.camera_code} with IDs {result.tracklets}")

        except Exception as ex:
            LOGGER.error(f"No objects tracked! [{ex}].")

        return result

    def predict(self, data, **kwargs):
        """
        Run the pedestrian detector/tracker on a list of images
        :param data: path to image.
        :return:
            img_with_bbox List[np.ndarray]
        """
        tresult = []

        imgs_ = list(map(lambda z: z[0], data)) 

        with FTimer("bytetracker_forward"):
            outputs, img_info = self.forward_batched(imgs_)
        
        # Did we detect something?
        if len(outputs) > 0:
            for idx, output in enumerate(outputs):
                try:
                    dettrack_obj = getattr(self, kwargs["op_mode"].value)(data[idx][0], output, img_info, data[idx][1])
                    if dettrack_obj is not None:
                        tresult.append(dettrack_obj)
                except AttributeError as ex:
                    LOGGER.error(f"Operational mode {kwargs['op_mode'].value} not supported. [{ex}]")
                    return []

            LOGGER.info(f"Transmitting {len(tresult)} DetTrack objects for ReID")        
        return tresult