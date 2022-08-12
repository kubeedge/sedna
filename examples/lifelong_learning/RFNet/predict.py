import os
os.environ['BACKEND_TYPE'] = 'PYTORCH'
# set at yaml
# os.environ["PREDICT_RESULT_DIR"] = "./inference_results"
# os.environ["EDGE_OUTPUT_URL"] = "./edge_kb"
# os.environ["video_url"] = "./video/radio.mp4"
# os.environ["MODEL_URLS"] = "./cloud_next_kb/index.pkl"


import cv2
import time
import torch
import numpy as np
from PIL import Image
import base64
import tempfile
import warnings
from io import BytesIO

from sedna.datasources import BaseDataSource
from sedna.core.lifelong_learning import LifelongLearning
from sedna.common.config import Context

from dataloaders import custom_transforms as tr
from torchvision import transforms

from accuracy import accuracy
from basemodel import preprocess, val_args, Model

def preprocess(samples):
    composed_transforms = transforms.Compose([
        tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        tr.ToTensor()])

    data = BaseDataSource(data_type="test")
    data.x = [(composed_transforms(samples), "")]
    return data

def init_ll_job():
    estimator = Model()

    task_allocation = {
        "method": "TaskAllocationByOrigin",
        "param": {
            "origins": ["real", "sim"],
            "default": "real"
        }
    }
    unseen_task_allocation = {
        "method": "UnseenTaskAllocationDefault"
    }

    ll_job = LifelongLearning(
        estimator,
        task_definition=None,
        task_relationship_discovery=None,
        task_allocation=task_allocation,
        task_remodeling=None,
        inference_integrate=None,
        task_update_decision=None,
        unseen_task_allocation=unseen_task_allocation,
        unseen_sample_recognition=None,
        unseen_sample_re_recognition=None)

    return ll_job

def predict():
    ll_job = init_ll_job()

    camera_address = Context.get_parameters('video_url')
    # use video streams for testing
    camera = cv2.VideoCapture(camera_address)
    fps = 10
    nframe = 0
    while 1:
        ret, input_yuv = camera.read()
        if not ret:
            time.sleep(5)
            camera = cv2.VideoCapture(camera_address)
            continue

        if nframe % fps:
            nframe += 1
            continue

        img_rgb = cv2.cvtColor(input_yuv, cv2.COLOR_BGR2RGB)
        nframe += 1
        if nframe % 1000 == 1:  # logs every 1000 frames
            warnings.warn(f"camera is open, current frame index is {nframe}")

        img_rgb = cv2.resize(np.array(img_rgb), (2048, 1024), interpolation=cv2.INTER_CUBIC)
        img_rgb = Image.fromarray(img_rgb)
        sample = {'image': img_rgb, "depth": img_rgb, "label": img_rgb}
        data = preprocess(sample)
        print("Inference results:", ll_job.inference(data=data))

if __name__ == '__main__':
    predict()
