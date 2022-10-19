import os
# set at yaml
# os.environ["test_dataset_url"] = "./data_txt/sedna_test.txt"
# os.environ["video_url"] = "./video/radio.mp4"
# os.environ["MODEL_URLS"] = "./cloud_kb/index.pkl"
# os.environ["unseen_save_url"] = "s3://kubeedge/sedna-robo/unseen_samples/"
# os.environ["OOD_backup_model"] = "./models/ramp_train1_200.pth"
# os.environ["OOD_model"] = "./models/lr_model.model"

import cv2
import time
import torch
import numpy as np
from PIL import Image
import base64
import tempfile
import warnings
from io import BytesIO

from torchvision import transforms
from sedna.datasources import BaseDataSource, TxtDataParse
from basemodel import preprocess_url, val_args, Model, preprocess_frames
from sedna.core.lifelong_learning import LifelongLearning
from sedna.common.config import Context

from dataloaders import custom_transforms as tr
from accuracy import accuracy

def preprocess(samples):
    data = BaseDataSource(data_type="test")
    data.x = [samples]
    return data

def postprocess(samples):
    image_names, imgs = [], []
    for sample in samples:
        img = sample.get("image")
        image_names.append("{}.png".format(str(time.time())))
        imgs.append(img)
    
    return image_names, imgs
        
def _load_txt_dataset(dataset_url):
    # use original dataset url
    original_dataset_url = Context.get_parameters('original_dataset_url', "")
    dataset_urls = dataset_url.split()
    dataset_urls = [
        os.path.join(
            os.path.dirname(original_dataset_url),
            dataset_url) for dataset_url in dataset_urls]
    return dataset_urls[:-1], dataset_urls[-1]

def init_ll_job(**kwargs):
    estimator = Model(num_class=31, **kwargs)

    task_allocation = {
        "method": "TaskAllocationByOrigin",
        "param": {
            "default": "real"
        }
    }
    unseen_task_allocation = {
        "method": "UnseenTaskAllocationDefault"
    }

    unseen_sample_recognition = {
        "method": "OodIdentification",
        "param": {
            "OOD_thresh": 0.15,
            "backup_model": os.getenv("OOD_backup_model"),
            "OOD_model_path": os.getenv("OOD_model"),
            "preprocess_func": preprocess_frames,
            "base_model": Model
        }
    }

    # unseen_sample_recognition = {
    #     "method": "SampleRegonitionRobotic"
    # }

    inference_integrate = {
        "method": "InferenceIntegrateByType"
    }
    

    ll_job = LifelongLearning(
        estimator,
        unseen_estimator=unseen_task_processing,
        task_definition=None,
        task_relationship_discovery=None,
        task_allocation=task_allocation,
        task_remodeling=None,
        inference_integrate=inference_integrate,
        task_update_decision=None,
        unseen_task_allocation=unseen_task_allocation,
        unseen_sample_recognition=unseen_sample_recognition,
        unseen_sample_re_recognition=None)

    return ll_job

def unseen_task_processing():
    return "Warning: unseen sample detected."

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
        data = {'image': img_rgb, "depth": img_rgb, "label": img_rgb}
        data = preprocess(data)
        print(postprocess)
        print("Inference results:", ll_job.inference(data=data, post_process=postprocess))

def predict_batch():
    ll_job = init_ll_job()

    test_dataset_url = Context.get_parameters("test_dataset_url")
    test_data = TxtDataParse(data_type="test", func=_load_txt_dataset)
    test_data.parse(test_dataset_url, use_raw=False)

    return ll_job.inference(data=test_data)

if __name__ == '__main__':
    print(predict())
