<<<<<<< HEAD
import os

import cv2
=======
>>>>>>> 8f1f777b (Code check and fix)
import time
os.environ["MODEL_URLS"] = "s3://kubeedge/sedna-robo/kb_next/index.pkl"

# set in yaml
os.environ["unseen_save_url"] = "s3://kubeedge/sedna-robo/unseen_samples/"
os.environ["OOD_model"] = "s3://kubeedge/sedna-robo/models/lr_model.model"
os.environ["OOD_thresh"] = "0.0001"

os.environ["robo_skill"] = "ramp_detection"
os.environ["ramp_detection"] = "s3://kubeedge/sedna-robo/models/ramp_train1_200.pth"
os.environ["curb_detection"] = "s3://kubeedge/sedna-robo/models/2048x1024_80.pth"

os.environ["S3_ENDPOINT_URL"] = "https://obs.cn-north-1.myhuaweicloud.com"
os.environ["SECRET_ACCESS_KEY"] = "OYPxi4uD9k5E90z0Od3Ug99symbJZ0AfyB4oveQc"
os.environ["ACCESS_KEY_ID"] = "EMPTKHQUGPO2CDUFD2YR"

from sedna.datasources import BaseDataSource
from sedna.core.lifelong_learning import LifelongLearning

from basemodel import Model


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

def init_ll_job():
    estimator = Model(num_class=31,
                      save_predicted_image=True,
                      merge=True)

    task_allocation = {
        "method": "TaskAllocationDefault"
    }
    unseen_task_allocation = {
        "method": "UnseenTaskAllocationDefault"
    }
   
    ll_job = LifelongLearning(
        estimator,
        unseen_estimator=unseen_task_processing,
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


def unseen_task_processing():
    return "Warning: unseen sample detected."
