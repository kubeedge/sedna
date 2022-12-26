import os
import time

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
