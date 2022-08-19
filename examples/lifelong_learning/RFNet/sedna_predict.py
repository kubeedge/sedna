import os

import torch
import numpy as np
from PIL import Image
import base64
import tempfile
from io import BytesIO
from torchvision.transforms import ToPILImage
from torchvision import transforms
from torch.utils.data import DataLoader

from sedna.datasources import IndexDataParse
from sedna.core.lifelong_learning import LifelongLearning
from sedna.common.config import Context

from eval import Validator
from accuracy import accuracy
from basemodel import preprocess, val_args, Model
from dataloaders.utils import Colorize
from dataloaders import custom_transforms as tr
from dataloaders.datasets.cityscapes import CityscapesSegmentation

def _load_txt_dataset(dataset_url):
    # use original dataset url
    original_dataset_url = Context.get_parameters('original_dataset_url')
    return os.path.join(os.path.dirname(original_dataset_url), dataset_url)

def fetch_data():
    test_dataset_url = Context.get_parameters("test_dataset_url")
    test_data = IndexDataParse(data_type="test", func=_load_txt_dataset)
    test_data.parse(test_dataset_url, use_raw=False)
    return test_data

def pre_data_process(samples):
    composed_transforms = transforms.Compose([
        tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        tr.ToTensor()])

    data = BaseDataSource(data_type="test")
    data.x = [(composed_transforms(samples), "")]
    return data

def post_process(res, is_unseen_task):
    if is_unseen_task:
        res, base64_string = None, None
    else:
        res = res[0].tolist()

    type = 0 if not is_unseen_task else 1
    mesg = {
        "msg": "",
        "result": {
            "type": type,
            "box": res
        },
        "code": 0
    }
    return mesg

def image_merge(raw_img, result):
    raw_img = ToPILImage()(raw_img)

    pre_colors = Colorize()(torch.from_numpy(result))
    pre_color_image = ToPILImage()(pre_colors[0])  # pre_colors.dtype = float64

    image = raw_img.resize(pre_color_image.size, Image.BILINEAR)
    image = image.convert('RGBA')
    label = pre_color_image.convert('RGBA')
    image = Image.blend(image, label, 0.6)
    with tempfile.NamedTemporaryFile(suffix='.png') as f:
        image.save(f.name)

        with open(f.name, 'rb') as open_file:
            byte_content = open_file.read()
        base64_bytes = base64.b64encode(byte_content)
        base64_string = base64_bytes.decode('utf-8')
    return base64_string

def init_ll_job():
    estimator = Model()
    inference_integrate = {
        "method": "BBoxInferenceIntegrate"
    }
    unseen_task_allocation = {
        "method": "UnseenTaskAllocationDefault"
    }
    unseen_sample_recognition = {
        "method": "SampleRegonitionByRFNet"
    }

    ll_job = LifelongLearning(
        estimator,
        task_definition=None,
        task_relationship_discovery=None,
        task_allocation=None,
        task_remodeling=None,
        inference_integrate=inference_integrate,
        task_update_decision=None,
        unseen_task_allocation=unseen_task_allocation,
        unseen_sample_recognition=unseen_sample_recognition,
        unseen_sample_re_recognition=None)

    args = val_args()
    args.weight_path = "./models/detection_model.pth"
    args.num_class = 31

    return ll_job, Validator(args, unseen_detection=True)

def predict(ll_job, data=None, validator=None):
    if data:
        data = pre_data_process(data)
    else:
        data = fetch_data()
        data.x = preprocess(data.x)

    res, is_unseen_task, _ = ll_job.inference(
        data, validator=validator, initial=False)
    return post_process(res, is_unseen_task)

if __name__ == '__main__':
    ll_job, validator = init_ll_job()
    print("Inference result:", predict(ll_job, validator=validator))
