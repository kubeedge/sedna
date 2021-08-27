import argparse
import mindspore as ms
from mindspore import Tensor
import mindspore.dataset.vision.c_transforms as C
import numpy as np
from lib.sedna.backend import set_backend
import cv2
from interface import Estimator

parser = argparse.ArgumentParser(description="resnet50 infer")
parser.add_argument('--image_path', type=str, default="")
parser.add_argument(
    '--device_target',
    type=str,
    default="Ascend",
    choices=(
        "Ascend",
        "CPU"),
    help="Device target, support Ascend, CPU")
parser.add_argument('--checkpoint_path', type=str)


def preprocess():
    resize = C.Resize((224, 224))
    rescale = C.Rescale(1.0 / 255.0, 0.0)
    normalize = C.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    transpose = C.HWC2CHW()
    return [resize, rescale, normalize, transpose]


def main():
    args = parser.parse_args()
    img = cv2.imread(args.image_path)
    data_preprocess = preprocess()
    for method in data_preprocess:
        img = method(img)
    img = np.expand_dims(img, 0)
    data = Tensor(img, ms.float32)
    model = set_backend(estimator=Estimator)
    return model.predict(data)


if __name__ == '__main__':
    main()
