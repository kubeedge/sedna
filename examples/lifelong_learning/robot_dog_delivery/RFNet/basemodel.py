import os

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision import transforms
from sedna.common.config import Context
from sedna.datasources import TxtDataParse
from sedna.common.file_ops import FileOps
from sedna.common.log import LOGGER
from sedna.common.config import BaseConfig

from dataloaders import custom_transforms as tr
from utils.args import TrainingArguments, EvaluationArguments
from estimators.train import Trainer
from estimators.eval import Validator, load_my_state_dict
from accuracy import robo_accuracy

os.environ["BACKEND_TYPE"] = ''


def preprocess_url(image_urls):
    transformed_images = []
    for paths in image_urls:
        if len(paths) == 2:
            img_path, depth_path = paths
            _img = Image.open(img_path).convert(
                'RGB').resize((2048, 1024), Image.BILINEAR)
            _depth = Image.open(depth_path).resize(
                (2048, 1024), Image.BILINEAR)
        else:
            img_path = paths[0]
            _img = Image.open(img_path).convert(
                'RGB').resize((2048, 1024), Image.BILINEAR)
            _depth = _img

        sample = {'image': _img, 'depth': _depth, 'label': _img}
        composed_transforms = transforms.Compose([
            # tr.CropBlackArea(),
            # tr.FixedResize(size=self.args.crop_size),
            tr.Normalize(
                mean=(
                    0.485, 0.456, 0.406), std=(
                    0.229, 0.224, 0.225)),
            tr.ToTensor()])

        transformed_images.append((composed_transforms(sample), img_path))

    return transformed_images


def preprocess_frames(frames):
    composed_transforms = transforms.Compose([
        tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        tr.ToTensor()])

    trainsformed_frames = []
    for frame in frames:
        img = frame.get('image')
        img = cv2.resize(np.array(img), (2048, 1024),
                         interpolation=cv2.INTER_CUBIC)
        img = Image.fromarray(np.array(img))
        sample = {'image': img, "depth": img, "label": img}
        trainsformed_frames.append((composed_transforms(sample), ""))

    return trainsformed_frames


def _load_txt_dataset(dataset_url):
    # use original dataset url
    original_dataset_url = Context.get_parameters('original_dataset_url', "")
    dataset_urls = dataset_url.split()
    dataset_urls = [
        os.path.join(
            os.path.dirname(original_dataset_url),
            dataset_url) for dataset_url in dataset_urls]
    return dataset_urls[:-1], dataset_urls[-1]


class Model:
    def __init__(self, **kwargs):
        self.train_args = TrainingArguments(**kwargs)
        self.val_args = EvaluationArguments(**kwargs)

        self.train_args.resume = Context.get_parameters(
            "PRETRAINED_MODEL_URL", None)
        self.trainer = None
        self.train_model_url = None

        label_save_dir = Context.get_parameters("INFERENCE_RESULT_DIR", os.path.join(
            BaseConfig.data_path_prefix, "inference_results"))
        self.val_args.color_label_save_path = os.path.join(
            label_save_dir, "color")
        self.val_args.merge_label_save_path = os.path.join(
            label_save_dir, "merge")
        self.val_args.label_save_path = os.path.join(label_save_dir, "label")
        self.val_args.weight_path = kwargs.get("weight_path")
        self.validator = Validator(self.val_args)

        # self.ramp_val_args = EvaluationArguments()
        # self.ramp_val_args.weight_path = "/home/lsq/RFNet/models/ramp_train1_200.pth"
        # self.ramp_val_args.merge = False
        # self.validator_ramp = Validator(self.ramp_val_args)

    def train(self, train_data, valid_data=None, **kwargs):
        self.trainer = Trainer(self.train_args, train_data=train_data)
        LOGGER.info("Total epoches: {}".format(self.trainer.args.epochs))
        for epoch in range(
                self.trainer.args.start_epoch,
                self.trainer.args.epochs):
            if epoch == 0 and self.trainer.val_loader:
                self.trainer.validation(epoch)
            self.trainer.training(epoch)

            if self.trainer.args.no_val and (epoch %
                                             self.trainer.args.eval_interval == (
                                                 self.trainer.args.eval_interval -
                                                 1) or epoch == self.trainer.args.epochs - 1):
                # save checkpoint when it meets eval_interval or the training finishes
                is_best = False
                train_model_url = self.trainer.saver.save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': self.trainer.model.state_dict(),
                    'optimizer': self.trainer.optimizer.state_dict(),
                    'best_pred': self.trainer.best_pred,
                }, is_best)

            # if not self.trainer.args.no_val and \
            #         epoch % self.train_args.eval_interval == (self.train_args.eval_interval - 1) \
            #         and self.trainer.val_loader:
            #     self.trainer.validation(epoch)

        self.trainer.writer.close()

        self.train_model_url = train_model_url
        return self.train_model_url

    def predict(self, data, **kwargs):
        prediction = kwargs.get('prediction')
        if isinstance(data[0], dict):
            data = preprocess_frames(data)
    
        if isinstance(data[0], np.ndarray):
            data = preprocess_url(data)
    
        self.validator.test_loader = DataLoader(
            data,
            batch_size=self.val_args.test_batch_size,
            shuffle=False,
            pin_memory=False)

        if not prediction:
            return self.validator.validate()
        else:
            return prediction

    # def predict(self, data, **kwargs):
    #     if isinstance(data[0], np.ndarray):
    #         data = preprocess_url(data)

    #     if isinstance(data[0], dict):
    #         data = preprocess_frames(data)

    #     self.validator.test_loader = DataLoader(
    #         data,
    #         batch_size=self.val_args.test_batch_size,
    #         shuffle=False,
    #         pin_memory=False)

    #     # TODO: predict ramp using specific model
    #     self.validator_ramp.test_loader = DataLoader(
    #         data,
    #         batch_size=self.val_args.test_batch_size,
    #         shuffle=False,
    #         pin_memory=False)

    #     prediction = kwargs.get('prediction')
    #     if not prediction:
    #         return (self.validator.validate(), self.validator_ramp.validate())
    #     else:
    #         return (prediction, self.validator_ramp.validate())

    def evaluate(self, data, **kwargs):
        predictions = self.predict(data.x)
        return robo_accuracy(data.y, predictions)

    def load(self, model_url, **kwargs):
        if model_url:
            self.validator.new_state_dict = torch.load(model_url)
            self.validator.model = load_my_state_dict(
                self.validator.model, self.validator.new_state_dict['state_dict'])

            self.train_args.resume = model_url
        else:
            raise Exception("model url does not exist.")

    def save(self, model_path=None):
        if not model_path:
            LOGGER.warning(f"Not specify model path.")
            return self.train_model_url

        return FileOps.upload(self.train_model_url, model_path)


if __name__ == '__main__':
    model = Model(num_class=31)
    txt = "/home/lsq/RFNet/data_index/train.txt"
    model.load("/home/lsq/RFNet/models/best_all_epoch_142_mean-iu_0.94952.pth")

    data = TxtDataParse(data_type="eval", func=_load_txt_dataset)
    data.parse(txt, use_raw=False)
    model.evaluate(data)
    # model.save("./models/e1_2f.pth")
