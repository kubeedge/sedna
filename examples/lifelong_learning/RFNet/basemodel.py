import os
import numpy as np
import torch
from PIL import Image
import argparse
from tqdm import tqdm

import cv2
from dataloaders import make_data_loader
from dataloaders import custom_transforms as tr
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision import transforms
from sedna.common.class_factory import ClassType, ClassFactory
from sedna.common.config import Context
from sedna.datasources import TxtDataParse
from sedna.common.file_ops import FileOps
from sedna.common.log import LOGGER

from utils.lr_scheduler import LR_Scheduler
from utils.metrics import Evaluator
from train import Trainer
from eval import Validator
from eval import load_my_state_dict

os.environ["BACKEND_TYPE"] = ''

def preprocess_url(image_urls):
    transformed_images = []
    for paths in image_urls:
        if len(paths) == 2:
            img_path, depth_path = paths
            _img = Image.open(img_path).convert('RGB').resize((2048, 1024), Image.BILINEAR)
            _depth = Image.open(depth_path).resize((2048, 1024), Image.BILINEAR)
        else:
            img_path = paths[0]
            _img = Image.open(img_path).convert('RGB').resize((2048, 1024), Image.BILINEAR)
            # _img = Image.open(img_path).convert('RGB')
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
        img = cv2.resize(np.array(img), (2048, 1024), interpolation=cv2.INTER_CUBIC)
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
        self.val_args = val_args()
        self.train_args = train_args()

        self.train_args.lr = float(kwargs.get("learning_rate", 1e-4))
        self.train_args.epochs = int(kwargs.get("epochs", 2))
        self.train_args.eval_interval = int(kwargs.get("eval_interval", 50))
        self.train_args.no_val = kwargs.get("no_val", True)
        self.train_args.resume = Context.get_parameters("PRETRAINED_MODEL_URL", None)
        self.train_args.num_class = int(kwargs.get("num_class", 31))
        self.trainer = None
        self.train_model_url = None

        label_save_dir = Context.get_parameters(
            "INFERENCE_RESULT_DIR", "./inference_results")
        self.val_args.color_label_save_path = os.path.join(
            label_save_dir, "color")
        self.val_args.merge_label_save_path = os.path.join(
            label_save_dir, "merge")
        self.val_args.label_save_path = os.path.join(label_save_dir, "label")
        self.val_args.save_predicted_image = kwargs.get(
            "save_predicted_image", True)
        self.val_args.num_class = int(kwargs.get("num_class", 31))
        self.val_args.weight_path = kwargs.get("weight_path")
        
        self.validator = Validator(self.val_args)
        # self.val_args.weight_path = "./models/ramp_train1_200.pth"
        # self.val_args.weight_path = "./models/last_None_epoch_268_mean-iu_0.00000.pth"
        self.validator_ramp = Validator(self.val_args)

    def train(self, train_data, valid_data=None, **kwargs):
        self.trainer = Trainer(self.train_args, train_data=train_data)
        print("Total epoches:", self.trainer.args.epochs)
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

    # def predict(self, data, **kwargs):
    #     prediction = kwargs.get('prediction')
    #     if isinstance(data[0], dict):
    #         data = preprocess_frames(data)
    #
    #     if isinstance(data[0], np.ndarray):
    #         data = preprocess_url(data)
    #
    #     self.validator.test_loader = DataLoader(
    #         data,
    #         batch_size=self.val_args.test_batch_size,
    #         shuffle=False,
    #         pin_memory=False)
    #     if not prediction:
    #         return self.validator.validate()
    #     else:
    #         return prediction

    def predict(self, data, **kwargs):
        if isinstance(data[0], np.ndarray):
            data = preprocess_url(data)

        if isinstance(data[0], dict):
            data = preprocess_frames(data)

        self.validator.test_loader = DataLoader(
            data,
            batch_size=self.val_args.test_batch_size,
            shuffle=False,
            pin_memory=False)

        # TODO: predict ramp using specific model
        self.validator_ramp.test_loader = DataLoader(
            data,
            batch_size=self.val_args.test_batch_size,
            shuffle=False,
            pin_memory=False)

        prediction = kwargs.get('prediction')
        if not prediction:
            return (self.validator.validate(), self.validator.validate())
        else:
            return (prediction, self.validator.validate())

    def evaluate(self, data, **kwargs):
        samples = preprocess_url(data.x)
        predictions = self.predict(samples)
        return robo_accuracy(data.y, predictions)

    def load(self, model_url, **kwargs):
        if model_url:
            self.validator.new_state_dict = torch.load(
                model_url, map_location=torch.device("cpu"))
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


def train_args():
    parser = argparse.ArgumentParser(description="PyTorch RFNet Training")
    parser.add_argument(
        '--depth',
        action="store_true",
        default=False,
        help='training with depth image or not (default: False)')
    parser.add_argument('--workers', type=int, default=0,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base-size', type=int, default=1024,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=768,
                        help='crop image size')
    parser.add_argument('--image-size', default=(2048, 1024),
                        help='original image size')
    parser.add_argument('--loss-type', type=str, default='ce',
                        choices=['ce', 'focal'],
                        help='loss func type (default: ce)')
    parser.add_argument('--epochs', type=int, default=None, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                    training (default: auto)')
    parser.add_argument('--val-batch-size', type=int, default=1,
                        metavar='N', help='input batch size for \
                                    testing (default: auto)')
    parser.add_argument('--test-batch-size', type=int, default=1,
                        metavar='N', help='input batch size for \
                                    testing (default: auto)')
    parser.add_argument(
        '--use-balanced-weights',
        action='store_true',
        default=False,
        help='whether to use balanced weights (default: True)')
    parser.add_argument("--class-weight-path", default="", help="if use balanced weights, specify weight path")
    parser.add_argument('--num-class', type=int, default=31,
                        help='number of training classes (default: 24')
    # optimizer params
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: auto)')
    parser.add_argument('--lr-scheduler', type=str, default='cos',
                        choices=['poly', 'step', 'cos', 'inv'],
                        help='lr scheduler mode: (default: cos)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=2.5e-5,
                        metavar='M', help='w-decay (default: 5e-4)')
    # cuda, seed and logging
    parser.add_argument(
        '--no-cuda',
        action='store_true',
        default=False,
        help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default="0",
                        help='use which gpu to train, must be a \
                            comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # checking point
    parser.add_argument('--resume', type=str,
                        default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default=None,
                        help='set the checkpoint name')
    # finetuning pre-trained models
    parser.add_argument('--ft', action='store_true', default=True,
                        help='finetuning on a different dataset')
    # evaluation option
    parser.add_argument('--eval-interval', type=int, default=1,
                        help='evaluation interval (default: 1)')
    parser.add_argument('--no-val', action='store_true', default=False,
                        help='skip validation during training')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    print(torch.cuda.is_available())
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError(
                'Argument --gpu_ids must be a comma-separated list of integers only')

    if args.epochs is None:
        args.epochs = 200

    if args.batch_size is None:
        args.batch_size = 4 * len(args.gpu_ids)

    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size

    if args.lr is None:
        lrs = {
            'cityscapes': 0.0001,
            'citylostfound': 0.0001,
            'cityrand': 0.0001
        }
        args.lr = lrs[args.dataset.lower()] / \
            (4 * len(args.gpu_ids)) * args.batch_size

    if args.checkname is None:
        args.checkname = 'RFNet'
    print(args)
    torch.manual_seed(args.seed)

    return args

def val_args():
    parser = argparse.ArgumentParser(description="PyTorch RFNet validation")
    parser.add_argument('--workers', type=int, default=0,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base-size', type=int, default=1024,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=768,
                        help='crop image size')
    parser.add_argument('--image-size', default=(2048, 1024),
                        help='original image size')
    parser.add_argument('--batch-size', type=int, default=6,
                        help='batch size for training')
    parser.add_argument('--val-batch-size', type=int, default=1,
                        metavar='N', help='input batch size for \
                                    validating (default: auto)')
    parser.add_argument('--test-batch-size', type=int, default=1,
                        metavar='N', help='input batch size for \
                                    testing (default: auto)')
    parser.add_argument('--num-class', type=int, default=31,
                        help='number of training classes (default: 24)')
    parser.add_argument(
        '--no-cuda',
        action='store_true',
        default=False,
        help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='1',
                        help='use which gpu to train, must be a \
                            comma-separated list of integers only (default=0)')
    parser.add_argument('--checkname', type=str, default=None,
                        help='set the checkpoint name')
    parser.add_argument(
        '--weight-path',
        type=str,
        default="./models/2048x1024_80.pth",
        help='enter your path of the weight')
    parser.add_argument(
        '--save-predicted-image',
        action='store_true',
        default=True,
        help='save predicted images')
    parser.add_argument('--color-label-save-path', type=str,
                        default='./test/color/',
                        help='path to save label')
    parser.add_argument('--merge-label-save-path', type=str,
                        default='./test/merge/',
                        help='path to save merged label')
    parser.add_argument('--label-save-path', type=str, default='./test/label/',
                        help='path to save merged label')
    parser.add_argument(
        '--merge',
        action='store_true',
        default=True,
        help='merge image and label')
    parser.add_argument(
        '--depth',
        action='store_true',
        default=False,
        help='add depth image or not')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError(
                'Argument --gpu_ids must be a comma-separated list of integers only')

    return args

def accuracy(y_true, y_pred, **kwargs):
    args = val_args()
    args.num_class = 31
    _, _, test_loader = make_data_loader(args, test_data=y_true)
    evaluator = Evaluator(args.num_class)
   
    tbar = tqdm(test_loader, desc='\r')
    for i, (sample, img_path) in enumerate(tbar):
        if args.depth:
            image, depth, target = sample['image'], sample['depth'], sample['label']
        else:
            image, target = sample['image'], sample['label']

        target[target > args.num_class - 1] = 255
        target = target.cpu().numpy()
        # Add batch sample into evaluator
        evaluator.add_batch(target, y_pred[i])

    print("MIoU:", evaluator.Mean_Intersection_over_Union())

def robo_accuracy(y_true, y_pred, **kwargs):
    y_pred = y_pred[0]
    args = val_args()
    # args.num_class = 4
    _, _, test_loader = make_data_loader(args, test_data=y_true)
    evaluator = Evaluator(args.num_class)

    tbar = tqdm(test_loader, desc='\r')
    for i, (sample, img_path) in enumerate(tbar):
        if args.depth:
            image, depth, target = sample['image'], sample['depth'], sample['label']
        else:
            image, target = sample['image'], sample['label']
        # resize = transforms.Resize([240, 424])
        target = resize(target)
        if args.cuda:
            image, target = image.cuda(), target.cuda()
            if args.depth:
                depth = depth.cuda()

        target[target > evaluator.num_class-1] = 255
        target = target.cpu().numpy()
        # Add batch sample into evaluator
        evaluator.add_batch(target, y_pred[i])

    mIoU = evaluator.Mean_Intersection_over_Union()

if __name__ == '__main__':
    model = Model(num_class=4)
    txt = "./data_txt/ramp_test.txt"
    model.load("./models/last_None_epoch_268_mean-iu_0.00000.pth")

    data = TxtDataParse(data_type="eval", func=_load_txt_dataset)
    data.parse(txt, use_raw=False)
    model.evaluate(data)
    # model.save("./models/e1_1f_1.pth")


    

