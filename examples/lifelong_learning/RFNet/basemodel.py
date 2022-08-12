import os
import numpy as np
import torch
from PIL import Image
import argparse
from train import Trainer
from eval import Validator
from tqdm import tqdm
from eval import load_my_state_dict
from utils.metrics import Evaluator
from dataloaders import make_data_loader
from dataloaders import custom_transforms as tr
from torchvision import transforms
from sedna.common.class_factory import ClassType, ClassFactory
from sedna.common.config import Context
from sedna.datasources import TxtDataParse
from torch.utils.data import DataLoader
from sedna.common.file_ops import FileOps
from utils.lr_scheduler import LR_Scheduler

def preprocess(image_urls):
    transformed_images = []
    for paths in image_urls:
        if len(paths) == 2:
            img_path, depth_path = paths
            _img = Image.open(img_path).convert('RGB')
            _depth = Image.open(depth_path)
        else:
            img_path = paths[0]
            _img = Image.open(img_path).convert('RGB')
            _depth = _img

        sample = {'image': _img, 'depth': _depth, 'label': _img}
        composed_transforms = transforms.Compose([
            # tr.CropBlackArea(),
            # tr.FixedResize(size=self.args.crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        transformed_images.append((composed_transforms(sample), img_path))

    return transformed_images

class Model:
    def __init__(self, **kwargs):
        self.val_args = val_args()
        self.train_args = train_args()

        self.train_args.lr = kwargs.get("learning_rate", 1e-4)
        self.train_args.epochs = kwargs.get("epochs", 2)
        self.train_args.eval_interval = kwargs.get("eval_interval", 2)
        self.train_args.no_val = kwargs.get("no_val", True)
        # self.train_args.resume = Context.get_parameters("PRETRAINED_MODEL_URL", None)
        self.trainer = None

        label_save_dir = Context.get_parameters("INFERENCE_RESULT_DIR", "./inference_results")
        self.val_args.color_label_save_path = os.path.join(label_save_dir, "color")
        self.val_args.merge_label_save_path = os.path.join(label_save_dir, "merge")
        self.val_args.label_save_path = os.path.join(label_save_dir, "label")
        self.validator = Validator(self.val_args)

    def train(self, train_data, valid_data=None, **kwargs):        
        self.trainer = Trainer(self.train_args, train_data=train_data)
        print("Total epoches:", self.trainer.args.epochs)
        for epoch in range(self.trainer.args.start_epoch, self.trainer.args.epochs):
            if epoch == 0 and self.trainer.val_loader:
                self.trainer.validation(epoch)
            self.trainer.training(epoch)

            if self.trainer.args.no_val and \
                    (epoch % self.trainer.args.eval_interval == (self.trainer.args.eval_interval - 1)
                     or epoch == self.trainer.args.epochs - 1):
                # save checkpoint when it meets eval_interval or the training finished
                is_best = False
                checkpoint_path = self.trainer.saver.save_checkpoint({
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

        return checkpoint_path

    def predict(self, data, **kwargs):
        if not isinstance(data[0][0], dict):
            data = preprocess(data)

        if type(data) is np.ndarray:
            data = data.tolist()

        self.validator.test_loader = DataLoader(data, batch_size=self.val_args.test_batch_size, shuffle=False,
                                                pin_memory=True)
        return self.validator.validate()

    def evaluate(self, data, **kwargs):
        self.val_args.save_predicted_image = kwargs.get("save_predicted_image", True)
        samples = preprocess(data.x)
        predictions = self.predict(samples)
        return accuracy(data.y, predictions)

    def load(self, model_url, **kwargs):
        if model_url:
            self.validator.new_state_dict = torch.load(model_url, map_location=torch.device("cpu"))
            self.train_args.resume = model_url
        else:
            raise Exception("model url does not exist.")
        self.validator.model = load_my_state_dict(self.validator.model, self.validator.new_state_dict['state_dict'])

    def save(self, model_path=None):
        # TODO: how to save unstructured data model
        pass                                               

def train_args():
    parser = argparse.ArgumentParser(description="PyTorch RFNet Training")
    parser.add_argument('--depth', action="store_true", default=False,
                        help='training with depth image or not (default: False)')
    parser.add_argument('--dataset', type=str, default='cityscapes',
                        choices=['citylostfound', 'cityscapes', 'cityrand', 'target', 'xrlab', 'e1', 'mapillary'],
                        help='dataset name (default: cityscapes)')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base-size', type=int, default=1024,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=768,
                        help='crop image size')
    parser.add_argument('--loss-type', type=str, default='ce',
                        choices=['ce', 'focal'],
                        help='loss func type (default: ce)')
    # training hyper params
    # parser.add_argument('--epochs', type=int, default=None, metavar='N',
    #                     help='number of epochs to train (default: auto)')
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
    parser.add_argument('--use-balanced-weights', action='store_true', default=False,
                        help='whether to use balanced weights (default: True)')
    parser.add_argument('--num-class', type=int, default=24,
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
    parser.add_argument('--no-cuda', action='store_true', default=
    False, help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
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
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.epochs is None:
        epoches = {
            'cityscapes': 200,
            'citylostfound': 200,
        }
        args.epochs = epoches[args.dataset.lower()]

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
        args.lr = lrs[args.dataset.lower()] / (4 * len(args.gpu_ids)) * args.batch_size

    if args.checkname is None:
        args.checkname = 'RFNet'
    print(args)
    torch.manual_seed(args.seed)

    return args

def val_args():
    parser = argparse.ArgumentParser(description="PyTorch RFNet validation")
    parser.add_argument('--dataset', type=str, default='cityscapes',
                        choices=['citylostfound', 'cityscapes', 'xrlab', 'mapillary'],
                        help='dataset name (default: cityscapes)')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base-size', type=int, default=1024,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=768,
                        help='crop image size')
    parser.add_argument('--batch-size', type=int, default=6,
                        help='batch size for training')
    parser.add_argument('--val-batch-size', type=int, default=1,
                        metavar='N', help='input batch size for \
                                    validating (default: auto)')
    parser.add_argument('--test-batch-size', type=int, default=1,
                        metavar='N', help='input batch size for \
                                    testing (default: auto)')
    parser.add_argument('--num-class', type=int, default=24,
                        help='number of training classes (default: 24')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                            comma-separated list of integers only (default=0)')
    parser.add_argument('--checkname', type=str, default=None,
                        help='set the checkpoint name')
    parser.add_argument('--weight-path', type=str, default="./models/530_exp3_2.pth",
                        help='enter your path of the weight')
    parser.add_argument('--save-predicted-image', action='store_true', default=False,
                        help='save predicted images')
    parser.add_argument('--color-label-save-path', type=str,
                        default='./test/color/',
                        help='path to save label')
    parser.add_argument('--merge-label-save-path', type=str,
                        default='./test/merge/',
                        help='path to save merged label')
    parser.add_argument('--label-save-path', type=str, default='./test/label/',
                        help='path to save merged label')
    parser.add_argument('--merge', action='store_true', default=True, help='merge image and label')
    parser.add_argument('--depth', action='store_true', default=False, help='add depth image or not')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    return args

def accuracy(y_true, y_pred, **kwargs):
    args = val_args()
    _, _, test_loader, num_class = make_data_loader(args, test_data=y_true)
    evaluator = Evaluator(num_class)

    tbar = tqdm(test_loader, desc='\r')
    for i, (sample, img_path) in enumerate(tbar):
        if args.depth:
            image, depth, target = sample['image'], sample['depth'], sample['label']
        else:
            image, target = sample['image'], sample['label']
        if args.cuda:
            image, target = image.cuda(), target.cuda()
            if args.depth:
                depth = depth.cuda()

        target[target > evaluator.num_class-1] = 255
        target = target.cpu().numpy()
        # Add batch sample into evaluator
        evaluator.add_batch(target, y_pred[i])

    # Test during the training
    # Acc = evaluator.Pixel_Accuracy()
    CPA = evaluator.Pixel_Accuracy_Class()
    mIoU = evaluator.Mean_Intersection_over_Union()
    FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()

    print("CPA:{}, mIoU:{}, fwIoU: {}".format(CPA, mIoU, FWIoU))
    return CPA

if __name__ == '__main__':
    model_path = "/tmp/RFNet/"
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    p1 = Process(target=exp_train, args=(10,))
    p1.start()
    p1.join()
