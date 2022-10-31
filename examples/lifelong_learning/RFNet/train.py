import argparse
import os
import numpy as np
from tqdm import tqdm

from sedna.datasources import BaseDataSource

import torch
from mypath import Path
from models.rfnet import RFNet
from models.resnet.resnet_single_scale_single_attention import *
from models.replicate import patch_replication_callback
from dataloaders import make_data_loader
from utils.loss import SegmentationLosses
from utils.calculate_weights import calculate_weigths_labels
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator

class Trainer(object):
    def __init__(self, args, train_data=None, valid_data=None):
        self.args = args
        # Define Saver
        self.saver = Saver(args)
        self.saver.save_experiment_config()
        # Define Tensorboard Summary
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()
        # denormalize for detph image
        self.mean_depth = torch.as_tensor(0.12176, dtype=torch.float32, device='cpu')
        self.std_depth = torch.as_tensor(0.09752, dtype=torch.float32, device='cpu')
        self.nclass = args.num_class
        # Define Dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': False}
        self.train_loader, self.val_loader, self.test_loader, _ = make_data_loader(args, train_data=train_data, 
                                                                                   valid_data=valid_data, **kwargs)                                                                                       
                                
        # Define network
        resnet = resnet18(pretrained=True, efficient=False, use_bn=True)
        model = RFNet(resnet, num_classes=self.nclass, use_bn=True)
        train_params = [{'params': model.random_init_params(), 'lr': args.lr},
                        {'params': model.fine_tune_params(), 'lr': 0.1*args.lr, 'weight_decay':args.weight_decay}]
        # Define Optimizer
        optimizer = torch.optim.Adam(train_params, lr=args.lr,
                                    weight_decay=args.weight_decay)
        # Define Criterion
        # whether to use class balanced weights
        if args.use_balanced_weights:
            classes_weights_path = os.path.join(Path.db_root_dir(args.dataset), args.dataset+'_classes_weights.npy')
            if os.path.isfile(classes_weights_path):
                weight = np.load(classes_weights_path)
            else:
                weight = calculate_weigths_labels(args.dataset, self.train_loader, self.nclass)
            weight = torch.from_numpy(weight.astype(np.float32))
        else:
            weight = None
        # Define loss function
        self.criterion = SegmentationLosses(weight=weight, cuda=args.cuda).build_loss(mode=args.loss_type)
        self.model, self.optimizer = model, optimizer
        # Define Evaluator
        self.evaluator = Evaluator(self.nclass)
        # # Define lr scheduler
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr, args.epochs, len(self.train_loader))
        # Using cuda
        if args.cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            patch_replication_callback(self.model)
            self.model = self.model.cuda()

        # Resuming checkpoint
        self.best_pred = 0.0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
            print(f"Training: load model from {args.resume}")
            checkpoint = torch.load(args.resume, map_location=torch.device("cpu"))
            args.start_epoch = checkpoint['epoch']
            # if args.cuda:
            #     self.model.load_state_dict(checkpoint['state_dict'])
            # else:
            #     self.model.load_state_dict(checkpoint['state_dict'])
            self.model.load_state_dict(checkpoint['state_dict'])
            if not args.ft:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))

        # Clear start epoch if fine-tuning
        if args.ft:
            args.start_epoch = 0

    def training(self, epoch):
        train_loss = 0.0
        print(self.optimizer.state_dict()['param_groups'][0]['lr'])
        self.model.train()
        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)
        for i, sample in enumerate(tbar):
            if self.args.depth:
                image, depth, target = sample['image'], sample['depth'], sample['label']
                #print(target.shape)
            else:
                image, target = sample['image'], sample['label']
                print(image.shape)
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
                if self.args.depth:
                    depth = depth.cuda()
            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()
            if self.args.depth:
                output = self.model(image, depth)
            else:
                output = self.model(image)
            #print(target.max())
            #print(output.shape)
            target[target > self.nclass-1] = 255
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            #print(self.optimizer.state_dict()['param_groups'][0]['lr'])
            train_loss += loss.item()
            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))
            self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)
            # Show 10 * 3 inference results each epoch
            if i % (num_img_tr // 10) == 0:
                global_step = i + num_img_tr * epoch
                if self.args.depth:
                    self.summary.visualize_image(self.writer, self.args.dataset, image, target, output, global_step)

                    depth_display = depth[0].cpu().unsqueeze(0)
                    depth_display = depth_display.mul_(self.std_depth).add_(self.mean_depth)
                    depth_display = depth_display.numpy()
                    depth_display = depth_display*255
                    depth_display = depth_display.astype(np.uint8)
                    self.writer.add_image('Depth', depth_display, global_step)

                else:
                    self.summary.visualize_image(self.writer, self.args.dataset, image, target, output, global_step)

        self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print('Loss: %.3f' % train_loss)

        # if self.args.no_val:
        #     # save checkpoint every epoch
        #     is_best = False
        #     checkpoint_path = self.saver.save_checkpoint({
        #                         'epoch': epoch + 1,
        #                         'state_dict': self.model.state_dict(),
        #                         'optimizer': self.optimizer.state_dict(),
        #                         'best_pred': self.best_pred,
        #                         }, is_best)

    def validation(self, epoch):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc='\r')
        test_loss = 0.0
        for i, (sample, img_path) in enumerate(tbar):
            if self.args.depth:
                image, depth, target = sample['image'], sample['depth'], sample['label']
            else:
                image, target = sample['image'], sample['label']
                # print(f"val image is {image}")
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
                if self.args.depth:
                    depth = depth.cuda()
            with torch.no_grad():
                if self.args.depth:
                    output = self.model(image, depth)
                else:
                    output = self.model(image)
            target[target > self.nclass-1] = 255
            loss = self.criterion(output, target)
            test_loss += loss.item()
            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            # Add batch sample into evaluator
            self.evaluator.add_batch(target, pred)

        # Fast test during the training
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        self.writer.add_scalar('val/total_loss_epoch', test_loss, epoch)
        self.writer.add_scalar('val/mIoU', mIoU, epoch)
        self.writer.add_scalar('val/Acc', Acc, epoch)
        self.writer.add_scalar('val/Acc_class', Acc_class, epoch)
        self.writer.add_scalar('val/fwIoU', FWIoU, epoch)
        print('Validation:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
        print('Loss: %.3f' % test_loss)

        new_pred = mIoU
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)

def train():
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
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
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
    parser.add_argument('--resume', type=str, default=None,
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
    args.resume = "/home/lsq/ianvs_new/examples/robo/workspace/pcb-algorithm-test/test-algorithm/6441f8be-d809-11ec-ac65-3b30682caaa6/knowledgeable/1/seen_task/checkpoint_1653029539.6730478.pth"
    trainer = Trainer(args, train_data=val_data, valid_data=val_data)
    trainer.validation(0)
    # print('Starting Epoch:', trainer.args.start_epoch)
    # print('Total Epoches:', trainer.args.epochs)
    # for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
    #     if epoch == 0:
    #        trainer.validation(epoch)
    #     trainer.training(epoch)
    #     if not trainer.args.no_val and epoch % args.eval_interval == (args.eval_interval - 1):
    #         trainer.validation(epoch)
    #
    # trainer.writer.close()

if __name__ == "__main__":
    val_data = BaseDataSource(data_type="eval")
    x, y = [], []
    with open("/home/lsq/ianvs_new/examples/robo/trainData_depth.txt", "r") as f:
        for line in f.readlines()[-120:]:
            lines = line.strip().split()
            x.append(lines[:2])
            y.append(lines[-1])

    val_data.x = x
    val_data.y = y

    train()