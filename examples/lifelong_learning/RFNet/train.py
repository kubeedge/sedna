import argparse
import os
import numpy as np
from tqdm import tqdm

from sedna.datasources import BaseDataSource

import torch
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
        self.train_loader, self.val_loader, self.test_loader = make_data_loader(args, train_data=train_data,
                                                                                valid_data=valid_data, **kwargs)
                                
        # Define network
        resnet = resnet18(pretrained=True, efficient=False, use_bn=True)
        model = RFNet(resnet, num_classes=self.nclass, use_bn=True)
        train_params = [{'params': model.random_init_params(), 'lr': args.lr},
                        {'params': model.fine_tune_params(), 'lr': 0.1*args.lr, 'weight_decay': args.weight_decay}]
        # Define Optimizer
        optimizer = torch.optim.Adam(train_params, lr=args.lr,
                                    weight_decay=args.weight_decay)
        # Define Criterion
        # whether to use class balanced weights
        if args.use_balanced_weights:
            classes_weights_path = os.path.join(args.class_weight_path, 'classes_weights.npy')
            if os.path.isfile(classes_weights_path):
                weight = np.load(classes_weights_path)
            else:
                weight = calculate_weigths_labels(args.class_weight_path, self.train_loader, self.nclass)
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
            else:
                image, target = sample['image'], sample['label']
            print(f"shape of image batch:", image.shape)
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
        
            target[target > self.nclass-1] = 255
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))
            self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)
            # Show 10 * 3 inference results each epoch
            if i % (num_img_tr // 10) == 0:
                global_step = i + num_img_tr * epoch
                if self.args.depth:
                    self.summary.visualize_image(self.writer, "cityscapes", image, target, output, global_step)

                    depth_display = depth[0].cpu().unsqueeze(0)
                    depth_display = depth_display.mul_(self.std_depth).add_(self.mean_depth)
                    depth_display = depth_display.numpy()
                    depth_display = depth_display*255
                    depth_display = depth_display.astype(np.uint8)
                    self.writer.add_image('Depth', depth_display, global_step)

                else:
                    self.summary.visualize_image(self.writer, "cityscapes", image, target, output, global_step)

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
