import os

import torch
from sedna.common.config import BaseConfig


class Arguments:
    '''
    Setting basic arguments for RFNet model
    '''

    def __init__(self, **kwargs):
        # whether to use depth images or not
        self.depth = kwargs.get("depth", False)
        # number of dataloader threads
        self.workers = int(kwargs.get("workers", 0))
        self.base_size = int(kwargs.get("base-size", 1024))  # base image size
        self.crop_size = int(kwargs.get("crop_size", 768))  # crop image size
        self.image_size = kwargs.get(
            "image_size", (2048, 1024))  # output image shape
        # input batch size for training
        self.batch_size = kwargs.get("batch_size")
        self.val_batch_size = int(kwargs.get(
            "val_batch_size", 1))  # input batch size for validation
        self.test_batch_size = int(kwargs.get(
            "test_batch_size", 1))  # input batch size for testing
        self.num_class = int(kwargs.get(
            "num_class", 31))  # number of training classes
        # whether to disable CUDA for training
        self.no_cuda = kwargs.get("no_cuda", False)
        # use which gpu to train which must be a comma-separated list of
        # integers only
        self.gpu_ids = kwargs.get("gpu_ids", "0, 1")
        self.checkname = kwargs.get(
            "checkname", "RFNet")  # set the checkpoint name

        self.cuda = not self.no_cuda and torch.cuda.is_available()
        if self.cuda:
            try:
                self.gpu_ids = [int(s) for s in self.gpu_ids.split(',')]
            except ValueError:
                raise ValueError(
                    'Argument --gpu_ids must be a comma-separated list of integers only')


class TrainingArguments(Arguments):
    '''
    Setting basic arguments for RFNet training
    '''

    def __init__(self, **kwargs):
        super(TrainingArguments, self).__init__(**kwargs)

        self.loss_type = kwargs.pop('loss_type', "ce")  # loss function type
        # number of epochs to train
        self.epochs = int(kwargs.get("epochs", 200))
        # the index of epoch to start training
        self.start_epoch = int(kwargs.get("start_epoch", 0))
        self.use_balanced_weights = kwargs.get(
            "use_balanced_weights",
            False)  # whether to use balanced weights
        # if use balanced weights, specify weight path
        self.class_weight_path = kwargs.get("class_weight_path", None)
        self.lr = float(kwargs.get("lr", 1e-4))  # learning rate
        self.lr_scheduler = kwargs.get(
            "lr_scheduler", "cos")  # lr scheduler mode
        self.momentum = float(kwargs.get("momentum", 0.9))
        self.weight_decay = float(kwargs.get("weight_decay", 2.5e-5))
        self.seed = int(kwargs.get("seed", 1))  # random seed
        # put the path to resuming file if needed
        self.resume = kwargs.get("resume", None)
        # whether to finetune on a different dataset
        self.ft = kwargs.get("ft", True)
        self.eval_interval = int(
            kwargs.get(
                "eval_interval",
                100))  # evaluation interval
        # whether to skip validation during training
        self.no_val = kwargs.get("no_val", True)

        if not self.batch_size:
            self.batch_size = 4 * len(self.gpu_ids)

        torch.manual_seed(self.seed)


class EvaluationArguments(Arguments):
    '''
    Setting basic arguments for RFNet evaluation
    '''

    def __init__(self, **kwargs):
        super(EvaluationArguments, self).__init__(**kwargs)

        self.weight_path = kwargs.get('weight_path')  # path of the weight
        # whether to merge images and labels
        self.merge = kwargs.get('merge', True)
        self.save_predicted_image = kwargs.get(
            'save_predicted_image',
            False)  # whether to save the predicted images
        self.color_label_save_path = kwargs.get('color_label_save_path', os.path.join(
            BaseConfig.data_path_prefix, "inference_results/color"))  # path to save colored label images
        self.merge_label_save_path = kwargs.get('merge_label_save_path', os.path.join(
            BaseConfig.data_path_prefix, "inference_results/merge"))  # path to save merged label images
        self.label_save_path = kwargs.get("label_save_path", os.path.join(
            BaseConfig.data_path_prefix, "inference_results/label"))  # path to save label images
