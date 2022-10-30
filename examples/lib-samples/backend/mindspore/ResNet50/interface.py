# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""train resnet."""
import os
import numpy as np
from mindspore import context
from mindspore import Tensor
from mindspore.nn.optim.momentum import Momentum
from mindspore.train.model import Model
from mindspore.context import ParallelMode
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.parallel import set_algo_parameters
import mindspore.nn as nn
import mindspore.common.initializer as weight_init
from src.lr_generator import get_lr

from src.resnet import resnet50 as resnet
from src.config import config1 as config
from src.dataset import create_dataset1 as create_dataset


class Estimator:
    def __init__(self) -> None:
        self.has_load = False
        self.network = None

    def train(self, train_data, **kwargs):
        """The whole process of model training

        The training process of the resnet model. At present, it supports single NPU and CPU.
        Multi-GPU and multi-NPU will be supported in the future.

        Args:
            train_data: training dataset path
            kwargs: Including args_opt and other parameters. args_opt is passed by train.py,
                    includes some key parameters

        """
        args_opt = kwargs.get("args_opt")
        target = args_opt.device_target
        if target == "CPU":
            args_opt.run_distribute = False

        ckpt_save_dir = args_opt.model_save_path

        # Multi-GPU/Multi-NPU
        if args_opt.run_distribute:
            if target == "Ascend":
                device_id = int(os.getenv('DEVICE_ID'))
                context.set_context(
                    device_id=device_id,
                    enable_auto_mixed_precision=True)
                context.set_auto_parallel_context(
                    device_num=args_opt.device_num,
                    parallel_mode=ParallelMode.DATA_PARALLEL,
                    gradients_mean=True)
                set_algo_parameters(elementwise_op_strategy_follow=True)
                context.set_auto_parallel_context(
                    all_reduce_fusion_config=[85, 160])
                init()
            # GPU target
            else:
                init()
                context.set_auto_parallel_context(
                    device_num=get_group_size(),
                    parallel_mode=ParallelMode.DATA_PARALLEL,
                    gradients_mean=True)
                if args_opt.net == "resnet50":
                    context.set_auto_parallel_context(
                        all_reduce_fusion_config=[85, 160])
            ckpt_save_dir = args_opt.save_checkpoint_path + \
                "ckpt_" + str(get_rank()) + "/"

        # create dataset
        dataset = create_dataset(
            dataset_path=train_data,
            do_train=True,
            repeat_num=1,
            batch_size=config.batch_size,
            target=target,
            distribute=args_opt.run_distribute)
        step_size = dataset.get_dataset_size()

        # define net
        net = resnet(class_num=config.class_num)

        # init weight
        if args_opt.pre_trained:
            param_dict = load_checkpoint(args_opt.pre_trained)
            load_param_into_net(net, param_dict)
        else:
            for _, cell in net.cells_and_names():
                if isinstance(cell, nn.Conv2d):
                    cell.weight.set_data(
                        weight_init.initializer(
                            weight_init.XavierUniform(),
                            cell.weight.shape,
                            cell.weight.dtype))
                if isinstance(cell, nn.Dense):
                    cell.weight.set_data(
                        weight_init.initializer(
                            weight_init.TruncatedNormal(),
                            cell.weight.shape,
                            cell.weight.dtype))

        # init lr
        lr = get_lr(
            lr_init=config.lr_init,
            lr_end=config.lr_end,
            lr_max=config.lr_max,
            warmup_epochs=config.warmup_epochs,
            total_epochs=config.epoch_size,
            steps_per_epoch=step_size,
            lr_decay_mode=config.lr_decay_mode)
        lr = Tensor(lr)

        # define opt
        decayed_params = []
        no_decayed_params = []
        for param in net.trainable_params():
            if 'beta' not in param.name and 'gamma' not in param.name and 'bias' not in param.name:
                decayed_params.append(param)
            else:
                no_decayed_params.append(param)

        group_params = [{'params': decayed_params, 'weight_decay': config.weight_decay},
                        {'params': no_decayed_params},
                        {'order_params': net.trainable_params()}]
        opt = Momentum(
            group_params,
            lr,
            config.momentum,
            loss_scale=config.loss_scale)

        # define loss, model
        if target == "Ascend":
            loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
            loss_scale = FixedLossScaleManager(
                config.loss_scale, drop_overflow_update=False)
            model = Model(
                net,
                loss_fn=loss,
                optimizer=opt,
                loss_scale_manager=loss_scale,
                metrics={'acc'},
                amp_level="O2",
                keep_batchnorm_fp32=False)
        else:
            # GPU and CPU target
            loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")

            if target != "CPU":
                opt = Momentum(
                    filter(
                        lambda x: x.requires_grad,
                        net.get_parameters()),
                    lr,
                    config.momentum,
                    config.weight_decay,
                    config.loss_scale)
                loss_scale = FixedLossScaleManager(
                    config.loss_scale, drop_overflow_update=False)
                # Mixed precision
                model = Model(
                    net,
                    loss_fn=loss,
                    optimizer=opt,
                    loss_scale_manager=loss_scale,
                    metrics={'acc'},
                    amp_level="O2",
                    keep_batchnorm_fp32=False)
            else:
                # fp32 training
                opt = Momentum(
                    filter(
                        lambda x: x.requires_grad,
                        net.get_parameters()),
                    lr,
                    config.momentum,
                    config.weight_decay)
                model = Model(
                    net,
                    loss_fn=loss,
                    optimizer=opt,
                    metrics={'acc'})

        # define callbacks
        time_cb = TimeMonitor(data_size=step_size)
        loss_cb = LossMonitor()
        cb = [time_cb, loss_cb]
        if config.save_checkpoint:
            config_ck = CheckpointConfig(
                save_checkpoint_steps=config.save_checkpoint_epochs * step_size,
                keep_checkpoint_max=config.keep_checkpoint_max)
            ckpt_cb = ModelCheckpoint(
                prefix="resnet",
                directory=ckpt_save_dir,
                config=config_ck)
            cb += [ckpt_cb]

        # train model
        dataset_sink_mode = target != "CPU"
        model.train(
            config.epoch_size - config.pretrain_epoch_size,
            dataset,
            callbacks=cb,
            sink_size=dataset.get_dataset_size(),
            dataset_sink_mode=dataset_sink_mode)

    def evaluate(self, valid_data, **kwargs):
        """The whole process of model evaluation.

        The evaluation process of the resnet model. At present, it supports single NPU and CPU.
        GPU will be supported in the future.

        Args:
            valid_data: evaluation dataset path.
            kwargs: Including args_opt and other parameters. args_opt is passed by eval.py,
                    includes some key parameters.

        """

        args_opt = kwargs.get("args_opt")
        target = args_opt.device_target
        if target == "Ascend":
            device_id = int(os.getenv('DEVICE_ID'))
            context.set_context(device_id=device_id)

        # create dataset
        dataset = create_dataset(
            dataset_path=valid_data,
            do_train=False,
            batch_size=config.batch_size,
            target=target)

        # define net
        net = self.network

        # define loss, model
        loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

        # define model
        model = Model(
            net,
            loss_fn=loss,
            metrics={
                'top_1_accuracy',
                'top_5_accuracy'})

        # eval model
        res = model.eval(dataset)
        print("result:", res, "ckpt=", args_opt.checkpoint_path)

    def predict(self, data):
        """Inference for the image data

        Infer the image data and output its category

        Args:
            data: image to be inferred
        """

        class_name = [
            'airplane',
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck"]

        # define model
        model = Model(self.network)

        # infer data
        res = model.predict(data)

        # The output of the model is the score of each category, which needs to be softmax.
        softmax = nn.Softmax()
        # get label result
        pred_class = class_name[np.argmax(softmax(res[0]))]

        print("This image belongs to: ", pred_class)
        return pred_class

    def load(self, model_url):
        """load checkpoint into model

        Initialize resnet model, and load the specified model file for evaluation and inference

        Args:
            model_url: Url of model file
        """

        print("load model url: ", model_url)
        self.network = resnet(class_num=config.class_num)
        param_dict = load_checkpoint(model_url)
        load_param_into_net(self.network, param_dict)
        self.network.set_train(False)
        self.has_load = True
