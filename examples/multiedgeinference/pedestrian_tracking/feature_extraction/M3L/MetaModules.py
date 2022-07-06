# Copyright 2021 The KubeEdge Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
from torch.autograd import Variable
import math
import numpy as np


def to_var(x, requires_grad=True):
    if torch.cuda.is_available(): x = x.cuda()
    return Variable(x, requires_grad=requires_grad)


class MetaModule(nn.Module):
    # adopted from: Adrien Ecoffet https://github.com/AdrienLE
    def params(self):
        for name, param in self.named_params(self):
            yield param

    def named_leaves(self):
        return []

    def named_submodules(self):
        return []

    def named_params(self, curr_module=None, memo=None, prefix=''):
        if memo is None:
            memo = set()

        if hasattr(curr_module, 'named_leaves'):
            for name, p in curr_module.named_leaves():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p
        else:
            for name, p in curr_module._parameters.items():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p

        for mname, module in curr_module.named_children():
            submodule_prefix = prefix + ('.' if prefix else '') + mname
            for name, p in self.named_params(module, memo, submodule_prefix):
                yield name, p

    def update_params(self, lr_inner, source_params=None,
                      solver='sgd', beta1=0.9, beta2=0.999, weight_decay=5e-4):
        if solver == 'sgd':
            for tgt, src in zip(self.named_params(self), source_params):
                name_t, param_t = tgt
                grad = src if src is not None else 0
                tmp = param_t - lr_inner * grad
                self.set_param(self, name_t, tmp)
        elif solver == 'adam':
            for tgt, gradVal in zip(self.named_params(self), source_params):
                name_t, param_t = tgt
                exp_avg, exp_avg_sq = torch.zeros_like(param_t.data), \
                                      torch.zeros_like(param_t.data)
                bias_correction1 = 1 - beta1
                bias_correction2 = 1 - beta2
                gradVal.add_(weight_decay, param_t)
                exp_avg.mul_(beta1).add_(1 - beta1, gradVal)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, gradVal, gradVal)
                exp_avg_sq.add_(1e-8)  # to avoid possible nan in backward
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(1e-8)
                step_size = lr_inner / bias_correction1
                newParam = param_t.addcdiv(-step_size, exp_avg, denom)
                self.set_param(self, name_t, newParam)

    def setParams(self, params):
        for tgt, param in zip(self.named_params(self), params):
            name_t, _ = tgt
            self.set_param(self, name_t, param)

    def set_param(self, curr_mod, name, param):
        if '.' in name:
            n = name.split('.')
            module_name = n[0]
            rest = '.'.join(n[1:])
            for name, mod in curr_mod.named_children():
                if module_name == name:
                    self.set_param(mod, rest, param)
                    break
        else:
            setattr(curr_mod, name, param)

    def setBN(self, inPart, name, param):
        if '.' in name:
            part = name.split('.')
            self.setBN(getattr(inPart, part[0]), '.'.join(part[1:]), param)
        else:
            setattr(inPart, name, param)

    def detach_params(self):
        for name, param in self.named_params(self):
            self.set_param(self, name, param.detach())

    def copyModel(self, newModel, same_var=False):
        # copy meta model to meta model
        tarName = list(map(lambda v: v, newModel.state_dict().keys()))

        # requires_grad
        partName, partW = list(map(lambda v: v[0], newModel.named_params(newModel))), list(
            map(lambda v: v[1], newModel.named_params(newModel)))  # new model's weight

        metaName, metaW = list(map(lambda v: v[0], self.named_params(self))), list(
            map(lambda v: v[1], self.named_params(self)))
        bnNames = list(set(tarName) - set(partName))

        # copy vars
        for name, param in zip(metaName, partW):
            if not same_var:
                param = to_var(param.data.clone(), requires_grad=True)
            self.set_param(self, name, param)
        # copy training mean var
        tarName = newModel.state_dict()
        for name in bnNames:
            param = to_var(tarName[name], requires_grad=False)
            self.setBN(self, name, param)

    def copyWeight(self, modelW):
        # copy state_dict to buffers
        curName = list(map(lambda v: v[0], self.named_params(self)))
        tarNames = set()
        for name in modelW.keys():
            # print(name)
            if name.startswith("module"):
                tarNames.add(".".join(name.split(".")[1:]))
            else:
                tarNames.add(name)
        # bnNames = list(tarNames - set(curName))
        for tgt in self.named_params(self):
            name_t, param_t = tgt
            # print(name_t)
            module_name_t = 'module.' + name_t
            if name_t in modelW:
                param = to_var(modelW[name_t], requires_grad=True)
                self.set_param(self, name_t, param)
            elif module_name_t in modelW:
                param = to_var(modelW['module.' + name_t], requires_grad=True)
                self.set_param(self, name_t, param)
            else:
                continue


    def load_param(self, path):
        modelW = torch.load(path, map_location=torch.device('cpu'))['state_dict']
        print("=> Loaded M3L ReID model  '{}'".format(path))

        # copy state_dict to buffers
        curName = list(map(lambda v: v[0], self.named_params(self)))
        tarNames = set()
        for name in modelW.keys():
            # print(name)
            if name.startswith("module"):
                tarNames.add(".".join(name.split(".")[1:]))
            else:
                tarNames.add(name)
        bnNames = list(tarNames - set(curName))  ## in BN resMeta bnNames only contains running var/mean
        for tgt in self.named_params(self):
            name_t, param_t = tgt
            # print(name_t)
            module_name_t = 'module.' + name_t
            if name_t in modelW:
                param = to_var(modelW[name_t], requires_grad=True)
                self.set_param(self, name_t, param)
            elif module_name_t in modelW:
                param = to_var(modelW['module.' + name_t], requires_grad=True)
                self.set_param(self, name_t, param)
            else:
                continue

        for name in bnNames:
            try:
                param = to_var(modelW[name], requires_grad=False)
            except:
                param = to_var(modelW['module.' + name], requires_grad=False)
            self.setBN(self, name, param)




class MetaConv2d(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.Conv2d(*args, **kwargs)
        self.in_channels = ignore.in_channels
        self.out_channels = ignore.out_channels
        self.stride = ignore.stride
        self.padding = ignore.padding
        self.dilation = ignore.dilation
        self.groups = ignore.groups
        self.kernel_size = ignore.kernel_size

        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))

        if ignore.bias is not None:
            self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))
        else:
            self.register_buffer('bias', None)

    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]


class MetaBatchNorm2d(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.BatchNorm2d(*args, **kwargs)

        self.num_features = ignore.num_features
        self.eps = ignore.eps
        self.momentum = ignore.momentum
        self.affine = ignore.affine
        self.track_running_stats = ignore.track_running_stats

        if self.affine:
            self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
            self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))

        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(self.num_features))
            self.register_buffer('running_var', torch.ones(self.num_features))
            self.register_buffer('num_batches_tracked', torch.LongTensor([0]).squeeze())
        else:
            self.register_buffer('running_mean', None)
            self.register_buffer('running_var', None)
            self.register_buffer('num_batches_tracked', None)

    def forward(self, x):
        val2 = self.weight.sum()
        res = F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias,
                           self.training or not self.track_running_stats, self.momentum, self.eps)
        return res

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]


class MetaBatchNorm1d(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.BatchNorm1d(*args, **kwargs)

        self.num_features = ignore.num_features
        self.eps = ignore.eps
        self.momentum = ignore.momentum
        self.affine = ignore.affine
        self.track_running_stats = ignore.track_running_stats

        if self.affine:
            self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
            self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))

        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(self.num_features))
            self.register_buffer('running_var', torch.ones(self.num_features))
            self.register_buffer('num_batches_tracked', torch.LongTensor([0]).squeeze())
        else:
            self.register_buffer('running_mean', None)
            self.register_buffer('running_var', None)
            self.register_buffer('num_batches_tracked', None)

    def forward(self, x):
        return F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias,
                            self.training or not self.track_running_stats, self.momentum, self.eps)
        ## meta test set this one to False self.training or not self.track_running_stats
    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]


class MetaInstanceNorm2d(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.InstanceNorm2d(*args, **kwargs)

        self.num_features = ignore.num_features
        self.eps = ignore.eps
        self.momentum = ignore.momentum
        self.affine = ignore.affine
        self.track_running_stats = ignore.track_running_stats

        if self.affine:
            self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
            self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))
        else:
            self.register_buffer('weight', None)
            self.register_buffer('bias', None)

        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(self.num_features))
            self.register_buffer('running_var', torch.ones(self.num_features))
            self.register_buffer('num_batches_tracked', torch.LongTensor([0]).squeeze())
        else:
            self.register_buffer('running_mean', None)
            self.register_buffer('running_var', None)
            self.register_buffer('num_batches_tracked', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.affine:
            init.constant_(self.weight, 1)
            init.constant_(self.bias, 0)

    def forward(self, x):

        res = F.instance_norm(x, self.running_mean, self.running_var, self.weight, self.bias,
            self.training or not self.track_running_stats, self.momentum, self.eps)
        return res

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]

class MixUpBatchNorm1d(MetaBatchNorm1d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True):
        super(MixUpBatchNorm1d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)

        self.register_buffer('meta_mean1', torch.zeros(self.num_features))
        self.register_buffer('meta_var1', torch.zeros(self.num_features))
        self.register_buffer('meta_mean2', torch.zeros(self.num_features))
        self.register_buffer('meta_var2', torch.zeros(self.num_features))
        self.device_count = torch.cuda.device_count()

    def forward(self, input, MTE='', save_index=0):
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training:
            if MTE == 'sample':
                from torch.distributions.normal import Normal
                Distri1 = Normal(self.meta_mean1, self.meta_var1)
                Distri2 = Normal(self.meta_mean2, self.meta_var2)
                sample1 = Distri1.sample([input.size(0), ])
                sample2 = Distri2.sample([input.size(0), ])
                lam = np.random.beta(1., 1.)
                inputmix1 = lam * sample1 + (1-lam) * input
                inputmix2 = lam * sample2 + (1-lam) * input

                mean1 = inputmix1.mean(dim=0)
                var1 = inputmix1.var(dim=0, unbiased=False)
                mean2 = inputmix2.mean(dim=0)
                var2 = inputmix2.var(dim=0, unbiased=False)

                output1 = (inputmix1 - mean1[None, :]) / (torch.sqrt(var1[None, :] + self.eps))
                output2 = (inputmix2 - mean2[None, :]) / (torch.sqrt(var2[None, :] + self.eps))
                if self.affine:
                    output1 = output1 * self.weight[None, :] + self.bias[None, :]
                    output2 = output2 * self.weight[None, :] + self.bias[None, :]
                return [output1, output2]

            else:
                mean = input.mean(dim=0)
                # use biased var in train
                var = input.var(dim=0, unbiased=False)
                n = input.numel() / input.size(1)

                with torch.no_grad():
                    running_mean = exponential_average_factor * mean \
                                   + (1 - exponential_average_factor) * self.running_mean
                    # update running_var with unbiased var
                    running_var = exponential_average_factor * var * n / (n - 1) \
                                  + (1 - exponential_average_factor) * self.running_var
                    self.running_mean.copy_(running_mean)
                    self.running_var.copy_(running_var)
                    if save_index == 1:
                        self.meta_mean1.copy_(mean)
                        self.meta_var1.copy_(var)
                    elif save_index == 2:
                        self.meta_mean2.copy_(mean)
                        self.meta_var2.copy_(var)

        else:
            mean = self.running_mean
            var = self.running_var

        input = (input - mean[None, :]) / (torch.sqrt(var[None, :] + self.eps))
        if self.affine:
            input = input * self.weight[None, :] + self.bias[None, :]

        return input
