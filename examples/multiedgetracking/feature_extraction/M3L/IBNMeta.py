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
from torchvision.models import resnet50
from .MetaModules import *


class IBN(nn.Module):
    def __init__(self, planes):
        super(IBN, self).__init__()
        half1 = int(planes / 2)
        self.half = half1
        half2 = planes - half1
        self.IN = MetaInstanceNorm2d(half1, affine=True)
        self.BN = MetaBatchNorm2d(half2)

    def forward(self, x):
        split = torch.split(x, self.half, 1)
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())
        out = torch.cat((out1, out2), 1)
        return out


class Bottleneck_IBN(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, ibn=False, stride=1, downsample=None):
        super(Bottleneck_IBN, self).__init__()
        self.conv1 = MetaConv2d(inplanes, planes, kernel_size=1, bias=False)
        if ibn:
            self.bn1 = IBN(planes)
        else:
            self.bn1 = MetaBatchNorm2d(planes)
        self.conv2 = MetaConv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = MetaBatchNorm2d(planes)
        self.conv3 = MetaConv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = MetaBatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class MetaResNet_IBN_a_base(MetaModule):

    def __init__(self, layers, block=Bottleneck_IBN):
        scale = 64
        self.inplanes = scale
        super(MetaResNet_IBN_a_base, self).__init__()
        self.conv1 = MetaConv2d(3, scale, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = MetaBatchNorm2d(scale)
        # self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, scale, layers[0])
        self.layer2 = self._make_layer(block, scale * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, scale * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, scale * 8, layers[3], stride=2)
        # self.avgpool = nn.AvgPool2d(7)
        # self.fc = nn.Linear(scale * 8 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, MetaConv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, MetaBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, MetaInstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                MetaConv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                MetaBatchNorm2d(planes * block.expansion),
            )

        layers = []
        ibn = True
        if planes == 512:
            ibn = False
        layers.append(block(self.inplanes, planes, ibn, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, ibn))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        # x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)

        return x


class MetaIBN(MetaModule):
    def __init_with_imagenet(self, baseModel):
        model = resnet50(pretrained=False)
        del model.fc
        baseModel.copyWeight(model.state_dict())

    def getBase(self):
        baseModel = MetaResNet_IBN_a_base([3, 4, 6, 3])
        self.__init_with_imagenet(baseModel)
        return baseModel

    def __init__(self, num_features=0, dropout=0, cut_at_pooling=False, norm=True, num_classes=[0,0,0], BNNeck=False) -> object:
        super(MetaIBN, self).__init__()
        self.num_features = num_features
        self.dropout = dropout
        self.cut_at_pooling = cut_at_pooling
        self.num_classes1 = num_classes[0]
        self.num_classes2 = num_classes[1]
        self.num_classes3 = num_classes[2]
        self.has_embedding = num_features > 0
        self.norm = norm
        self.BNNeck = BNNeck
        if self.dropout > 0:
            self.drop = nn.Dropout(self.dropout)
        # Construct base (pretrained) resnet
        self.base = self.getBase()
        self.base.layer4[0].conv2.stride = (1, 1)
        self.base.layer4[0].downsample[0].stride = (1, 1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        out_planes = 2048
        if self.has_embedding:
            self.feat = MetaLinear(out_planes, self.num_features)
            init.kaiming_normal_(self.feat.weight, mode='fan_out')
            init.constant_(self.feat.bias, 0)
        else:
            # Change the num_features to CNN output channels
            self.num_features = out_planes

        self.feat_bn = MixUpBatchNorm1d(self.num_features)
        init.constant_(self.feat_bn.weight, 1)
        init.constant_(self.feat_bn.bias, 0)

        self.reset_IN()

    def reset_IN(self):
        for m in self.modules():
            if isinstance(m, MetaInstanceNorm2d):
                if m.affine:
                    init.constant_(m.weight, 1)
                    init.constant_(m.bias, 0)

    def forward(self, x, MTE='', save_index=0):
        x= self.base(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)

        if self.cut_at_pooling:
            return x

        if self.has_embedding:
            bn_x = self.feat_bn(self.feat(x))
        else:
            bn_x= self.feat_bn(x, MTE, save_index)
        tri_features = x

        if self.training is False:
            bn_x = F.normalize(bn_x)
            return bn_x

        if isinstance(bn_x, list):
            output = []
            for bnfeature in bn_x:
                if self.norm:
                    bnfeature = F.normalize(bnfeature)
                output.append(bnfeature)
            if self.BNNeck:
                return output, tri_features
            else:
                return output

        if self.norm:
            bn_x = F.normalize(bn_x)
        elif self.has_embedding:
            bn_x = F.relu(bn_x)

        if self.dropout > 0:
            bn_x = self.drop(bn_x)

        if self.BNNeck:
            return bn_x, tri_features
        else:
            return bn_x


