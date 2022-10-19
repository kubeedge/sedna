import torch.nn as nn
from itertools import chain # 串联多个迭代对象

from .util import _BNReluConv, upsample


class RFNet(nn.Module):
    def __init__(self, backbone, num_classes, use_bn=True):
        super(RFNet, self).__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        #self.bottleneck = _BNReluConv(self.backbone.num_features, 128, k = 3, batch_norm=use_bn)
        #self.logits = _BNReluConv(128, self.num_classes+1, k = 1, batch_norm=use_bn)
        self.logits = _BNReluConv(self.backbone.num_features, self.num_classes, batch_norm=use_bn)
        #self.logits_target = _BNReluConv(self.backbone.num_features, self.num_classes, batch_norm=use_bn)
        self.logits_aux = _BNReluConv(self.backbone.num_features, self.num_classes, batch_norm=use_bn)

    def forward(self, rgb_inputs, depth_inputs = None):
        x, additional, da_features = self.backbone(rgb_inputs, depth_inputs)
        #print(additional['features'][0].shape)
        #bottleneck = self.bottleneck(x)
        logits = self.logits.forward(x)
        logits_aux = self.logits_aux.forward(x)
        #print(logits_aux.shape)
        logits_upsample = upsample(logits, rgb_inputs.shape[2:])
        logits_aux_upsample = upsample(logits_aux, rgb_inputs.shape[2:])
        return logits_upsample, logits_aux_upsample, da_features

    def random_init_params(self):
        return chain(*([self.logits.parameters(), self.logits_aux.parameters(), self.backbone.random_init_params()]))

    def fine_tune_params(self):
        return self.backbone.fine_tune_params()
