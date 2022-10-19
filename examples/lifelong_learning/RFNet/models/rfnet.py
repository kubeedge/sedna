import torch.nn as nn
from itertools import chain # 串联多个迭代对象

from .util import _BNReluConv, upsample

class RFNet(nn.Module):
    def __init__(self, backbone, num_classes, use_bn=True):
        super(RFNet, self).__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        print(self.backbone.num_features)
        self.logits = _BNReluConv(self.backbone.num_features, self.num_classes, batch_norm=use_bn)

    def forward(self, rgb_inputs, depth_inputs = None):
        x, additional = self.backbone(rgb_inputs, depth_inputs)
        logits = self.logits.forward(x)
        logits_upsample = upsample(logits, rgb_inputs.shape[2:])
        #print(logits_upsample.size)
        return logits_upsample

    def random_init_params(self):
        return chain(*([self.logits.parameters(), self.backbone.random_init_params()]))

    def fine_tune_params(self):
        return self.backbone.fine_tune_params()

# class RFNet(nn.Module):
#     def __init__(self, trunk, num_classes, use_bn=True):
#         super(RFNet, self).__init__()
#         self.trunk = trunk
#         self.num_classes = num_classes
#         print(self.trunk.num_features)
#         self.logits = _BNReluConv(self.trunk.num_features, self.num_classes, batch_norm=use_bn)
#
#     def forward(self, rgb_inputs, depth_inputs = None):
#         x, additional = self.trunk(rgb_inputs, depth_inputs)
#         logits = self.logits.forward(x)
#         logits_upsample = upsample(logits, rgb_inputs.shape[2:])
#         #print(logits_upsample.size)
#         return logits_upsample
#
#     def random_init_params(self):
#         return chain(*([self.logits.parameters(), self.trunk.random_init_params()]))
#
#     def fine_tune_params(self):
#         return self.trunk.fine_tune_params()

