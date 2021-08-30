import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision import models


class AlexNetConv4(nn.Module):
    def __init__(self, break_point):
        super(AlexNetConv4, self).__init__()
        original_model = models.alexnet(pretrained=True)
        self.features = nn.Sequential(
            # stop at conv4
            *list(original_model.features.children())[:break_point]
        )

    def forward(self, x):
        x = self.features(x)
        return x


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x


class AlexNetConv5(nn.Module):
    def __init__(self, break_point):
        super(AlexNetConv5, self).__init__()
        original_model = models.alexnet(pretrained=True)
        self.features = nn.Sequential(
            # start at conv5
            *(list(original_model.features.children())[break_point:] + [nn.AvgPool2d(1), Flatten()] + list(
                original_model.classifier.children()))

        )

    def forward(self, x):
        x = self.features(x)
        return x