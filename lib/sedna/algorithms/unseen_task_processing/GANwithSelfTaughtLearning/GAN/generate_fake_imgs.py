import torch

from models import Generator, weights_init

import matplotlib.pyplot as plt

import os

from collections import OrderedDict

import numpy as np

from skimage import io


device = 'cuda'

ngf = 64
nz = 256
im_size = 1024
netG = Generator(ngf=ngf, nz=nz, im_size=im_size).to(device)
weights_init(netG)
weights = torch.load(os.getcwd() + '/train_results/test1/models/50000.pth')
netG_weights = OrderedDict()
for name, weight in weights['g'].items():
    name = name.split('.')[1:]
    name = '.'.join(name)
    netG_weights[name] = weight
netG.load_state_dict(netG_weights)
current_batch_size = 1


index = 1
while index <= 3000:
    noise = torch.Tensor(current_batch_size, nz).normal_(0, 1).to(device)
    fake_images = netG(noise)[0]
    for fake_image in fake_images:
        fake_image = fake_image.detach().cpu().numpy().transpose(1, 2, 0)
        fake_image = fake_image * np.array([0.5, 0.5, 0.5])
        fake_image = fake_image + np.array([0.5, 0.5, 0.5])
        fake_image = (fake_image * 255).astype(np.uint8)
        io.imsave('../data/fake_imgs1/' + str(index) + '.png', fake_image)
        index += 1
