from util import load_yaml
import sys
import lpips
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torchvision import utils as vutils

import argparse
import random
from tqdm import tqdm

import csv

from models import weights_init, Discriminator, Generator
from operation import copy_G_params, load_params, get_dir
from operation import ImageFolder, InfiniteSamplerWrapper
from diffaug import DiffAugment
policy = 'color,translation'
percept = lpips.PerceptualLoss(model='net-lin', net='vgg', use_gpu=True)


def crop_image_by_part(image, part):
    hw = image.shape[2] // 2
    if part == 0:
        return image[:, :, :hw, :hw]
    if part == 1:
        return image[:, :, :hw, hw:]
    if part == 2:
        return image[:, :, hw:, :hw]
    if part == 3:
        return image[:, :, hw:, hw:]


def train_d(net, data, label="real"):
    """Train function of discriminator"""
    if label == "real":
        part = random.randint(0, 3)
        pred, [rec_all, rec_small, rec_part] = net(data, label, part=part)
        err = F.relu(torch.rand_like(pred) * 0.2 + 0.8 - pred).mean() + \
            percept(rec_all, F.interpolate(data, rec_all.shape[2])).sum() + \
            percept(rec_small, F.interpolate(data, rec_small.shape[2])).sum() + \
            percept(rec_part, F.interpolate(
                crop_image_by_part(data, part), rec_part.shape[2])).sum()
        err.backward()
        return pred.mean().item(), rec_all, rec_small, rec_part
    else:
        pred = net(data, label)
        err = F.relu(torch.rand_like(pred) * 0.2 + 0.8 + pred).mean()
        err.backward()
        return pred.mean().item()


def train(args):
    data_root = args['path']
    total_iterations = args['iter']
    # checkpoint = args.ckpt
    batch_size = args['batch_size']
    im_size = args['im_size']
    ndf = 64
    ngf = 64
    nz = 256
    nlr = 0.0002
    nbeta1 = 0.5
    use_cuda = True
    multi_gpu = False
    dataloader_workers = 8
    current_iteration = 0
    save_interval = 100
    saved_model_folder, saved_image_folder = get_dir(args)

    device = torch.device("cpu")
    if use_cuda:
        device = torch.device("cuda:0")

    transform_list = [
        transforms.Resize((int(im_size), int(im_size))),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ]
    trans = transforms.Compose(transform_list)

    if 'lmdb' in data_root:
        from operation import MultiResolutionDataset
        dataset = MultiResolutionDataset(data_root, trans, 1024)
    else:
        dataset = ImageFolder(root=data_root, transform=trans)

    dataloader = iter(DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                 sampler=InfiniteSamplerWrapper(dataset), num_workers=dataloader_workers,
                                 pin_memory=True))
    netG = Generator(ngf=ngf, nz=nz, im_size=im_size)
    netG.apply(weights_init)

    netD = Discriminator(ndf=ndf, im_size=im_size)
    netD.apply(weights_init)

    netG.to(device)
    netD.to(device)

    avg_param_G = copy_G_params(netG)

    fixed_noise = torch.FloatTensor(8, nz).normal_(0, 1).to(device)

    optimizerG = optim.Adam(netG.parameters(), lr=nlr, betas=(nbeta1, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=nlr, betas=(nbeta1, 0.999))

    if multi_gpu:
        netG = nn.DataParallel(netG.to(device))
        netD = nn.DataParallel(netD.to(device))

    with open('train_cityscape.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['epoch', 'd_loss', 'g_loss'])

    for iteration in tqdm(range(current_iteration, total_iterations + 1)):
        real_image = next(dataloader)
        real_image = real_image.to(device)
        current_batch_size = real_image.size(0)
        noise = torch.Tensor(current_batch_size, nz).normal_(0, 1).to(device)

        fake_images = netG(noise)

        real_image = DiffAugment(real_image, policy=policy)
        fake_images = [DiffAugment(fake, policy=policy)
                       for fake in fake_images]

        # 2. train Discriminator
        netD.zero_grad()

        err_dr, rec_img_all, rec_img_small, rec_img_part = train_d(
            netD, real_image, label="real")
        train_d(netD, [fi.detach() for fi in fake_images], label="fake")
        optimizerD.step()

        # 3. train Generator
        netG.zero_grad()
        pred_g = netD(fake_images, "fake")
        err_g = -pred_g.mean()

        err_g.backward()
        optimizerG.step()

        for p, avg_p in zip(netG.parameters(), avg_param_G):
            avg_p.mul_(0.999).add_(0.001 * p.data)

        if iteration % (save_interval * 10) == 0:
            backup_para = copy_G_params(netG)
            load_params(netG, avg_param_G)
            with torch.no_grad():
                vutils.save_image(netG(fixed_noise)[0].add(1).mul(0.5), saved_image_folder + '/%d.jpg' % iteration,
                                  nrow=4)
                vutils.save_image(torch.cat([
                    F.interpolate(real_image, 128),
                    rec_img_all, rec_img_small,
                    rec_img_part]).add(1).mul(0.5), saved_image_folder + '/rec_%d.jpg' % iteration)
            load_params(netG, backup_para)

        if iteration % (save_interval * 50) == 0 or iteration == total_iterations:
            backup_para = copy_G_params(netG)
            load_params(netG, avg_param_G)
            torch.save({'g': netG.state_dict(), 'd': netD.state_dict()},
                       saved_model_folder + '/%d.pth' % iteration)
            load_params(netG, backup_para)
            torch.save({'g': netG.state_dict(),
                        'd': netD.state_dict(),
                        'g_ema': avg_param_G,
                        'opt_g': optimizerG.state_dict(),
                        'opt_d': optimizerD.state_dict()}, saved_model_folder + '/all_%d.pth' % iteration)

        with open('train_cityscape.csv', 'a') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([iteration, err_dr, -err_g.item()])


if __name__ == "__main__":
    configs = load_yaml('../config.yaml')
    args = dict()
    args['path'] = configs['dataset_path']
    args['iter'] = configs['GAN'][0]['iter']
    args['batch_size'] = configs['GAN'][1]['batch_size']
    args['im_size'] = configs['GAN'][2]['im_size']
    args['name'] = configs['GAN'][3]['name']
    train(args)
