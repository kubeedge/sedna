from torchvision.utils import save_image
from models import Autoencoder
import torch
import cv2
import numpy as np
from torch.utils.data import DataLoader
import os
import torch.nn as nn
import torch.optim as optim
import csv
import time
import sys
from util import load_yaml


class DatasetAutoEncoder(torch.utils.data.Dataset):
    def __init__(self, fake_images_path):
        self.img_dir = fake_images_path

        self.new_img_w = 2048

        self.new_img_h = 1024

        self.examples = []

        file_names = os.listdir(fake_images_path)

        for file_name in file_names:
            img_path = fake_images_path + file_name
            self.examples.append({'img_path': img_path})

        self.examples = self.examples[0:60]

        self.num_examples = len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]

        img_path = example["img_path"]
        img = cv2.imread(img_path, -1)
        img = cv2.resize(img, (self.new_img_w, self.new_img_h),
                         interpolation=cv2.INTER_NEAREST)
        img = img / 255.0
        img = img - np.array([0.485, 0.456, 0.406])
        img = img / np.array([0.229, 0.224, 0.225])
        img = np.transpose(img, (2, 0, 1))

        img = img.astype(np.float32)

        img = torch.from_numpy(img)
        img = torch.Tensor(img)
        return img

    def __len__(self):
        return self.num_examples


def save_decoded_image(img, name):
    img = img.view(1, 3, 1024, 2048)
    save_image(img, name)


def train(unseen_samples):
    configs = load_yaml('../config.yaml')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    LEARNING_RATE = configs['STL'][1]['lr']
    NUM_EPOCHS = configs['STL'][0]['iter']
    batch_size = configs['STL'][2]['batch_size']
    name = configs['STL'][3]['name']
    save_dir = 'train_results/' + name
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    net = Autoencoder().to(device)
    # encoder_dataset = DatasetAutoEncoder(fake_images_path='../data/fake_imgs/')
    encoder_dataset = unseen_samples
    encoder_loader = DataLoader(
        dataset=encoder_dataset, batch_size=batch_size, drop_last=True)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    train_loss = []
    with open('train_loss1.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['epoch', 'loss'])
    for epoch in range(1, NUM_EPOCHS + 1):
        running_loss = 0.0
        for batch_idx, img in enumerate(encoder_loader):
            img = img.to(device)
            optimizer.zero_grad()
            outputs = net(img)
            loss = criterion(outputs, img)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        loss = running_loss / len(encoder_loader)
        train_loss.append(loss)
        with open('train_loss1.csv', 'a') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([epoch, loss])
        torch.save(net.encoder.state_dict(), save_dir +
                   '/encoder{}.pth'.format(epoch))
        save_decoded_image(img[0].cpu().data,
                           name=save_dir + '/original{}.png'.format(epoch))
        save_decoded_image(
            outputs[0].cpu().data, name=save_dir + '/decoded{}.png'.format(epoch))


if __name__ == '__main__':
    configs = load_yaml('../config.yaml')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    LEARNING_RATE = configs['STL'][1]['lr']
    NUM_EPOCHS = configs['STL'][0]['iter']
    batch_size = configs['STL'][2]['batch_size']
    name = configs['STL'][3]['name']
    save_dir = 'train_results/' + name
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    net = Autoencoder().to(device)
    encoder_dataset = DatasetAutoEncoder(fake_images_path='../data/fake_imgs/')
    encoder_loader = DataLoader(
        dataset=encoder_dataset, batch_size=batch_size, drop_last=True)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    train_loss = []
    with open('train_loss1.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['epoch', 'loss'])
    for epoch in range(1, NUM_EPOCHS + 1):
        running_loss = 0.0
        for batch_idx, img in enumerate(encoder_loader):
            img = img.to(device)
            optimizer.zero_grad()
            outputs = net(img)
            loss = criterion(outputs, img)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        loss = running_loss / len(encoder_loader)
        train_loss.append(loss)
        with open('train_loss1.csv', 'a') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([epoch, loss])
        torch.save(net.encoder.state_dict(), save_dir +
                   '/encoder{}.pth'.format(epoch))
        save_decoded_image(img[0].cpu().data,
                           name=save_dir + '/original{}.png'.format(epoch))
        save_decoded_image(
            outputs[0].cpu().data, name=save_dir + '/decoded{}.png'.format(epoch))
