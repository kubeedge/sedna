import argparse
import os
import numpy as np
from tqdm import tqdm
import time
import torch
from torchvision.transforms import ToPILImage
from PIL import Image
import cv2

from dataloaders import make_data_loader
from dataloaders.utils import decode_seg_map_sequence, Colorize
from utils.metrics import Evaluator
from models.rfnet import RFNet
from models import rfnet_for_unseen
from models.resnet.resnet_single_scale_single_attention import *
from models.resnet import resnet_single_scale_single_attention_unseen
import torch.backends.cudnn as cudnn

class Validator(object):
    def __init__(self, args, data=None, unseen_detection=False):
        self.args = args
        self.time_train = []
        self.num_class = args.num_class

        # Define Dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': False}
        # _, self.val_loader, _, self.custom_loader, self.num_class = make_data_loader(args, **kwargs)
        _, _, self.test_loader = make_data_loader(args, test_data=data, **kwargs)
        print('num_classes:', str(self.num_class))

        # Define evaluator
        self.evaluator = Evaluator(self.num_class)

        # Define network
        self.resnet = resnet18(pretrained=False, efficient=False, use_bn=True)
        self.model = RFNet(self.resnet, num_classes=self.num_class, use_bn=True)

        if args.cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            self.model = self.model.cuda()
            self.model.to(f'cuda:{self.args.gpu_ids[0]}')
            cudnn.benchmark = True  # accelarate speed
        print('Model loaded successfully!')
        
        # load model
        if self.args.weight_path is not None and os.path.exists(self.args.weight_path):
            self.new_state_dict = torch.load(self.args.weight_path, map_location=torch.device("cpu"))
            self.model = load_my_state_dict(self.model, self.new_state_dict['state_dict'])

    def validate(self):        
        self.model.eval()
        self.evaluator.reset()
        # tbar = tqdm(self.test_loader, desc='\r')
        predictions = []
        for sample, image_name in self.test_loader:
            if self.args.depth:
                image, depth, target = sample['image'], sample['depth'], sample['label']
            else:
                image, target = sample['image'], sample['label']
            
            if self.args.cuda:
                image = image.cuda()
                if self.args.depth:
                    depth = depth.cuda()
                    
            with torch.no_grad():
                if self.args.depth:
                    output = self.model(image, depth)
                else:
                    output = self.model(image)
                    
            if self.args.cuda:
                torch.cuda.synchronize()

            pred = output.data.cpu().numpy()
            # todo
            pred = np.argmax(pred, axis=1)
            predictions.append(pred)

            if not self.args.save_predicted_image:
                continue
            
            pre_colors = Colorize(n=self.args.num_class)(torch.max(output, 1)[1].detach().cpu().byte())
            pre_labels = torch.max(output, 1)[1].detach().cpu().byte()
            # save
            for i in range(pre_colors.shape[0]):
                if not image_name[0]:
                    img_name = f"test_{time.time()}.png"
                else:
                    img_name = os.path.basename(image_name[0])

                color_label_name = os.path.join(self.args.color_label_save_path, img_name)
                label_name = os.path.join(self.args.label_save_path, img_name)
                merge_label_name = os.path.join(self.args.merge_label_save_path, img_name)

                os.makedirs(os.path.dirname(color_label_name), exist_ok=True)
                os.makedirs(os.path.dirname(merge_label_name), exist_ok=True)
                os.makedirs(os.path.dirname(label_name), exist_ok=True)

                pre_color_image = ToPILImage()(pre_colors[i])  # pre_colors.dtype = float64
                # color = paint_trapezoid(np.array(pre_color_image))
                # cv2.imwrite(color_label_name, color)
                pre_color_image.save(color_label_name)

                pre_label_image = ToPILImage()(pre_labels[i])
                pre_label_image.save(label_name)

                if (self.args.merge):
                    image_merge(image[i], pre_color_image, merge_label_name)
                    print('save image: {}'.format(merge_label_name))
            
        return predictions
  
    def task_divide(self):
        seen_task_samples, unseen_task_samples = [], []
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.test_loader, desc='\r')
        for i, (sample, image_name) in enumerate(tbar):

            if self.args.depth:
                image, depth, target = sample['image'], sample['depth'], sample['label']
            else:
                image, target = sample['image'], sample['label']
            if self.args.cuda:
                image = image.cuda()
                if self.args.depth:
                    depth = depth.cuda()
            start_time = time.time()
            with torch.no_grad():
                if self.args.depth:
                    output_, output, _ = self.model(image, depth)
                else:
                    output_, output, _ = self.model(image)
            if self.args.cuda:
                torch.cuda.synchronize()
            if i != 0:
                fwt = time.time() - start_time
                self.time_train.append(fwt)
                print("Forward time per img (bath size=%d): %.3f (Mean: %.3f)" % (
                    self.args.val_batch_size, fwt / self.args.val_batch_size,
                    sum(self.time_train) / len(self.time_train) / self.args.val_batch_size))
            time.sleep(0.1)  # to avoid overheating the GPU too much

            # pred colorize
            pre_colors = Colorize()(torch.max(output, 1)[1].detach().cpu().byte())
            pre_labels = torch.max(output, 1)[1].detach().cpu().byte()
            for i in range(pre_colors.shape[0]):
                task_sample = dict()
                task_sample.update(image=sample["image"][i])
                task_sample.update(label=sample["label"][i])
                if self.args.depth:
                    task_sample.update(depth=sample["depth"][i])

                if torch.max(pre_labels) == output.shape[1] - 1:
                    unseen_task_samples.append((task_sample, image_name[i]))
                else:
                    seen_task_samples.append((task_sample, image_name[i]))

        return seen_task_samples, unseen_task_samples
    
def image_merge(image, label, save_name):
    image = ToPILImage()(image.detach().cpu().byte())
    # # width, height = image.size
    # left = 140
    # top = 30
    # right = 2030
    # bottom = 900
    # # crop
    # image = image.crop((left, top, right, bottom))
    # # resize
    image = image.resize(label.size, Image.BILINEAR)

    image = image.convert('RGBA')
    label = label.convert('RGBA')
    image = Image.blend(image, label, 0.6)
    image.save(save_name)

def paint_trapezoid(color):
    input_height, input_width, _ = color.shape

    # big trapezoid
    big_closest = np.array([
        [0, int(input_height)],
        [int(input_width),
         int(input_height)],
        [int(0.882 * input_width + .5),
         int(.8 * input_height + .5)],
        [int(0.118 * input_width + .5),
         int(.8 * input_height + .5)]
    ])

    big_future = np.array([
        [int(0.118 * input_width + .5),
         int(.8 * input_height + .5)],
        [int(0.882 * input_width + .5),
         int(.8 * input_height + .5)],
        [int(.765 * input_width + .5),
         int(.66 * input_height + .5)],
        [int(.235 * input_width + .5),
         int(.66 * input_height + .5)]
    ])

    # small trapezoid
    small_closest = np.array([
        [488, int(input_height)],
        [1560, int(input_height)],
        [1391, int(.8 * input_height + .5)],
        [621, int(.8 * input_height + .5)]
    ])

    small_future = np.array([
        [741, int(.66 * input_height + .5)],
        [1275, int(.66 * input_height + .5)],
        [1391, int(.8 * input_height + .5)],
        [621, int(.8 * input_height + .5)]
    ])

    big_closest_color = [0, 191, 255]
    big_future_color = [255, 69, 0]

    small_closest_color = [0, 100, 100]
    small_future_color = [69, 69, 69]

    height, width, channel = color.shape
    img = np.zeros((height, width, channel), dtype=np.uint8)
    img = cv2.fillPoly(img, [big_closest], big_closest_color)
    img = cv2.fillPoly(img, [big_future], big_future_color)
    img = cv2.fillPoly(img, [small_closest], small_closest_color)
    img = cv2.fillPoly(img, [small_future], small_future_color)

    img_array = 0.3 * img + color

    return img_array

def load_my_state_dict(model, state_dict):  # custom function to load model when not all dict elements
    own_state = model.state_dict()
    for name, param in state_dict.items():
        # name = "module." + name
        if name not in own_state:
            print('{} not in model_state'.format(name))
            continue
        else:
            own_state[name].copy_(param)

    return model

if __name__ == "__main__":
    main()
