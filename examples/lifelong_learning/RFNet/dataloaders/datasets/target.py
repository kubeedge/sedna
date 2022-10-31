import os
import numpy as np
import scipy.misc as m
from PIL import Image
from torch.utils import data
from mypath import Path
from torchvision import transforms
from dataloaders import custom_transforms as tr

class CityscapesSegmentation(data.Dataset):
    NUM_CLASSES = 24

    def __init__(self, args, root=Path.db_root_dir('target'), split="train"):

        self.root = root
        self.split = split
        self.args = args
        self.images = {}
        self.disparities = {}
        self.labels = {}

        self.images_base = os.path.join(self.root, 'leftImg8bit', self.split)
        self.disparities_base = os.path.join(self.root, 'disparity', self.split)
        self.annotations_base = os.path.join(self.root, 'gtFine', self.split)

        self.images[split] = self.recursive_glob(rootdir=self.images_base, suffix='.png')
        self.images[split].sort()

        self.disparities[split] = self.recursive_glob(rootdir=self.disparities_base, suffix='.png')
        self.disparities[split].sort()

        self.labels[split] = self.recursive_glob(rootdir=self.annotations_base, suffix='TrainIds.png')
        self.labels[split].sort()
        

        self.ignore_index = 255

        if not self.images[split]:
            raise Exception("No RGB images for split=[%s] found in %s" % (split, self.images_base))
        if not self.disparities[split]:
            raise Exception("No depth images for split=[%s] found in %s" % (split, self.disparities_base))

        print("Found %d %s RGB images" % (len(self.images[split]), split))
        print("Found %d %s disparity images" % (len(self.disparities[split]), split))


    def __len__(self):
        return len(self.images[self.split])

    def __getitem__(self, index):

        img_path = self.images[self.split][index].rstrip()
        disp_path = self.disparities[self.split][index].rstrip()
        #print(index)
        try:
            lbl_path = self.labels[self.split][index].rstrip()
            _img = Image.open(img_path).convert('RGB')
            _depth = Image.open(disp_path)
            _target = Image.open(lbl_path)
            sample = {'image': _img,'depth':_depth, 'label': _target}
        except:
            _img = Image.open(img_path).convert('RGB')
            _depth = Image.open(disp_path)
            sample = {'image': _img,'depth':_depth, 'label': _img}

        if self.split == 'train':
            return self.transform_tr(sample)
        elif self.split == 'val':
            return self.transform_val(sample), img_path
        elif self.split == 'test':
            return self.transform_ts(sample), img_path
        elif self.split == 'custom_resize':
            return self.transform_ts(sample), img_path


    def recursive_glob(self, rootdir='.', suffix=''):
        """Performs recursive glob with given suffix and rootdir
            :param rootdir is the root directory
            :param suffix is the suffix to be searched
        """
        return [os.path.join(looproot, filename)
                for looproot, _, filenames in os.walk(rootdir)
                for filename in filenames if filename.endswith(suffix)]

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.CropBlackArea(),
            tr.RandomHorizontalFlip(),
            tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size, fill=255),
            # tr.RandomGaussianBlur(),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            tr.CropBlackArea(),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_ts(self, sample):

        composed_transforms = transforms.Compose([
            tr.CropBlackArea(),
            #tr.FixedResize(size=self.args.crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

if __name__ == '__main__':
    from dataloaders.utils import decode_segmap
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.base_size = 513
    args.crop_size = 513

    cityscapes_train = CityscapesSegmentation(args, split='train')

    dataloader = DataLoader(cityscapes_train, batch_size=2, shuffle=True, num_workers=2)

    for ii, sample in enumerate(dataloader):
        for jj in range(sample["image"].size()[0]):
            img = sample['image'].numpy()
            gt = sample['label'].numpy()
            tmp = np.array(gt[jj]).astype(np.uint8)
            segmap = decode_segmap(tmp, dataset='cityscapes')
            img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
            img_tmp *= (0.229, 0.224, 0.225)
            img_tmp += (0.485, 0.456, 0.406)
            img_tmp *= 255.0
            img_tmp = img_tmp.astype(np.uint8)
            plt.figure()
            plt.title('display')
            plt.subplot(211)
            plt.imshow(img_tmp)
            plt.subplot(212)
            plt.imshow(segmap)

        if ii == 1:
            break

    plt.show(block=True)

