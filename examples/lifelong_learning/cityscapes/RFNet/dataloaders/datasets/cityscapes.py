import os

from PIL import Image
from torch.utils import data
from torchvision import transforms

from dataloaders import custom_transforms as tr


class CityscapesSegmentation(data.Dataset):

    def __init__(self, args, data=None, split="train"):

        self.split = split
        self.args = args
        self.images = {}
        self.disparities = {}
        self.labels = {}

        self.images[split] = [img[0]
                              for img in data.x] if hasattr(data, "x") else data

        if hasattr(data, "x") and len(data.x[0]) == 1:
            self.disparities[split] = self.images[split]
        elif hasattr(data, "x") and len(data.x[0]) == 2:
            self.disparities[split] = [img[1] for img in data.x]
        else:
            self.disparities[split] = data

        self.labels[split] = data.y if hasattr(data, "y") else data

        self.ignore_index = 255

        if len(self.images[split]) == 0:
            raise Exception("No RGB images for split=[%s] found in %s" % (
                split, self.images_base))
        if len(self.disparities[split]) == 0:
            raise Exception("No depth images for split=[%s] found in %s" % (
                split, self.disparities_base))

        print("Found %d %s RGB images" % (len(self.images[split]), split))
        print("Found %d %s disparity images" %
              (len(self.disparities[split]), split))

    def __len__(self):
        return len(self.images[self.split])

    def __getitem__(self, index):
        img_path = self.images[self.split][index].rstrip()
        disp_path = self.disparities[self.split][index].rstrip()
        try:
            lbl_path = self.labels[self.split][index].rstrip()
            _img = Image.open(img_path).convert('RGB').resize(
                self.args.image_size, Image.BILINEAR)
            _depth = Image.open(disp_path).resize(
                self.args.image_size, Image.BILINEAR)
            _target = Image.open(lbl_path).resize(
                self.args.image_size, Image.BILINEAR)

            sample = {'image': _img, 'depth': _depth, 'label': _target}
        except:
            _img = Image.open(img_path).convert('RGB').resize(
                self.args.image_size, Image.BILINEAR)
            _depth = Image.open(disp_path).resize(
                self.args.image_size, Image.BILINEAR)
           
            sample = {'image': _img, 'depth': _depth, 'label': _img}

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
            tr.RandomScaleCrop(base_size=self.args.base_size,
                               crop_size=self.args.crop_size, fill=255),
            tr.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            tr.CropBlackArea(),
            tr.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_ts(self, sample):

        composed_transforms = transforms.Compose([
            tr.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)
