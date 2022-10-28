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

import mindspore.dataset as ds
import mindspore.dataset.vision as vision
from sedna.datasources import BaseDataSource


class ImgDataset(BaseDataSource):
    def __init__(self, data_type="train", func=None):
        super(ImgDataset, self).__init__(data_type=data_type, func=func)

    def parse(self, *args, path=None, train=True, image_shape=224, batch_size=2,num_parallel_workers=1, **kwargs):
        dataset = ds.ImageFolderDataset(
            path, num_parallel_workers=num_parallel_workers,
            class_indexing={"croissants": 0, "dog": 1})
        mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
        std = [0.229 * 255, 0.224 * 255, 0.225 * 255]
        if train:
            trans = [
                vision.RandomCropDecodeResize(image_shape, scale=(0.08, 1.0), ratio=(0.75, 1.333)),
                vision.RandomHorizontalFlip(prob=0.5),
                vision.Normalize(mean=mean, std=std),
                vision.HWC2CHW()
            ]
        else:
            trans = [
                vision.Decode(),
                vision.Resize(256),
                vision.CenterCrop(image_shape),
                vision.Normalize(mean=mean, std=std),
                vision.HWC2CHW()
            ]
        dataset = dataset.map(operations=trans,
                              input_columns="image",
                              num_parallel_workers=num_parallel_workers)
        dataset = dataset.batch(batch_size, drop_remainder=True)
        return dataset

    '''
    def download_dataset(self):
        dataset_url = "https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/beginner/DogCroissants.zip"
        path = "./datasets"
        dl = DownLoad()
        dl.download_and_extract_archive(dataset_url, path)
    '''

