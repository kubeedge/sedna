from typing import Tuple
import time

import torch
import numpy as np
from torch.utils.data import DataLoader
import torchvision
from PIL import Image

from sedna.backend import set_backend
from sedna.common.file_ops import FileOps
from sedna.datasources import BaseDataSource
from sedna.common.class_factory import ClassFactory, ClassType

__all__ = ('OodIdentification',)


@ClassFactory.register(ClassType.UTD, alias="OodIdentification")
class OodIdentification:
    """
    Corresponding to `OodIdentification`

    Parameters
    ----------
    task_extractor : Dict
        used to match target tasks
    origins: List[Metadata]
        metadata is usually a class feature
        label with a finite values.
    """

    def __init__(self, task_index, **kwargs):
        print(task_index)
        if isinstance(task_index, str) and FileOps.exists(task_index):
            self.task_index = FileOps.load(task_index)
        else:
            self.task_index = task_index

        estimator = kwargs.get("base_model")()
        self.estimator = set_backend(estimator)

        self.backup_model = kwargs.get('backup_model')
        if not self.backup_model:
            self.seen_task_groups = self.task_index.get("seen_task").get("task_groups")
            self.seen_model = [task.model for task in self.seen_task_groups][0]
            self.estimator.load(self.seen_model.model)
        else:
            self.estimator.load(self.backup_model)

        self.OOD_thresh = float(kwargs.get("OOD_thresh"))
        self.OOD_model_path = kwargs.get("OOD_model_path")
        self.ood_model = FileOps.load(self.OOD_model_path)
        self.preprocess_func = kwargs.get("preprocess_func")

    def __call__(self, samples: BaseDataSource, **
                 kwargs) -> Tuple[BaseDataSource, BaseDataSource]:
        '''
        Parameters
        ----------
        samples : BaseDataSource
            inference samples

        Returns
        -------
        seen_task_samples: BaseDataSource
        unseen_task_samples: BaseDataSource
        '''

        data = samples.x
        data = self.preprocess_func(data)

        self.estimator.estimator.validator.test_loader = DataLoader(
            data,
            batch_size=len(data),
            shuffle=False,
            pin_memory=True)
        self.seg_model = self.estimator.estimator.validator.model
        self.data_loader = self.estimator.estimator.validator.test_loader

        seen_task_samples = BaseDataSource(data_type=samples.data_type)
        unseen_task_samples = BaseDataSource(data_type=samples.data_type)
        OoD_list = []
        InD_list = []

        input = None
        predictions = []
        self.seg_model.eval()
        for i, (sample, image_name) in enumerate(self.data_loader):
            image = sample['image'].cuda()
            with torch.no_grad():
                output = self.seg_model(image)

            torch.cuda.synchronize()

            pred = output.data.cpu().numpy()
            pred = np.argmax(pred, axis=1)

            maxLogit = torch.max(output, 1)[0].unsqueeze(1)
            maxLogit = self.batch_min_max(maxLogit)

            softmaxDistance = self.get_softmaxDistance(output).unsqueeze(1)
            maxLogit, softmaxDistance = maxLogit.mean(
                1, keepdim=True), softmaxDistance.mean(
                1, keepdim=True)
            origin_shape = maxLogit.shape
            maxLogit, softmaxDistance = maxLogit.flatten(), softmaxDistance.flatten()

            effec_shape = maxLogit.shape[0]
            if input == 'maxLogit':
                temp_x = maxLogit.reshape(effec_shape, 1)
            elif input == 'softmaxDistance':
                temp_x = softmaxDistance.reshape(effec_shape, 1)
            else:
                temp_x = torch.cat([maxLogit.reshape(
                    effec_shape, 1), softmaxDistance.reshape(effec_shape, 1)], dim=1)

            OOD_pred = self.ood_model.predict(temp_x.cpu().numpy())
            OOD_pred_show = OOD_pred + 1
            OOD_pred_show = OOD_pred_show.reshape(origin_shape)
            maxLogit = maxLogit.reshape(origin_shape)

            for j in range(maxLogit.shape[0]):
                OOD_score = (OOD_pred_show[j] == 1).sum(
                ) / (OOD_pred_show[j] != 0).sum()
                print('OOD_score:', OOD_score)
                if OOD_score > self.OOD_thresh:
                    OoD_list.append(samples.x[0])
                else:
                    InD_list.append(samples.x[0])
                    predictions.append(pred)

        seen_task_samples.x = InD_list
        unseen_task_samples.x = OoD_list
        return seen_task_samples, unseen_task_samples, predictions

    def batch_min_max(self, img):
        max_value = torch.amax(img, [1, 2, 3]).unsqueeze(dim=1)
        min_value = torch.amin(img, [1, 2, 3]).unsqueeze(dim=1)

        [b, n, h, w] = img.shape
        img1 = img.reshape(b, -1)
        img2 = (img1 - min_value) / (max_value - min_value)
        img2 = img2.reshape([b, n, h, w])
        return img2

    def get_softmaxDistance(self, logits):
        seg_softmax_out = torch.nn.Softmax(dim=1)(logits.detach())
        distance, _ = torch.topk(seg_softmax_out, 2, dim=1)
        max_softmaxLogit = distance[:, 0, :, :]
        max2nd_softmaxLogit = distance[:, 1, :, :]
        return max_softmaxLogit - max2nd_softmaxLogit
