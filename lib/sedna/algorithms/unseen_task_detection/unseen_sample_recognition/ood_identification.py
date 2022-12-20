from typing import Tuple, List

import torch
import numpy as np
from torch.utils.data import DataLoader
import pandas as pd
from sedna.common.constant import KBResourceConstant
from sedna.backend import set_backend
from sedna.common.file_ops import FileOps
from sedna.datasources import BaseDataSource
from sedna.common.class_factory import ClassFactory, ClassType
from sedna.algorithms.seen_task_learning.artifact import Model, Task

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
        if isinstance(task_index, str):
            task_index = FileOps.load(task_index)

        self.base_model = kwargs.get("base_model")(num_class=31)

        self.seen_task_key = KBResourceConstant.SEEN_TASK.value
        self.task_group_key = KBResourceConstant.TASK_GROUPS.value
        self.extractor_key = KBResourceConstant.EXTRACTOR.value

        self.backup_model = kwargs.get('backup_model')
        if not self.backup_model:
            self.seen_extractor = task_index.get(self.seen_task_key).get(self.extractor_key)
            if isinstance(self.seen_extractor, str):
                self.seen_extractor = FileOps.load(self.seen_extractor)
            self.seen_task_groups = task_index.get(self.seen_task_key).get(self.task_group_key)
            self.seen_models = [task.model for task in self.seen_task_groups]
        else:
            self.backup_model = FileOps.download(self.backup_model)        

        self.OOD_thresh = float(kwargs.get("OOD_thresh"))
        self.ood_model = FileOps.load(kwargs.get("OOD_model"))
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
        origin = kwargs.get("origin", "garden")
        seen_task_samples = BaseDataSource(data_type=samples.data_type)
        unseen_task_samples = BaseDataSource(data_type=samples.data_type)
        seen_task_samples.x, unseen_task_samples.x = [], []

        if not self.backup_model:
            allocations = [self.seen_extractor.get(origin) for _ in samples.x]
            samples, models = self._task_remodeling(samples=samples, mappings=allocations)
        else:
            models = [self.backup_model]
            samples.inx = range(samples.num_examples())
            samples = [samples]

        tasks = []
        for inx, df in enumerate(samples):
            m = models[inx]
            if not isinstance(m, Model):
                continue
            if isinstance(m.model, str):
                evaluator = set_backend(estimator=self.base_model)
                evaluator.load(m.model)
            else:
                evaluator = m.model
            InD_list, OoD_list, pred = self.ood_predict(evaluator, df.x, **kwargs)
            seen_task_samples.x.extend(InD_list)
            unseen_task_samples.x.extend(OoD_list)
            task = Task(entry=m.entry, samples=df)
            task.result = pred
            task.model = m
            tasks.append(task)
        res = self._inference_integrate(tasks)
        return seen_task_samples, unseen_task_samples, res, tasks

    def ood_predict(self, evaluator, samples, **kwargs):
        data = self.preprocess_func(samples)
        evaluator.estimator.validator.test_loader = DataLoader(
            data,
            batch_size=2,
            shuffle=False,
            pin_memory=True)

        self.seg_model = evaluator.estimator.validator.model
        self.data_loader = evaluator.estimator.validator.test_loader

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
                    OoD_list.append(samples[i])
                else:
                    InD_list.append(samples[i])
                    predictions.append(pred)

        return InD_list, OoD_list, predictions

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

    def _task_remodeling(self, samples: BaseDataSource, mappings: List):
        """
        Grouping based on assigned tasks
        """
        mappings = np.array(mappings)
        data, models = [], []
        d_type = samples.data_type
        for m in np.unique(mappings):
            task_df = BaseDataSource(data_type=d_type)
            _inx = np.where(mappings == m)
            if isinstance(samples.x, pd.DataFrame):
                task_df.x = samples.x.iloc[_inx]
            else:
                task_df.x = np.array(samples.x)[_inx]
            if d_type != "test":
                if isinstance(samples.x, pd.DataFrame):
                    task_df.y = samples.y.iloc[_inx]
                else:
                    task_df.y = np.array(samples.y)[_inx]
            task_df.inx = _inx[0].tolist()
            if samples.meta_attr is not None:
                task_df.meta_attr = np.array(samples.meta_attr)[_inx]
            data.append(task_df)
            # TODO: if m is out of index
            try:
                model = self.seen_models[m]
            except Exception as err:
                print(f"self.models[{m}] not exists. {err}")
                model = self.seen_models[0]
            models.append(model)
        return data, models

    def _inference_integrate(self, tasks):
        res = {}
        for task in tasks:
            res.update(dict(zip(task.samples.inx, task.result)))
        return np.array([z[1]
                        for z in sorted(res.items(), key=lambda x: x[0])])
