# Copyright 2023 The KubeEdge Authors.
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

from tqdm import tqdm
from sedna.common.class_factory import ClassType, ClassFactory
from sedna.common.log import LOGGER

from utils.args import EvaluationArguments
from utils.metrics import Evaluator
from dataloaders import make_data_loader

__all__ = ('accuracy', )


@ClassFactory.register(ClassType.GENERAL)
def accuracy(y_true, y_pred, **kwargs):
    args = EvaluationArguments()
    _, _, test_loader = make_data_loader(args, test_data=y_true)
    evaluator = Evaluator(args.num_class)

    tbar = tqdm(test_loader, desc='\r')
    for i, (sample, _) in enumerate(tbar):
        if args.depth:
            image, depth, target = sample['image'], sample['depth'], sample['label']
        else:
            image, target = sample['image'], sample['label']
        if args.cuda:
            image, target = image.cuda(), target.cuda()
            if args.depth:
                depth = depth.cuda()

        target[target > evaluator.num_class - 1] = 255
        target = target.cpu().numpy()
        # Add batch sample into evaluator
        evaluator.add_batch(target, y_pred[i])

    # Test during the training
    CPA = evaluator.Pixel_Accuracy_Class()
    mIoU = evaluator.Mean_Intersection_over_Union()
    FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()

    LOGGER.info("CPA:{}, mIoU:{}, fwIoU: {}".format(CPA, mIoU, FWIoU))
    return mIoU
