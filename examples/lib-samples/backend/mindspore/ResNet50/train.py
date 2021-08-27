# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""train resnet."""
import argparse
import ast
from mindspore.common import set_seed
from lib.sedna.backend import set_backend
from interface import Estimator


parser = argparse.ArgumentParser(description='Image classification')
parser.add_argument('--run_distribute', type=ast.literal_eval, default=False, help='Run distribute')
parser.add_argument('--device_num', type=int, default=1, help='Device num.')
parser.add_argument('--model_save_path', type=str, default="")

parser.add_argument('--dataset_path', type=str, default="", help='Dataset path')
parser.add_argument('--device_target', type=str, default='Ascend', choices=("Ascend", "GPU", "CPU"),
                    help="Device target, support Ascend, GPU and CPU.")
parser.add_argument('--pre_trained', type=str, default=None, help='Pretrained checkpoint path')

set_seed(1)


def main():
    args_opt = parser.parse_args()
    train_data = args_opt.dataset_path
    instance = set_backend(estimator=Estimator)
    return instance.train(train_data, args_opt=args_opt)


if __name__ == '__main__':
    main()
