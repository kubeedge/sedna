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

import mindspore as ms
from mindvision.classification.models import mobilenet_v2
from mindvision.dataset import DownLoad


class mobilenet_v2_fine_tune:
    def __init__(self, base_model_url):
        self.network = mobilenet_v2(num_classes=2, resize=224)
        self.param_dict = ms.load_checkpoint(base_model_url)

    def get_train_network(self):
        filter_list = [x.name for x in self.network.head.classifier.get_parameters()]
        for key in list(self.param_dict.keys()):
            for name in filter_list:
                if name in key:
                    print("Delete parameter from checkpoint: ", key)
                    del self.param_dict[key]
                    break
        ms.load_param_into_net(self.network, self.param_dict)
        return self.network

    def get_eval_network(self):
        ms.load_param_into_net(self.network, self.param_dict)
        return self.network
