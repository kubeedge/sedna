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
import os.path

from sedna.common.config import Context
from sedna.core.incremental_learning import IncrementalLearning

from interface import Estimator
from dataset import ImgDataset

def main():

    class_names=Context.get_parameters("class_name")
    print(Context.get_parameters("model_path"))
    #read parameters from deployment config
    input_shape=int(Context.get_parameters("input_shape"))
    batch_size=int(Context.get_parameters("batch_size"))
    original_dataset_url=Context.get_parameters("ORIGINAL_DATASET_URL")
    num_parallel_workers=int(Context.get_parameters("num_parallel_workers"))
    if original_dataset_url:
        print("ORIGINAL_DATASET_URL"+ original_dataset_url)
    else:
        print("ORIGINAL_DATASET_URL: NULL" )
    eval_dataset_path=os.path.dirname(original_dataset_url)+r"/eval"
    test_data=ImgDataset(data_type="eval").parse(path=eval_dataset_path,
                                                  train=False,
                                                  image_shape=input_shape,
                                                  batch_size=batch_size,
                                                  num_parallel_workers=num_parallel_workers)
    incremental_instance = IncrementalLearning(estimator=Estimator)
    return incremental_instance.evaluate(test_data,
                                      class_names=class_names,
                                      input_shape=input_shape)

if __name__ == "__main__":
    main()

