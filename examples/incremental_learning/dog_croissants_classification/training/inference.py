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
import glob
import os


from PIL import Image
from sedna.common.config import Context
from sedna.core.incremental_learning import IncrementalLearning
from interface import Estimator
import shutil
import mindspore as ms
from mobilenet_v2 import mobilenet_v2_fine_tune


he_saved_url = Context.get_parameters("HE_SAVED_URL", './tmp')

def output_deal(is_hard_example, infer_image_path):
    img_name=infer_image_path.split(r"/")[-1]
    img_category = infer_image_path.split(r"/")[-2]
    if is_hard_example:
        shutil.copy(infer_image_path,f"{he_saved_url}/{img_category}_{img_name}")

def main():

    hard_example_mining = IncrementalLearning.get_hem_algorithm_from_config(
        random_ratio=0.3
    )
    incremental_instance = IncrementalLearning(estimator=Estimator, hard_example_mining=hard_example_mining)
    class_names=Context.get_parameters("class_name")
    #read parameters from deployment config
    input_shape=int(Context.get_parameters("input_shape"))
    # load ckpt
    model_url=Context.get_parameters("model_url")
    print("model_url=" + model_url)
    # load model ckpt here
    network = mobilenet_v2_fine_tune(base_model_url=model_url).get_eval_network()
    #ms.load_checkpoint(model_url, network)
    model = ms.Model(network)
    # load dataset
    #train_dataset_url = BaseConfig.train_dataset_url
    infer_dataset_url=Context.get_parameters("infer_url")
    print(infer_dataset_url)
    # get each image unber infer_dataset_url with wildcard
    while True:
        for each_img in glob.glob(infer_dataset_url+"/*/*"):
            infer_data=Image.open(each_img)
            results, _, is_hard_example = incremental_instance.inference(data=infer_data,
                                                        model=model,
                                                        class_names=class_names,
                                                        input_shape=input_shape)
            hard_example="is hard example" if is_hard_example else "is not hard example"
            print(f"{each_img}--->{results}-->{hard_example}")
            output_deal(is_hard_example, each_img)

if __name__ == "__main__":
    main()

