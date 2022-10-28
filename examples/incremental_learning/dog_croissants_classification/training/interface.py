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

from __future__ import division


import os

import PIL
import numpy as np
from PIL import Image
import mindspore as ms
import mindspore.nn as nn
from mindvision.engine.loss import CrossEntropySmooth
from mindvision.engine.callback import ValAccMonitor
from mobilenet_v2 import mobilenet_v2_fine_tune


os.environ['BACKEND_TYPE'] = 'MINDSPORE'

def preprocess(img:PIL.Image.Image):
    #image=Image.open(img_path).convert("RGB").resize((224 ,224))
    image=img.convert("RGB").resize((224,224))
    mean = np.array([0.485 * 255, 0.456 * 255, 0.406 * 255])
    std = np.array([0.229 * 255, 0.224 * 255, 0.225 * 255])
    image = np.array(image)
    image = (image - mean) / std
    image = image.astype(np.float32)

    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)
    return image

class Estimator:


    def __init__(self,**kwargs):
        self.trained_ckpt_url=None


    # TODO:save url
    # example : https://www.mindspore.cn/doc/programming_guide/zh-CN/r1.0/train.html#id3
    def train(self, train_data,base_model_url, trained_ckpt_url, valid_data=None,epochs=10, **kwargs):
        network=mobilenet_v2_fine_tune(base_model_url).get_train_network()
        network_opt=nn.Momentum(params=network.trainable_params(),learning_rate=0.01,momentum=0.9)
        network_loss=CrossEntropySmooth(sparse=True, reduction="mean", smooth_factor=0.1, classes_num=2)
        metrics = {"Accuracy" : nn.Accuracy()}
        model=ms.Model(network, loss_fn=network_loss, optimizer=network_opt, metrics=metrics)
        num_epochs = epochs
        #best_ckpt_name=deploy_model_url.split(r"/")[-1]
        #ckpt_dir=deploy_model_url.replace(best_ckpt_name, "")
        model.train(num_epochs, train_data, callbacks=[ValAccMonitor(model, valid_data, num_epochs, save_best_ckpt=True, ckpt_directory=trained_ckpt_url), ms.TimeMonitor()])
        self.trained_ckpt_url=trained_ckpt_url+"/best.ckpt"
        # sedna will save model checkpoint in the path which is the value of MODEL_URL or MODEL_PATH
        #ms.save_checkpoint(network, deploy_model_url)


    def evaluate(self,data,model_path="",class_name="",input_shape=(224,224),**kwargs):
        # load
        network = mobilenet_v2_fine_tune(model_path).get_eval_network()
        # eval
        network_loss = CrossEntropySmooth(sparse=True,
                                          reduction="mean",
                                          smooth_factor=0.1,
                                          classes_num=2)
        model = ms.Model(network, loss_fn=network_loss, optimizer=None, metrics={'acc'})
        acc=model.eval(data, dataset_sink_mode=False)
        print(acc)
        return acc


    def predict(self, data,model, input_shape=None, **kwargs):
        # load

        # preprocess
        preprocessed_data=preprocess(data)
        # predict
        pre=model.predict(ms.Tensor(preprocessed_data))
        result=np.argmax(pre)
        class_name={0:"Croissants", 1:"Dog"}
        #print(class_name[result])
        #return class_name[result]
        return pre

    def load(self, model_url):
        pass

    def save(self, model_path=None):
        if not model_path:
            return
        #model_dir, model_name = os.path.split(model_path)
        network = mobilenet_v2_fine_tune(self.trained_ckpt_url).get_eval_network()
        ms.save_checkpoint(network, model_path)










