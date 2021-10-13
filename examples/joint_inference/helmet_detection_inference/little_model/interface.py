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

import os
import logging

import cv2
import numpy as np
import tensorflow as tf

LOG = logging.getLogger(__name__)
os.environ['BACKEND_TYPE'] = 'TENSORFLOW'
flags = tf.flags.FLAGS


def create_input_feed(sess, new_image, img_data):
    """Create input feed for edge model inference"""
    input_feed = {}

    input_img_data = sess.graph.get_tensor_by_name('images:0')
    input_feed[input_img_data] = new_image

    input_img_shape = sess.graph.get_tensor_by_name('shapes:0')
    input_feed[input_img_shape] = [img_data.shape[0], img_data.shape[1]]

    return input_feed


def create_output_fetch(sess):
    """Create output fetch for edge model inference"""
    output_classes = sess.graph.get_tensor_by_name('concat_19:0')
    output_scores = sess.graph.get_tensor_by_name('concat_18:0')
    output_boxes = sess.graph.get_tensor_by_name('concat_17:0')

    output_fetch = [output_classes, output_scores, output_boxes]
    return output_fetch


class Estimator:

    def __init__(self, **kwargs):
        """
        initialize logging configuration
        """
        graph = tf.Graph()
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.1
        self.session = tf.Session(graph=graph, config=config)
        self.input_shape = [416, 736]
        self.create_input_feed = create_input_feed
        self.create_output_fetch = create_output_fetch

    def load(self, model_url=""):
        with self.session.as_default():
            with self.session.graph.as_default():
                with tf.gfile.FastGFile(model_url, 'rb') as handle:
                    LOG.info(f"Load model {model_url}, "
                             f"ParseFromString start .......")
                    graph_def = tf.GraphDef()
                    graph_def.ParseFromString(handle.read())
                    LOG.info("ParseFromString end .......")

                    tf.import_graph_def(graph_def, name='')
                    LOG.info("Import_graph_def end .......")

        LOG.info("Import model from pb end .......")

    @staticmethod
    def preprocess(image, input_shape):
        """Preprocess functions in edge model inference"""

        # resize image with unchanged aspect ratio using padding by opencv

        h, w, _ = image.shape

        input_h, input_w = input_shape
        scale = min(float(input_w) / float(w), float(input_h) / float(h))
        nw = int(w * scale)
        nh = int(h * scale)

        image = cv2.resize(image.astype(np.float32), (nw, nh))

        new_image = np.zeros((input_h, input_w, 3), np.float32)
        new_image.fill(128)
        bh, bw, _ = new_image.shape
        new_image[int((bh - nh) / 2):(nh + int((bh - nh) / 2)),
                  int((bw - nw) / 2):(nw + int((bw - nw) / 2)), :] = image

        new_image /= 255.
        new_image = np.expand_dims(new_image, 0)  # Add batch dimension.
        return new_image

    @staticmethod
    def postprocess(model_output):
        all_classes, all_scores, all_bboxes = model_output
        bboxes = []
        for c, s, bbox in zip(all_classes, all_scores, all_bboxes):
            bbox[0], bbox[1], bbox[2], bbox[3] = bbox[1].tolist(
            ), bbox[0].tolist(), bbox[3].tolist(), bbox[2].tolist()
            bboxes.append(bbox.tolist() + [s.tolist(), c.tolist()])

        return bboxes

    def predict(self, data, **kwargs):
        img_data_np = np.array(data)
        with self.session.as_default():
            new_image = self.preprocess(img_data_np, self.input_shape)
            input_feed = self.create_input_feed(
                self.session, new_image, img_data_np)
            output_fetch = self.create_output_fetch(self.session)
            output = self.session.run(output_fetch, input_feed)
            return self.postprocess(output)
