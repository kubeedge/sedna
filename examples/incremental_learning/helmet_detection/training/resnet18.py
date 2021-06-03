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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

WEIGHT_DECAY_KEY = 'WEIGHT_DECAY'


def _residual_block_first(x, is_training, out_channel, strides, name="unit"):
    in_channel = x.get_shape().as_list()[-1]
    with tf.variable_scope(name) as scope:
        print('\tBuilding residual unit: %s' % scope.name)

        # Shortcut connection
        if in_channel == out_channel:
            print('in_channel == out_channel')
            if strides == 1:
                shortcut = tf.identity(x)
            else:
                shortcut = tf.nn.max_pool(x, [1, strides, strides, 1],
                                          [1, strides, strides, 1], 'VALID')
        else:
            shortcut = _conv(x, 1, out_channel, strides, name='shortcut')
        # Residual
        x = _conv(x, 3, out_channel, strides, name='conv_1')
        x = _bn(x, is_training, name='bn_1')
        x = _relu(x, name='relu_1')
        print(x)
        x = _conv(x, 3, out_channel, 1, name='conv_2')
        x = _bn(x, is_training, name='bn_2')
        print(x)
        # Merge
        x = x + shortcut
        x = _relu(x, name='relu_2')
        print(x)
    return x


def _residual_block(x, is_training, name="unit"):
    num_channel = x.get_shape().as_list()[-1]
    with tf.variable_scope(name) as scope:
        print('\tBuilding residual unit: %s' % scope.name)
        # Shortcut connection
        shortcut = x
        # Residual
        x = _conv(x, 3, num_channel, 1, name='conv_1')
        x = _bn(x, is_training, name='bn_1')
        x = _relu(x, name='relu_1')
        print(x)
        x = _conv(x, 3, num_channel, 1, name='conv_2')
        x = _bn(x, is_training, name='bn_2')
        print(x)

        x = x + shortcut
        x = _relu(x, name='relu_2')
        print(x)
    return x


def _conv(x, filter_size, out_channel, strides, name="conv"):
    """
    Helper functions(counts FLOPs and number of weights)
    """
    in_shape = x.get_shape()
    with tf.variable_scope(name):
        # Main operation: conv2d
        kernel = tf.get_variable('kernel',
                                 [filter_size, filter_size, in_shape[3],
                                  out_channel], tf.float32,
                                 initializer=tf.random_normal_initializer(
                                     stddev=np.sqrt(
                                         2.0 / filter_size /
                                         filter_size / out_channel)))
        if kernel not in tf.get_collection(WEIGHT_DECAY_KEY):
            tf.add_to_collection(WEIGHT_DECAY_KEY, kernel)
        if strides == 1:
            conv = tf.nn.conv2d(x, kernel, [1, strides, strides, 1],
                                padding='SAME')
        else:
            kernel_size_effective = filter_size
            pad_total = kernel_size_effective - 1
            pad_beg = pad_total // 2
            pad_end = pad_total - pad_beg
            x = tf.pad(x, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end],
                           [0, 0]])
            conv = tf.nn.conv2d(x, kernel, [1, strides, strides, 1],
                                padding='VALID')
    return conv


def _fc(x, out_dim, name="fc"):
    with tf.variable_scope(name):
        # Main operation: fc
        with tf.device('/CPU:0'):
            w = tf.get_variable('weights', [x.get_shape()[1], out_dim],
                                tf.float32,
                                initializer=tf.random_normal_initializer(
                                    stddev=np.sqrt(1.0 / out_dim)))
            b = tf.get_variable('biases', [out_dim], tf.float32,
                                initializer=tf.constant_initializer(0.0))
        if w not in tf.get_collection(WEIGHT_DECAY_KEY):
            tf.add_to_collection(WEIGHT_DECAY_KEY, w)
        fc = tf.nn.bias_add(tf.matmul(x, w), b)
    return fc


def _bn(x, is_training, name="bn"):
    bn = tf.layers.batch_normalization(inputs=x, momentum=0.99, epsilon=1e-5,
                                       center=True, scale=True,
                                       training=is_training, name=name,
                                       fused=True)
    return bn


def _relu(x, name="relu"):
    return tf.nn.relu(x, name=name)


class ResNet18(object):

    def __init__(self, images, is_training):
        self._build_network(images, is_training)

    def _build_network(self, images, is_training, num_classes=None):
        _counted_scope = []
        self.end_points = {}

        print('Building resnet18 model')
        # filters = [128, 128, 256, 512, 1024]
        filters = [64, 64, 128, 256, 512]
        kernels = [7, 3, 3, 3, 3]
        strides = [2, 0, 2, 2, 2]

        # conv1
        print('\tBuilding unit: conv1')
        with tf.variable_scope('conv1'):
            x = _conv(images, kernels[0], filters[0], strides[0])
            x = _bn(x, is_training)
            x = _relu(x)
            print(x)
            x = tf.nn.max_pool(x, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME')
            print(x)
        self.end_points['conv1_output'] = x

        # conv2_x
        x = _residual_block(x, is_training, name='conv2_1')
        x = _residual_block(x, is_training, name='conv2_2')
        self.end_points['conv2_output'] = x

        # conv3_x
        x = _residual_block_first(x, is_training, filters[2], strides[2],
                                  name='conv3_1')
        x = _residual_block(x, is_training, name='conv3_2')
        self.end_points['conv3_output'] = x

        # conv4_x
        x = _residual_block_first(x, is_training, filters[3], strides[3],
                                  name='conv4_1')
        x = _residual_block(x, is_training, name='conv4_2')
        self.end_points['conv4_output'] = x

        # conv5_x
        x = _residual_block_first(x, is_training, filters[4], strides[4],
                                  name='conv5_1')
        x = _residual_block(x, is_training, name='conv5_2')
        self.end_points['conv5_output'] = x

        # Logit
        if num_classes is not None:
            with tf.variable_scope('logits') as scope:
                print('\tBuilding unit: %s' % scope.name)
                x = tf.reduce_mean(x, [1, 2], name="logits_bottleneck")
                # x = _fc(x, num_classes)    # original resnet18 code used only
                # 8 output classes
                self.end_points['logits'] = x
        # print (self.end_points)

        self.model = x

    def output(self):
        return self.model
