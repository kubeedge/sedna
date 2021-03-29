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

import logging

import cv2
import numpy as np
import os
import tensorflow as tf

from resnet18 import ResNet18

LOG = logging.getLogger(__name__)
flags = tf.flags.FLAGS


class Yolo3:

    def __init__(self, sess, is_training, config):
        LOG.info('is_training: %s' % is_training)
        LOG.info('model dir: %s' % flags.train_url)
        LOG.info('input_shape: (%d, %d)' % (flags.input_shape[0], flags.input_shape[1]))
        LOG.info('learning rate: %f' % float(flags.learning_rate))

        self.is_training = is_training
        self.model_dir = flags.train_url
        self.norm_epsilon = config.norm_epsilon
        self.norm_decay = config.norm_decay
        self.obj_threshold = float(flags.obj_threshold)
        self.nms_threshold = float(flags.nms_threshold)

        self.anchors = np.array([float(x) for x in config.anchors]).reshape(-1, 2)
        self.class_names = flags.class_names
        self.num_classes = len(self.class_names)
        self.input_shape = flags.input_shape
        self.nas_sequence = flags.nas_sequence

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        print("anchors : ", self.anchors)
        print("class_names : ", self.class_names)

        if is_training:
            self.images = tf.placeholder(shape=[None, None, None, 3], dtype=tf.float32, name='images')
        else:
            self.images = tf.placeholder(shape=[1, self.input_shape[0], self.input_shape[1], 3], dtype=tf.float32,
                                         name='images')

        self.image_shape = tf.placeholder(dtype=tf.int32, shape=(2,), name='shapes')

        self.bbox_true_13 = tf.placeholder(shape=[None, None, None, 3, self.num_classes + 5], dtype=tf.float32)
        self.bbox_true_26 = tf.placeholder(shape=[None, None, None, 3, self.num_classes + 5], dtype=tf.float32)
        self.bbox_true_52 = tf.placeholder(shape=[None, None, None, 3, self.num_classes + 5], dtype=tf.float32)
        bbox_true = [self.bbox_true_13, self.bbox_true_26, self.bbox_true_52]

        features_out, filters_yolo_block, conv_index = self._resnet18(self.images, self.is_training)

        self.output = self.yolo_inference(features_out, filters_yolo_block, conv_index, len(self.anchors) / 3,
                                          self.num_classes, self.is_training)
        self.loss = self.yolo_loss(self.output, bbox_true, self.anchors, self.num_classes, config.ignore_thresh)

        self.global_step = tf.Variable(0, trainable=False)

        if self.is_training:
            learning_rate = tf.train.exponential_decay(float(flags.learning_rate), self.global_step, 1000, 0.95,
                                                       staircase=True)
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_op = optimizer.minimize(loss=self.loss, global_step=self.global_step)
        else:
            self.boxes, self.scores, self.classes = self.yolo_eval(self.output, self.image_shape, config.max_boxes)

        self.saver = tf.train.Saver()

        ckpt = tf.train.get_checkpoint_state(flags.train_url)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            if not flags.label_changed:
                print('restore model', ckpt.model_checkpoint_path)
                self.saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                print('restore model', ckpt.model_checkpoint_path)
                sess.run(tf.global_variables_initializer())
                sess.run(tf.local_variables_initializer())

                variables = tf.global_variables()
                vars_restore = [var for var in variables if not ("Adam" in var.name
                                                                 or '25' in var.name
                                                                 or '33' in var.name
                                                                 or '41' in var.name)]  # or ("yolo" in var.name))]
                saver_restore = tf.train.Saver(vars_restore)
                saver_restore.restore(sess, ckpt.model_checkpoint_path)
        else:
            print('initialize model with fresh weights...')
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

    def load_weights(self, sess, fpath):
        sess = tf.get_default_session()
        variables = sess.graph.get_collection("variables")
        data = np.load(fpath)
        for v in variables:
            vname = v.name.replace(':0', '')
            if vname not in data:
                print("----------skip %s----------" % vname)
                continue
            print("assigning %s" % vname)
            sess.run(v.assign(data[vname]))

    def step(self, sess, batch_data, is_training):
        """step, read one batch, generate gradients
        """

        # Input feed
        input_feed = {}
        input_feed[self.images] = batch_data['images']
        input_feed[self.bbox_true_13] = batch_data['bbox_true_13']
        input_feed[self.bbox_true_26] = batch_data['bbox_true_26']
        input_feed[self.bbox_true_52] = batch_data['bbox_true_52']

        # Output feed: depends on training or test
        output_feed = [self.loss]  # Loss for this batch.
        if is_training:
            output_feed.append(self.train_op)  # Gradient updates

        outputs = sess.run(output_feed, input_feed)
        return outputs[0]  # loss

    def _batch_normalization_layer(self, input_layer, name=None, training=True, norm_decay=0.997, norm_epsilon=1e-5):
        """Batch normalization is used for feature map extracted from
            convolution layer

        :param input_layer: four dimensional tensor of input
        :param name: the name of batchnorm layer
        :param training: is training or not
        :param norm_decay: The decay rate of moving average is calculated
            during prediction
        :param norm_epsilon: Variance plus a minimal number to prevent
            division by 0

        :return bn_layer: batch normalization处理之后的feature map
        """
        bn_layer = tf.layers.batch_normalization(inputs=input_layer,
                                                 momentum=norm_decay, epsilon=norm_epsilon, center=True,
                                                 scale=True, training=training, name=name, fused=True)
        return tf.nn.relu(bn_layer)
        # return tf.nn.leaky_relu(bn_layer, alpha = 0.1)

    def _conv2d_layer(self, inputs, filters_num, kernel_size, name, use_bias=False, strides=1):
        """Use tf.layers.conv2d Reduce the weight and bias matrix
            initialization process, as well as convolution plus bias operation

        :param inputs: Input variables
        :param filters_num: Number of convolution kernels
        :param strides: Convolution step
        :param name: Convolution layer name
        :param training: is a training process or not
        :param use_bias: use bias or not
        :param kernel_size: the kernels size

        :return conv: Feature map after convolution
        """
        if strides > 1:  # modified 0327
            inputs = tf.pad(inputs, paddings=[[0, 0], [1, 0], [1, 0], [0, 0]], mode='CONSTANT')
        conv = tf.layers.conv2d(inputs=inputs, filters=filters_num,
                                kernel_size=kernel_size, strides=[strides, strides],
                                padding=('SAME' if strides == 1 else 'VALID'),  # padding = 'SAME', #
                                use_bias=use_bias,
                                name=name)  # , kernel_initializer = tf.contrib.layers.xavier_initializer()
        return conv

    def _Residual_block(self, inputs, filters_num, blocks_num, conv_index, training=True, norm_decay=0.997,
                        norm_epsilon=1e-5):
        layer = self._conv2d_layer(inputs, filters_num, kernel_size=3, strides=2, name="conv2d_" + str(conv_index))
        layer = self._batch_normalization_layer(layer, name="batch_normalization_" + str(conv_index), training=training,
                                                norm_decay=norm_decay, norm_epsilon=norm_epsilon)
        conv_index += 1
        for _ in range(blocks_num):
            shortcut = layer
            layer = self._conv2d_layer(layer, filters_num // 2, kernel_size=1, strides=1,
                                       name="conv2d_" + str(conv_index))
            layer = self._batch_normalization_layer(layer, name="batch_normalization_" + str(conv_index),
                                                    training=training, norm_decay=norm_decay, norm_epsilon=norm_epsilon)
            conv_index += 1
            layer = self._conv2d_layer(layer, filters_num, kernel_size=3, strides=1, name="conv2d_" + str(conv_index))
            layer = self._batch_normalization_layer(layer, name="batch_normalization_" + str(conv_index),
                                                    training=training, norm_decay=norm_decay, norm_epsilon=norm_epsilon)
            conv_index += 1
            layer += shortcut
        return layer, conv_index

    def _resnet18(self, inputs, training=True):
        cnn_model = ResNet18(inputs, training)
        for k, v in cnn_model.end_points.items():
            print(k)
            print(v)
        features_out = [cnn_model.end_points['conv5_output'], cnn_model.end_points['conv4_output'],
                        cnn_model.end_points['conv3_output']]
        filters_yolo_block = [256, 128, 64]
        conv_index = 19
        return features_out, filters_yolo_block, conv_index

    def _yolo_block(self, inputs, filters_num, out_filters, conv_index, training=True, norm_decay=0.997,
                    norm_epsilon=1e-5):
        conv = self._conv2d_layer(inputs, filters_num=filters_num, kernel_size=1, strides=1,
                                  name="conv2d_" + str(conv_index))
        conv = self._batch_normalization_layer(conv, name="batch_normalization_" + str(conv_index), training=training,
                                               norm_decay=norm_decay, norm_epsilon=norm_epsilon)
        conv_index += 1
        conv = self._conv2d_layer(conv, filters_num=filters_num * 2, kernel_size=3, strides=1,
                                  name="conv2d_" + str(conv_index))
        conv = self._batch_normalization_layer(conv, name="batch_normalization_" + str(conv_index), training=training,
                                               norm_decay=norm_decay, norm_epsilon=norm_epsilon)
        conv_index += 1
        conv = self._conv2d_layer(conv, filters_num=filters_num, kernel_size=1, strides=1,
                                  name="conv2d_" + str(conv_index))
        conv = self._batch_normalization_layer(conv, name="batch_normalization_" + str(conv_index), training=training,
                                               norm_decay=norm_decay, norm_epsilon=norm_epsilon)
        conv_index += 1
        conv = self._conv2d_layer(conv, filters_num=filters_num * 2, kernel_size=3, strides=1,
                                  name="conv2d_" + str(conv_index))
        conv = self._batch_normalization_layer(conv, name="batch_normalization_" + str(conv_index), training=training,
                                               norm_decay=norm_decay, norm_epsilon=norm_epsilon)
        conv_index += 1
        conv = self._conv2d_layer(conv, filters_num=filters_num, kernel_size=1, strides=1,
                                  name="conv2d_" + str(conv_index))
        conv = self._batch_normalization_layer(conv, name="batch_normalization_" + str(conv_index), training=training,
                                               norm_decay=norm_decay, norm_epsilon=norm_epsilon)
        conv_index += 1
        route = conv
        conv = self._conv2d_layer(conv, filters_num=filters_num * 2, kernel_size=3, strides=1,
                                  name="conv2d_" + str(conv_index))
        conv = self._batch_normalization_layer(conv, name="batch_normalization_" + str(conv_index), training=training,
                                               norm_decay=norm_decay, norm_epsilon=norm_epsilon)
        conv_index += 1
        conv = self._conv2d_layer(conv, filters_num=out_filters, kernel_size=1, strides=1,
                                  name="conv2d_" + str(conv_index), use_bias=True)
        conv_index += 1
        return route, conv, conv_index

    def yolo_inference(self, features_out, filters_yolo_block, conv_index, num_anchors, num_classes, training=True):
        conv = features_out[0]
        conv2d_45 = features_out[1]
        conv2d_26 = features_out[2]

        print('conv : ', conv)
        print('conv2d_45 : ', conv2d_45)
        print('conv2d_26 : ', conv2d_26)

        with tf.variable_scope('yolo'):
            conv2d_57, conv2d_59, conv_index = self._yolo_block(conv, filters_yolo_block[0],
                                                                num_anchors * (num_classes + 5), conv_index=conv_index,
                                                                training=training, norm_decay=self.norm_decay,
                                                                norm_epsilon=self.norm_epsilon)
            print('conv2d_59 : ', conv2d_59)
            print('conv2d_57 : ', conv2d_57)

            conv2d_60 = self._conv2d_layer(conv2d_57, filters_num=filters_yolo_block[1], kernel_size=1, strides=1,
                                           name="conv2d_" + str(conv_index))
            conv2d_60 = self._batch_normalization_layer(conv2d_60, name="batch_normalization_" + str(conv_index),
                                                        training=training, norm_decay=self.norm_decay,
                                                        norm_epsilon=self.norm_epsilon)
            print('conv2d_60 : ', conv2d_60)

            conv_index += 1
            upSample_0 = tf.image.resize_nearest_neighbor(conv2d_60,
                                                          [2 * tf.shape(conv2d_60)[1], 2 * tf.shape(conv2d_60)[2]],
                                                          name='upSample_0')
            print('upSample_0 : ', upSample_0)

            route0 = tf.concat([upSample_0, conv2d_45], axis=-1, name='route_0')
            print('route0 : ', route0)

            conv2d_65, conv2d_67, conv_index = self._yolo_block(route0, filters_yolo_block[1],
                                                                num_anchors * (num_classes + 5), conv_index=conv_index,
                                                                training=training, norm_decay=self.norm_decay,
                                                                norm_epsilon=self.norm_epsilon)
            print('conv2d_67 : ', conv2d_67)
            print('conv2d_65 : ', conv2d_65)

            conv2d_68 = self._conv2d_layer(conv2d_65, filters_num=filters_yolo_block[2], kernel_size=1, strides=1,
                                           name="conv2d_" + str(conv_index))
            conv2d_68 = self._batch_normalization_layer(conv2d_68, name="batch_normalization_" + str(conv_index),
                                                        training=training, norm_decay=self.norm_decay,
                                                        norm_epsilon=self.norm_epsilon)
            print('conv2d_68 : ', conv2d_68)

            conv_index += 1
            upSample_1 = tf.image.resize_nearest_neighbor(conv2d_68,
                                                          [2 * tf.shape(conv2d_68)[1], 2 * tf.shape(conv2d_68)[2]],
                                                          name='upSample_1')
            print('upSample_1 : ', upSample_1)

            route1 = tf.concat([upSample_1, conv2d_26], axis=-1, name='route_1')
            print('route1 : ', route1)

            _, conv2d_75, _ = self._yolo_block(route1, filters_yolo_block[2], num_anchors * (num_classes + 5),
                                               conv_index=conv_index, training=training, norm_decay=self.norm_decay,
                                               norm_epsilon=self.norm_epsilon)
            print('conv2d_75 : ', conv2d_75)

        return [conv2d_59, conv2d_67, conv2d_75]

    def yolo_head(self, feats, anchors, num_classes, input_shape, training=True):
        num_anchors = len(anchors)
        anchors_tensor = tf.reshape(tf.constant(anchors, dtype=tf.float32), [1, 1, 1, num_anchors, 2])
        grid_size = tf.shape(feats)[1:3]
        predictions = tf.reshape(feats, [-1, grid_size[0], grid_size[1], num_anchors, num_classes + 5])
        grid_y = tf.tile(tf.reshape(tf.range(grid_size[0]), [-1, 1, 1, 1]), [1, grid_size[1], 1, 1])
        grid_x = tf.tile(tf.reshape(tf.range(grid_size[1]), [1, -1, 1, 1]), [grid_size[0], 1, 1, 1])
        grid = tf.concat([grid_x, grid_y], axis=-1)
        grid = tf.cast(grid, tf.float32)
        box_xy = (tf.sigmoid(predictions[..., :2]) + grid) / tf.cast(grid_size[::-1], tf.float32)
        box_wh = tf.exp(predictions[..., 2:4]) * anchors_tensor / input_shape[::-1]
        box_confidence = tf.sigmoid(predictions[..., 4:5])
        box_class_probs = tf.sigmoid(predictions[..., 5:])
        if training == True:
            return grid, predictions, box_xy, box_wh
        return box_xy, box_wh, box_confidence, box_class_probs

    def yolo_boxes_scores(self, feats, anchors, num_classes, input_shape, image_shape):
        input_shape = tf.cast(input_shape, tf.float32)
        image_shape = tf.cast(image_shape, tf.float32)
        box_xy, box_wh, box_confidence, box_class_probs = self.yolo_head(feats, anchors, num_classes, input_shape,
                                                                         training=False)
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]
        new_shape = tf.round(image_shape * tf.reduce_min(input_shape / image_shape))
        offset = (input_shape - new_shape) / 2. / input_shape
        scale = input_shape / new_shape
        box_yx = (box_yx - offset) * scale
        box_hw = box_hw * scale

        box_min = box_yx - box_hw / 2.
        box_max = box_yx + box_hw / 2.
        boxes = tf.concat(
            [box_min[..., 0:1],
             box_min[..., 1:2],
             box_max[..., 0:1],
             box_max[..., 1:2]],
            axis=-1
        )
        boxes *= tf.concat([image_shape, image_shape], axis=-1)
        boxes = tf.reshape(boxes, [-1, 4])
        boxes_scores = box_confidence * box_class_probs
        boxes_scores = tf.reshape(boxes_scores, [-1, num_classes])
        return boxes, boxes_scores

    def box_iou(self, box1, box2):
        box1 = tf.expand_dims(box1, -2)
        box1_xy = box1[..., :2]
        box1_wh = box1[..., 2:4]
        box1_mins = box1_xy - box1_wh / 2.
        box1_maxs = box1_xy + box1_wh / 2.

        box2 = tf.expand_dims(box2, 0)
        box2_xy = box2[..., :2]
        box2_wh = box2[..., 2:4]
        box2_mins = box2_xy - box2_wh / 2.
        box2_maxs = box2_xy + box2_wh / 2.

        intersect_mins = tf.maximum(box1_mins, box2_mins)
        intersect_maxs = tf.minimum(box1_maxs, box2_maxs)
        intersect_wh = tf.maximum(intersect_maxs - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        box1_area = box1_wh[..., 0] * box1_wh[..., 1]
        box2_area = box2_wh[..., 0] * box2_wh[..., 1]
        iou = intersect_area / (box1_area + box2_area - intersect_area)
        return iou

    def yolo_loss(self, yolo_output, y_true, anchors, num_classes, ignore_thresh=.5):
        loss = 0.0
        anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        input_shape = tf.shape(yolo_output[0])[1: 3] * 32
        input_shape = tf.cast(input_shape, tf.float32)
        grid_shapes = [tf.cast(tf.shape(yolo_output[l])[1:3], tf.float32) for l in range(3)]
        for index in range(3):
            object_mask = y_true[index][..., 4:5]
            class_probs = y_true[index][..., 5:]
            grid, predictions, pred_xy, pred_wh = self.yolo_head(yolo_output[index], anchors[anchor_mask[index]],
                                                                 num_classes, input_shape, training=True)
            pred_box = tf.concat([pred_xy, pred_wh], axis=-1)
            raw_true_xy = y_true[index][..., :2] * grid_shapes[index][::-1] - grid
            object_mask_bool = tf.cast(object_mask, dtype=tf.bool)
            raw_true_wh = tf.log(
                tf.where(tf.equal(y_true[index][..., 2:4] / anchors[anchor_mask[index]] * input_shape[::-1], 0),
                         tf.ones_like(y_true[index][..., 2:4]),
                         y_true[index][..., 2:4] / anchors[anchor_mask[index]] * input_shape[::-1]))
            box_loss_scale = 2 - y_true[index][..., 2:3] * y_true[index][..., 3:4]
            ignore_mask = tf.TensorArray(dtype=tf.float32, size=1, dynamic_size=True)

            def loop_body(internal_index, ignore_mask):
                true_box = tf.boolean_mask(y_true[index][internal_index, ..., 0:4],
                                           object_mask_bool[internal_index, ..., 0])
                iou = self.box_iou(pred_box[internal_index], true_box)
                best_iou = tf.reduce_max(iou, axis=-1)
                ignore_mask = ignore_mask.write(internal_index, tf.cast(best_iou < ignore_thresh, tf.float32))
                return internal_index + 1, ignore_mask

            _, ignore_mask = tf.while_loop(
                lambda internal_index, ignore_mask: internal_index < tf.shape(yolo_output[0])[0], loop_body,
                [0, ignore_mask])
            ignore_mask = ignore_mask.stack()
            ignore_mask = tf.expand_dims(ignore_mask, axis=-1)
            xy_loss = object_mask * box_loss_scale * tf.nn.sigmoid_cross_entropy_with_logits(
                labels=raw_true_xy,
                logits=predictions[..., 0:2])
            wh_loss = object_mask * box_loss_scale * 0.5 * tf.square(raw_true_wh - predictions[..., 2:4])
            confidence_loss = object_mask * tf.nn.sigmoid_cross_entropy_with_logits(
                labels=object_mask,
                logits=predictions[..., 4:5]) + (1 - object_mask) * tf.nn.sigmoid_cross_entropy_with_logits(
                labels=object_mask,
                logits=predictions[..., 4:5]) * ignore_mask
            class_loss = object_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=class_probs,
                                                                               logits=predictions[..., 5:])
            xy_loss = tf.reduce_sum(xy_loss) / tf.cast(tf.shape(yolo_output[0])[0], tf.float32)
            wh_loss = tf.reduce_sum(wh_loss) / tf.cast(tf.shape(yolo_output[0])[0], tf.float32)
            confidence_loss = tf.reduce_sum(confidence_loss) / tf.cast(tf.shape(yolo_output[0])[0], tf.float32)
            class_loss = tf.reduce_sum(class_loss) / tf.cast(tf.shape(yolo_output[0])[0], tf.float32)

            loss += xy_loss + wh_loss + confidence_loss + class_loss

        return loss

    def yolo_eval(self, yolo_outputs, image_shape, max_boxes=20):
        with tf.variable_scope('boxes_scores'):
            anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
            boxes = []
            box_scores = []
            input_shape = tf.shape(yolo_outputs[0])[1: 3] * 32
            for i in range(len(yolo_outputs)):
                _boxes, _box_scores = self.yolo_boxes_scores(yolo_outputs[i], self.anchors[anchor_mask[i]],
                                                             len(self.class_names), input_shape, image_shape)
                boxes.append(_boxes)
                box_scores.append(_box_scores)
            boxes = tf.concat(boxes, axis=0)
            box_scores = tf.concat(box_scores, axis=0)

        with tf.variable_scope('nms'):
            mask = box_scores >= self.obj_threshold
            max_boxes_tensor = tf.constant(max_boxes, dtype=tf.int32)
            boxes_ = []
            scores_ = []
            classes_ = []
            for c in range(len(self.class_names)):
                class_boxes = tf.boolean_mask(boxes, mask[:, c])
                class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
                nms_index = tf.image.non_max_suppression(class_boxes, class_box_scores, max_boxes_tensor,
                                                         iou_threshold=self.nms_threshold)
                class_boxes = tf.gather(class_boxes, nms_index)
                class_box_scores = tf.gather(class_box_scores, nms_index)
                classes = tf.ones_like(class_box_scores, 'int32') * c
                boxes_.append(class_boxes)
                scores_.append(class_box_scores)
                classes_.append(classes)

        with tf.variable_scope('output'):
            boxes_ = tf.concat(boxes_, axis=0, name='boxes')
            scores_ = tf.concat(scores_, axis=0, name='scores')
            classes_ = tf.concat(classes_, axis=0, name='classes')
        return boxes_, scores_, classes_


class YoloConfig:
    gpu_index = "3"

    net_type = 'resnet18'

    anchors = [10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 163, 326]

    max_boxes = 50
    jitter = 0.3
    hue = 0.1
    sat = 1.0
    cont = 0.8
    bri = 0.1
    norm_decay = 0.99
    norm_epsilon = 1e-5
    ignore_thresh = 0.5


class YOLOInference(object):

    # pylint: disable=too-many-arguments, too-many-instance-attributes
    def __init__(self, sess, pb_model_path, input_shape):
        """
        initialization
        """

        self.load_model(sess, pb_model_path)
        self.input_shape = input_shape

    def load_model(self, sess, pb_model_path):
        """
        import model and load parameters from pb file
        """

        logging.info("Import yolo model from pb start .......")

        with sess.as_default():
            with sess.graph.as_default():
                with tf.gfile.FastGFile(pb_model_path, 'rb') as f_handle:
                    logging.info("ParseFromString start .......")
                    graph_def = tf.GraphDef()
                    graph_def.ParseFromString(f_handle.read())
                    logging.info("ParseFromString end .......")

                    tf.import_graph_def(graph_def, name='')
                    logging.info("Import_graph_def end .......")

        logging.info("Import yolo model from pb end .......")

    # pylint: disable=too-many-locals
    # pylint: disable=invalid-name
    def predict(self, sess, img_data):
        """
        prediction for image rectangle by input_feed and output_feed
        """

        with sess.as_default():
            new_image = self.preprocess(img_data, self.input_shape)
            input_feed = self.create_input_feed(sess, new_image, img_data)
            output_fetch = self.create_output_fetch(sess)
            all_classes, all_scores, all_bboxes = sess.run(output_fetch, input_feed)

            return all_classes, all_scores, all_bboxes

    def create_input_feed(self, sess, new_image, img_data):
        """
        create input feed data
        """

        input_feed = {}

        input_img_data = sess.graph.get_tensor_by_name('images:0')
        input_feed[input_img_data] = new_image

        input_img_shape = sess.graph.get_tensor_by_name('shapes:0')
        input_feed[input_img_shape] = [img_data.shape[0], img_data.shape[1]]

        return input_feed

    def create_output_fetch(self, sess):
        """
        create output fetch tensors
        """

        output_classes = sess.graph.get_tensor_by_name('output/classes:0')
        output_scores = sess.graph.get_tensor_by_name('output/scores:0')
        output_boxes = sess.graph.get_tensor_by_name('output/boxes:0')

        output_fetch = [output_classes, output_scores, output_boxes]

        return output_fetch

    def preprocess(self, image, input_shape):
        """
        resize image with unchanged aspect ratio using padding by opencv
        """
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape

        input_h, input_w = input_shape
        scale = min(float(input_w) / float(w), float(input_h) / float(h))
        nw = int(w * scale)
        nh = int(h * scale)

        image = cv2.resize(image, (nw, nh))

        new_image = np.zeros((input_h, input_w, 3), np.float32)
        new_image.fill(128)
        bh, bw, _ = new_image.shape
        new_image[int((bh - nh) / 2):(nh + int((bh - nh) / 2)), int((bw - nw) / 2):(nw + int((bw - nw) / 2)), :] = image

        new_image /= 255.
        new_image = np.expand_dims(new_image, 0)  # Add batch dimension.
        return new_image
