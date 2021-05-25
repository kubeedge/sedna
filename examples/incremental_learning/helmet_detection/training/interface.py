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
import six
import tensorflow as tf
from tqdm import tqdm
from data_gen import DataGen
from validate_utils import validate
from yolo3_multiscale import Yolo3
from yolo3_multiscale import YoloConfig

os.environ['BACKEND_TYPE'] = 'TENSORFLOW'
LOG = logging.getLogger(__name__)


def preprocess(image, input_shape):
    """Preprocess functions in edge model inference"""
    # resize image with unchanged aspect ratio using padding by opencv
    h, w, _ = image.shape
    input_h, input_w = input_shape
    scale = min(float(input_w) / float(w), float(input_h) / float(h))
    nw = int(w * scale)
    nh = int(h * scale)
    image = cv2.resize(image, (nw, nh))
    new_image = np.zeros((input_h, input_w, 3), np.float32)
    new_image.fill(128)
    bh, bw, _ = new_image.shape
    new_image[int((bh - nh) / 2):(nh + int((bh - nh) / 2)),
    int((bw - nw) / 2):(nw + int((bw - nw) / 2)), :] = image
    new_image /= 255.
    new_image = np.expand_dims(new_image, 0)  # Add batch dimension.
    return new_image


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
    output_classes = sess.graph.get_tensor_by_name('output/classes:0')
    output_scores = sess.graph.get_tensor_by_name('output/scores:0')
    output_boxes = sess.graph.get_tensor_by_name('output/boxes:0')
    output_fetch = [output_classes, output_scores, output_boxes]
    return output_fetch


class Estimator:

    def __init__(self, **kwargs):
        """
        initialize logging configuration
        """
        sess_config = tf.ConfigProto()
        self.graph = tf.Graph()
        self.session = tf.compat.v1.Session(config=sess_config, graph=self.graph)

    def train(self, train_data, valid_data=None, **kwargs):
        """
        train
        """
        yolo_config = YoloConfig()

        data_gen = DataGen(yolo_config, train_data.x)

        max_epochs = int(kwargs.get("max_epochs", "1"))
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:

            model = Yolo3(sess, True, yolo_config)

            if os.path.exists(model.model_dir):
                saver = tf.train.Saver()
                latest_ckpt = tf.train.latest_checkpoint(model.model_dir)
                if latest_ckpt:
                    LOG.info(f"latest_ckpt={latest_ckpt}")
                    saver.restore(sess, latest_ckpt)
            else:
                os.makedirs(model.model_dir)
            steps_per_epoch = int(round(data_gen.train_data_size / data_gen.batch_size))
            total = steps_per_epoch * max_epochs
            loss = []
            with tqdm(desc='Train: ', total=total) as pbar:
                for epoch in range(max_epochs):
                    LOG.info('Epoch %d...' % epoch)
                    for step in range(steps_per_epoch):  # Get a batch and make a step.

                        batch_data = data_gen.next_batch_train()  # get batch data from Queue
                        if not batch_data:
                            continue

                        batch_loss = model.step(sess, batch_data, True)
                        # pbar.set_description('Train, loss={:.8f}'.format(batch_loss))
                        pbar.set_description('Train, input_shape=(%d, %d), loss=%.4f' % (
                            batch_data['input_shape'][0], batch_data['input_shape'][1], batch_loss))
                        pbar.update()
                        loss.append(batch_loss)
                    LOG.info("Saving model, global_step: %d" % model.global_step.eval())
                    checkpoint_path = os.path.join(model.model_dir, "yolo3-epoch%03d.ckpt" % (epoch))
                    model.saver.save(sess, checkpoint_path, global_step=model.global_step, write_meta_graph=False)
            return {"loss": float(np.mean(loss))}

    def evaluate(self, valid_data, model_path="", class_names="", input_shape=(352, 640), **kwargs):
        """
        validate
        """
        precision, recall, all_precisions, all_recalls = validate(model_path=model_path,
                                                                  test_dataset=valid_data.x,
                                                                  class_names=class_names,
                                                                  input_shapre=input_shape)
        return {
                "recall": recall, "precision": precision
            }

    def avg_checkpoints(self):
        """
        Average the last N checkpoints in the model_dir.
        """

        LOG.info("average checkpoints start .......")

        with self.session.as_default() as sess:

            yolo_config = YoloConfig()
            model = Yolo3(sess, False, yolo_config)

            model_dir = model.model_dir
            num_last_checkpoints = 5
            global_step = model.global_step.eval()
            global_step_name = model.global_step.name.split(":")[0]

            checkpoint_state = tf.train.get_checkpoint_state(model_dir)
            if not checkpoint_state:
                logging.info("# No checkpoint file found in directory: %s" % model_dir)
                return None

            # Checkpoints are ordered from oldest to newest.
            checkpoints = (checkpoint_state.all_model_checkpoint_paths[-num_last_checkpoints:])

            if len(checkpoints) < num_last_checkpoints:
                logging.info("# Skipping averaging checkpoints because not enough checkpoints is avaliable.")
                return None

            avg_model_dir = os.path.join(model_dir, "avg_checkpoints")
            if not tf.gfile.Exists(avg_model_dir):
                logging.info("# Creating new directory %s for saving averaged checkpoints." % avg_model_dir)
                tf.gfile.MakeDirs(avg_model_dir)

            logging.info("# Reading and averaging variables in checkpoints:")
            var_list = tf.contrib.framework.list_variables(checkpoints[0])
            var_values, var_dtypes = {}, {}
            for (name, shape) in var_list:
                if name != global_step_name:
                    var_values[name] = np.zeros(shape)

            for checkpoint in checkpoints:
                logging.info("        %s" % checkpoint)
                reader = tf.contrib.framework.load_checkpoint(checkpoint)
                for name in var_values:
                    tensor = reader.get_tensor(name)
                    var_dtypes[name] = tensor.dtype
                    var_values[name] += tensor

            for name in var_values:
                var_values[name] /= len(checkpoints)

            # Build a graph with same variables in the checkpoints, and save the averaged
            # variables into the avg_model_dir.
            with tf.Graph().as_default():
                tf_vars = [tf.get_variable(v, shape=var_values[v].shape, dtype=var_dtypes[name])
                           for v in var_values]

                placeholders = [tf.placeholder(v.dtype, shape=v.shape) for v in tf_vars]
                assign_ops = [tf.assign(v, p) for (v, p) in zip(tf_vars, placeholders)]
                global_step_var = tf.Variable(global_step, name=global_step_name, trainable=False)
                saver = tf.train.Saver(tf.global_variables())

                with tf.Session() as sess:
                    sess.run(tf.global_variables_initializer())
                    for p, assign_op, (name, value) in zip(placeholders, assign_ops, six.iteritems(var_values)):
                        sess.run(assign_op, {p: value})

                    # Use the built saver to save the averaged checkpoint. Only keep 1
                    # checkpoint and the best checkpoint will be moved to avg_best_metric_dir.
                    saver.save(sess, os.path.join(avg_model_dir, "translate.ckpt"))

        logging.info("average checkpoints end .......")

    def predict(self, data, input_shape=None, **kwargs):
        img_data_np = np.array(data)
        with self.session.as_default():
            new_image = preprocess(img_data_np, input_shape)
            input_feed = create_input_feed(self.session, new_image, img_data_np)
            output_fetch = create_output_fetch(self.session)
            output = self.session.run(output_fetch, input_feed)

            return output

    def load(self, model_url):
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

    def save(self, model_path=None):
        """
        save model as a single pb file from checkpoint
        """
        model_dir = ""
        model_name = "model.pb"
        if model_path:
            model_dir, model_name = os.path.split(model_path)
        logging.info("save model as .pb start .......")
        tf.reset_default_graph()

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:

            yolo_config = YoloConfig()

            model = Yolo3(sess, False, yolo_config)
            if not (model_dir and os.path.isdir(model_dir)):
                model_dir = model.model_dir
            input_graph_def = sess.graph.as_graph_def()
            output_tensors = [model.boxes, model.scores, model.classes]
            output_tensors = [t.op.name for t in output_tensors]
            graph = tf.graph_util.convert_variables_to_constants(sess, input_graph_def, output_tensors)
            tf.train.write_graph(graph, model_dir, model_name, False)

        logging.info("save model as .pb end .......")
