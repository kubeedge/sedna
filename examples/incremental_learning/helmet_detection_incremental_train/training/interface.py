import logging

import numpy as np
import os
import six
import tensorflow as tf
from tqdm import tqdm

from data_gen import DataGen
from sedna.incremental_learning.incremental_learning import IncrementalConfig
from yolo3_multiscale import Yolo3
from yolo3_multiscale import YoloConfig

LOG = logging.getLogger(__name__)
BASE_MODEL_URL = IncrementalConfig().base_model_url

flags = tf.flags.FLAGS


class Interface:

    def __init__(self):
        """
        initialize logging configuration
        """

    def train(self, train_data, valid_data):
        """
        train
        """
        yolo_config = YoloConfig()

        data_gen = DataGen(yolo_config, train_data, valid_data)

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            model = Yolo3(sess, True, yolo_config)

            if BASE_MODEL_URL and os.path.exists(BASE_MODEL_URL):
                LOG.info(f"loading base model, BASE_MODEL_URL={BASE_MODEL_URL}")
                saver = tf.train.Saver()
                latest_ckpt = tf.train.latest_checkpoint(BASE_MODEL_URL)
                LOG.info(f"latest_ckpt={latest_ckpt}")
                saver.restore(sess, latest_ckpt)

            steps_per_epoch = int(round(data_gen.train_data_size / data_gen.batch_size))
            total = steps_per_epoch * flags.max_epochs
            with tqdm(desc='Train: ', total=total) as pbar:
                for epoch in range(flags.max_epochs):
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

                    # LOG.info('validating...')
                    # val_loss = self.validate(sess, model, data_gen, flags.batch_size)
                    # LOG.info('loss of validate data : %.2f' % val_loss)

                    LOG.info("Saving model, global_step: %d" % model.global_step.eval())
                    checkpoint_path = os.path.join(model.model_dir, "yolo3-epoch%03d.ckpt" % (epoch))
                    model.saver.save(sess, checkpoint_path, global_step=model.global_step, write_meta_graph=False)

    def validate(self, sess, model, data_gen, batch_size):
        """
        validate
        """

        total_loss = 0.0
        val_steps = int(round(data_gen.val_data_size / batch_size))
        if val_steps <= 0:
            return -1.0
        for _ in range(val_steps):  # Get a batch and make a step.

            batch_data = data_gen.next_batch_validate()
            if not batch_data:
                continue

            total_loss += model.step(sess, batch_data, False)

        return (total_loss / val_steps)

    def avg_checkpoints(self):
        """
        Average the last N checkpoints in the model_dir.
        """

        LOG.info("average checkpoints start .......")

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:

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

    def save_model_pb(self, saved_model_name):
        """
        save model as a single pb file from checkpoint
        """

        logging.info("save model as .pb start .......")

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:

            yolo_config = YoloConfig()

            model = Yolo3(sess, False, yolo_config)

            input_graph_def = sess.graph.as_graph_def()
            if flags.inference_device == '310D':
                output_tensors = model.output
            else:
                output_tensors = [model.boxes, model.scores, model.classes]
            print('output_tensors : ', output_tensors)
            output_tensors = [t.op.name for t in output_tensors]
            graph = tf.graph_util.convert_variables_to_constants(sess, input_graph_def, output_tensors)
            tf.train.write_graph(graph, model.model_dir, saved_model_name, False)

        logging.info("save model as .pb end .......")
