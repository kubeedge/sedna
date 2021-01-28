import logging

import numpy as np
import random
import tensorflow as tf
from PIL import Image
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

LOG = logging.getLogger(__name__)

flags = tf.flags.FLAGS


class DataGen(object):

    def __init__(self, config, train_data, valid_data):

        LOG.info("DataGen build start .......")

        self.input_shape = flags.input_shape

        self.batch_size = flags.batch_size
        self.anchors = np.array([float(x) for x in config.anchors]).reshape(-1, 2)
        self.class_names = flags.class_names
        self.num_classes = len(self.class_names)
        self.max_boxes = config.max_boxes

        self.train_curr_index = 0
        self.train_data = train_data
        self.train_data_size = len(self.train_data)
        LOG.info('size of train data is : %d' % self.train_data_size)

        self.val_curr_index = 0
        self.val_data = valid_data
        self.val_data_size = len(self.val_data)
        LOG.info('size of validation data is : %d' % self.val_data_size)

        self.batch_index = 0
        self.cur_shape = flags.input_shape

        LOG.info("DataGen build end .......")

    def next_batch_train(self):
        multi_scales = [self.input_shape]
        for i in range(1, 3):
            multi_scales.append((self.input_shape[0] - 32 * i, self.input_shape[1] - 32 * i))
            multi_scales.append((self.input_shape[0] + 32 * i, self.input_shape[1] + 32 * i))

        if self.batch_index % 25 == 0:
            self.cur_shape = random.choice(multi_scales)

        self.batch_index += 1
        count, batch_data = self.next_batch(self.train_curr_index, self.train_data, self.train_data_size,
                                            self.cur_shape, True)

        if not count:
            self.train_curr_index = 0
            random.shuffle(self.train_data)
            return None
        else:
            self.train_curr_index += count
            batch_data['input_shape'] = self.cur_shape
            return batch_data

    def next_batch_validate(self):
        count, batch_data = self.next_batch(self.val_curr_index, self.val_data, self.val_data_size, self.input_shape,
                                            False)
        if not count:
            self.val_curr_index = 0
            return None
        else:
            self.val_curr_index += count
            return batch_data

    def next_batch(self, curr_index, dataset, data_size, input_shape, is_training):

        count = 0
        img_data_list = []
        box_data_list = []
        while curr_index < data_size:
            if curr_index % 10000 == 0:
                LOG.info("processing label line %d" % curr_index)

            curr_line = dataset[curr_index]
            count += 1
            curr_index += 1

            if len(curr_line.strip()) <= 0:
                LOG.info("current line length less than 0......")
                continue

            image_data, box_data = self.read_data(curr_line, input_shape, is_training, self.max_boxes)
            if image_data is None or box_data is None:
                continue

            img_data_list.append(image_data)
            box_data_list.append(box_data)

            if len(img_data_list) >= self.batch_size:
                batch_data = dict()
                batch_data['images'] = np.array(img_data_list)
                bbox_true_13, bbox_true_26, bbox_true_52 = self.preprocess_true_boxes(np.array(box_data_list),
                                                                                      input_shape)
                batch_data['bbox_true_13'] = bbox_true_13  # np.array(bbox_13_list)
                batch_data['bbox_true_26'] = bbox_true_26  # np.array(bbox_26_list)
                batch_data['bbox_true_52'] = bbox_true_52  # np.array(bbox_52_list)
                return count, batch_data

        LOG.info('reaching the last line of data ~~~')
        return None, None

    def rand(self, a=0., b=1.):
        return np.random.rand() * (b - a) + a

    def read_data(self, annotation_line, input_shape=416, random=True, max_boxes=50, jitter=.3, hue=.1, sat=1.5,
                  val=1.5, proc_img=True):
        """
        random preprocessing for real-time data augmentation
        """

        line = annotation_line.split()
        image = Image.open(line[0])
        iw, ih = image.size
        h, w = input_shape
        box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])

        if not random:
            # resize image
            scale = min(float(w) / float(iw), float(h) / float(ih))
            nw = int(iw * scale)
            nh = int(ih * scale)
            dx = (w - nw) // 2
            dy = (h - nh) // 2
            image_data = 0
            if proc_img:
                image = image.resize((nw, nh), Image.BICUBIC)
                new_image = Image.new('RGB', (w, h), (128, 128, 128))
                new_image.paste(image, (dx, dy))
                image_data = np.array(new_image) / 255.

            # correct boxes
            box_data = np.zeros((max_boxes, 5))
            if len(box) > 0:
                np.random.shuffle(box)
                if len(box) > max_boxes: box = box[:max_boxes]
                box[:, [0, 2]] = box[:, [0, 2]] * scale + dx
                box[:, [1, 3]] = box[:, [1, 3]] * scale + dy
                box_data[:len(box)] = box
                return image_data, box_data
            else:
                return None, None

        # resize image
        new_ar = float(w) / float(h) * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
        scale = self.rand(.25, 2)

        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)

        # place image
        dx = int(self.rand(0, w - nw))
        dy = int(self.rand(0, h - nh))
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image = new_image

        # flip image or not
        flip = self.rand() < .5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

        # convert image to gray or not
        gray = self.rand() < .25
        if gray: image = image.convert('L').convert('RGB')

        # distort image
        hue = self.rand(-hue, hue)
        sat = self.rand(1, sat) if self.rand() < .5 else 1 / self.rand(1, sat)
        val = self.rand(1, val) if self.rand() < .5 else 1 / self.rand(1, val)
        x = rgb_to_hsv(np.array(image) / 255.)
        x[..., 0] += hue
        x[..., 0][x[..., 0] > 1] -= 1
        x[..., 0][x[..., 0] < 0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x > 1] = 1
        x[x < 0] = 0
        image_data = hsv_to_rgb(x)  # numpy array, 0 to 1

        # correct boxes
        box_data = np.zeros((max_boxes, 5))
        if len(box) > 0:
            np.random.shuffle(box)
            box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
            box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
            if flip: box[:, [0, 2]] = w - box[:, [2, 0]]
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > w] = w
            box[:, 3][box[:, 3] > h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box

            if len(box) > max_boxes:
                box = box[:max_boxes]
            if len(box) == 0:
                return None, None

            box_data[:len(box)] = box

        return image_data, box_data

    def preprocess_true_boxes(self, true_boxes, in_shape=416):
        """Preprocesses the ground truth box of the training data

        :param true_boxes: ground truth box shape is [boxes, 5], x_min, y_min,
            x_max, y_max, class_id
        """

        num_layers = self.anchors.shape[0] // 3
        anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        true_boxes = np.array(true_boxes, dtype='float32')
        # input_shape = np.array([in_shape, in_shape], dtype='int32')
        input_shape = np.array(in_shape, dtype='int32')
        boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2.
        boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
        true_boxes[..., 0:2] = boxes_xy / input_shape[::-1]
        true_boxes[..., 2:4] = boxes_wh / input_shape[::-1]

        m = true_boxes.shape[0]
        grid_shapes = [input_shape // 32, input_shape // 16, input_shape // 8]
        y_true = [np.zeros((m, grid_shapes[l][0], grid_shapes[l][1], len(anchor_mask[l]), 5 + self.num_classes),
                           dtype='float32') for l in range(num_layers)]
        # The dimension is expanded to calculate the IOU between the
        # anchors of all boxes in each graph by broadcasting
        anchors = np.expand_dims(self.anchors, 0)
        anchors_max = anchors / 2.
        anchors_min = -anchors_max
        # Because we padded the box before, we need to remove all 0 lines
        valid_mask = boxes_wh[..., 0] > 0

        for b in range(m):
            wh = boxes_wh[b, valid_mask[b]]
            if len(wh) == 0: continue

            # Expanding dimensions for broadcasting applications
            wh = np.expand_dims(wh, -2)
            # wh shape is [box_num, 1, 2]
            boxes_max = wh / 2.
            boxes_min = -boxes_max

            intersect_min = np.maximum(boxes_min, anchors_min)
            intersect_max = np.minimum(boxes_max, anchors_max)
            intersect_wh = np.maximum(intersect_max - intersect_min, 0.)
            intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
            box_area = wh[..., 0] * wh[..., 1]
            anchor_area = anchors[..., 0] * anchors[..., 1]
            iou = intersect_area / (box_area + anchor_area - intersect_area)

            # Find out the largest anchor box with the IOU of the ground truth
            # box, and then set the corresponding positions of different
            # proportions responsible for the ground turn box as the
            # coordinates of the ground truth box
            best_anchor = np.argmax(iou, axis=-1)
            for t, n in enumerate(best_anchor):
                for l in range(num_layers):
                    if n in anchor_mask[l]:
                        i = np.floor(true_boxes[b, t, 0] * grid_shapes[l][1]).astype('int32')
                        j = np.floor(true_boxes[b, t, 1] * grid_shapes[l][0]).astype('int32')
                        k = anchor_mask[l].index(n)

                        c = true_boxes[b, t, 4].astype('int32')
                        y_true[l][b, j, i, k, 0:4] = true_boxes[b, t, 0:4]
                        y_true[l][b, j, i, k, 4] = 1.
                        y_true[l][b, j, i, k, 5 + c] = 1.
        return y_true[0], y_true[1], y_true[2]
