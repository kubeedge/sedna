import logging

import numpy as np

import sedna

LOG = logging.getLogger(__name__)


def preprocess(image, input_shape):
    ih, iw = input_shape

    org_img_shape = w, h = image.size

    scale = min(iw / w, ih / h)
    nw, nh = int(scale * w), int(scale * h)
    image_resized = image.resize((nw, nh))

    image_paded = np.full(shape=[ih, iw, 3], fill_value=128.0)
    dw, dh = (iw - nw) // 2, (ih - nh) // 2
    image_paded[dh:nh + dh, dw:nw + dw, :] = image_resized
    image_paded = image_paded / 255.

    preprocessed_data = image_paded.astype(np.float32)[np.newaxis, :]
    return preprocessed_data, org_img_shape


def postprocess(data, ori_img_shape):
    pred_sbbox, pred_mbbox, pred_lbbox = data[1], data[2], data[0]
    num_classes = 4
    score_threshold = 0.3
    input_size = 544
    iou_threshold = 0.4
    sigma = 0.3
    pred_bbox = np.concatenate(
        [np.reshape(pred_sbbox, (-1, 5 + num_classes)),
         np.reshape(pred_mbbox, (-1, 5 + num_classes)),
         np.reshape(pred_lbbox, (-1, 5 + num_classes))], axis=0)

    valid_scale = [0, np.inf]
    pred_bbox = np.array(pred_bbox)

    pred_xywh = pred_bbox[:, 0:4]
    pred_conf = pred_bbox[:, 4]
    pred_prob = pred_bbox[:, 5:]

    # (1) (x, y, w, h) --> (xmin, ymin, xmax, ymax)
    pred_coor = np.concatenate(
        [pred_xywh[:, :2] - pred_xywh[:, 2:] * 0.5,
         pred_xywh[:, :2] + pred_xywh[:, 2:] * 0.5], axis=-1)
    # (2) (xmin, ymin, xmax, ymax) -> (xmin_org, ymin_org, xmax_org, ymax_org)
    org_w, org_h = ori_img_shape
    resize_ratio = min(1.0 * input_size / org_w, 1.0 * input_size / org_h)

    dw = (input_size - resize_ratio * org_w) / 2.
    dh = (input_size - resize_ratio * org_h) / 2.

    pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
    pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio

    # # (3) clip some boxes those are out of range
    pred_coor = np.concatenate(
        [np.maximum(pred_coor[:, :2], [0, 0]),
         np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1])], axis=-1)
    invalid_mask = np.logical_or((pred_coor[:, 0] > pred_coor[:, 2]),
                                 (pred_coor[:, 1] > pred_coor[:, 3]))
    pred_coor[invalid_mask] = 0

    # # (4) discard some invalidboxes
    bboxes_scale = np.sqrt(
        np.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1))
    scale_mask = np.logical_and((valid_scale[0] < bboxes_scale),
                                (bboxes_scale < valid_scale[1]))

    # # (5) discard some boxes with low scores
    classes = np.argmax(pred_prob, axis=-1)
    scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
    score_mask = scores > score_threshold
    # mask = np.logical_and(scale_mask, score_mask)
    mask = score_mask
    coors, scores, classes = pred_coor[mask], scores[mask], classes[mask]

    bboxes = np.concatenate(
        [coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1)
    bboxes = nms(bboxes, 0.4)
    return bboxes


def bboxes_iou(boxes1, boxes2):
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (
            boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (
            boxes2[..., 3] - boxes2[..., 1])

    left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    ious = np.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps)

    return ious


def nms(bboxes, iou_threshold, sigma=0.3, method='nms'):
    """
    :param bboxes: (xmin, ymin, xmax, ymax, score, class)

    Note: soft-nms, https://arxiv.org/pdf/1704.04503.pdf
          https://github.com/bharatsingh430/soft-nms
    """
    classes_in_img = list(set(bboxes[:, 5]))
    best_bboxes = []

    for cls in classes_in_img:
        cls_mask = (bboxes[:, 5] == cls)
        cls_bboxes = bboxes[cls_mask]

        while len(cls_bboxes) > 0:
            max_ind = np.argmax(cls_bboxes[:, 4])
            best_bbox = cls_bboxes[max_ind]
            best_bbox_ = best_bbox.tolist()

            # cast into int for cls
            best_bbox_[5] = int(best_bbox[5])

            best_bboxes.append(best_bbox_)
            cls_bboxes = np.concatenate(
                [cls_bboxes[: max_ind], cls_bboxes[max_ind + 1:]])
            iou = bboxes_iou(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
            weight = np.ones((len(iou),), dtype=np.float32)

            assert method in ['nms', 'soft-nms']

            if method == 'nms':
                iou_mask = iou > iou_threshold
                weight[iou_mask] = 0.0

            if method == 'soft-nms':
                weight = np.exp(-(1.0 * iou ** 2 / sigma))

            cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
            score_mask = cls_bboxes[:, 4] > 0.
            cls_bboxes = cls_bboxes[score_mask]

    return best_bboxes


def create_input_feed(sess, img_data):
    input_feed = {}

    input_img_data = sess.graph.get_tensor_by_name('input/input_data:0')
    input_feed[input_img_data] = img_data

    return input_feed


def create_output_fetch(sess):
    """Create output fetch for edge model inference"""
    pred_sbbox = sess.graph.get_tensor_by_name('pred_sbbox/concat_2:0')
    pred_mbbox = sess.graph.get_tensor_by_name('pred_mbbox/concat_2:0')
    pred_lbbox = sess.graph.get_tensor_by_name('pred_lbbox/concat_2:0')

    output_fetch = [pred_sbbox, pred_mbbox, pred_lbbox]
    return output_fetch


def run():
    input_shape_str = sedna.context.get_parameters("input_shape")
    input_shape = tuple(int(v) for v in input_shape_str.split(","))

    sedna.joint_inference.TSBigModelService(
        preprocess=preprocess,
        postprocess=postprocess,
        input_shape=input_shape,
        create_input_feed=create_input_feed,
        create_output_fetch=create_output_fetch
    )


if __name__ == "__main__":
    run()
