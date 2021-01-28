import logging
import os
import time

import cv2
import numpy as np

import sedna
from sedna.incremental_learning import InferenceResult

LOG = logging.getLogger(__name__)

he_saved_url = sedna.context.get_parameters('HE_SAVED_URL')

class_names = ['person', 'helmet', 'helmet_on', 'helmet_off']


def draw_boxes(img, labels, scores, bboxes, class_names, colors):
    line_type = 2
    text_thickness = 1
    box_thickness = 1
    #  get color code
    colors = colors.split(",")
    colors_code = []
    for color in colors:
        if color == 'green':
            colors_code.append((0, 255, 0))
        elif color == 'blue':
            colors_code.append((255, 0, 0))
        elif color == 'yellow':
            colors_code.append((0, 255, 255))
        else:
            colors_code.append((0, 0, 255))

    label_dict = {i: label for i, label in enumerate(class_names)}

    for i in range(bboxes.shape[0]):
        bbox = bboxes[i]
        if float("inf") in bbox or float("-inf") in bbox:
            continue
        label = int(labels[i])
        score = "%.2f" % round(scores[i], 2)
        text = label_dict.get(label) + ":" + score
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[2]), int(bbox[3]))
        if (p2[0] - p1[0] < 1) or (p2[1] - p1[1] < 1):
            continue
        cv2.rectangle(img, p1[::-1], p2[::-1], colors_code[labels[i]],
                      box_thickness)
        cv2.putText(img, text, (p1[1], p1[0] + 20 * (label + 1)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0),
                    text_thickness, line_type)

    return img


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


def output_deal(inference_result: InferenceResult, nframe, img_rgb):
    # save and show image
    img_rgb = np.array(img_rgb)
    img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    colors = 'yellow,blue,green,red'
    if inference_result.is_hard_example:
        lables, scores, bbox_list_pred = inference_result.infer_result
        img = draw_boxes(img_rgb, lables, scores, bbox_list_pred, class_names,
                         colors)
        cv2.imwrite(f"{he_saved_url}/{nframe}.jpeg", img)


def mkdir(path):
    path = path.strip()
    path = path.rstrip()
    is_exists = os.path.exists(path)
    if not is_exists:
        os.makedirs(path)
        LOG.info(f"{path} is not exists, create the dir")


def run():
    input_shape_str = sedna.context.get_parameters("input_shape")
    input_shape = tuple(int(v) for v in input_shape_str.split(","))
    camera_address = sedna.context.get_parameters('video_url')

    mkdir(he_saved_url)

    # create little model object
    model = sedna.incremental_learning.TSModel(
        preprocess=preprocess,
        input_shape=input_shape,
        create_input_feed=create_input_feed,
        create_output_fetch=create_output_fetch
    )

    # create inference object
    inference_instance = sedna.incremental_learning.Inference(model)

    # use video streams for testing
    camera = cv2.VideoCapture(camera_address)
    fps = 10
    nframe = 0
    # the input of video stream
    while 1:
        ret, input_yuv = camera.read()
        if not ret:
            LOG.info(
                f"camera is not open, camera_address={camera_address},"
                f" sleep 5 second.")
            time.sleep(5)
            camera = cv2.VideoCapture(camera_address)
            continue

        if nframe % fps:
            nframe += 1
            continue

        img_rgb = cv2.cvtColor(input_yuv, cv2.COLOR_BGR2RGB)
        nframe += 1
        LOG.info(f"camera is open, current frame index is {nframe}")
        inference_result = inference_instance.inference(img_rgb)
        output_deal(inference_result, nframe, img_rgb)


if __name__ == "__main__":
    run()
