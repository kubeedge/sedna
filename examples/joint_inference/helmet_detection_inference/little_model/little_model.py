import logging
import time
import copy

import cv2
import numpy as np
import os

import sedna
from sedna.hard_example_mining import IBTFilter
from sedna.joint_inference.joint_inference import InferenceResult

LOG = logging.getLogger(__name__)

# Predefined color values for frames and display categories
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
          (0, 255, 255), (255, 255, 255)]
class_names = ['person', 'helmet', 'helmet_on', 'helmet_off']
all_output_path = sedna.context.get_parameters(
    'all_examples_inference_output'
)
hard_example_edge_output_path = sedna.context.get_parameters(
    'hard_example_edge_inference_output'
)
hard_example_cloud_output_path = sedna.context.get_parameters(
    'hard_example_cloud_inference_output'
)


def draw_boxes(img, bboxes, colors, text_thickness, box_thickness):
    img_copy = copy.deepcopy(img)

    line_type = 2
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

    for bbox in bboxes:
        if float("inf") in bbox or float("-inf") in bbox:
            continue
        label = int(bbox[5])
        score = "%.2f" % round(bbox[4], 2)
        text = label_dict.get(label) + ":" + score
        p1 = (int(bbox[1]), int(bbox[0]))
        p2 = (int(bbox[3]), int(bbox[2]))
        if (p2[0] - p1[0] < 1) or (p2[1] - p1[1] < 1):
            continue
        cv2.rectangle(img_copy, p1[::-1], p2[::-1], colors_code[label],
                      box_thickness)
        cv2.putText(img_copy, text, (p1[1], p1[0] + 20 * (label + 1)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0),
                    text_thickness, line_type)

    return img_copy


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
    output_classes = sess.graph.get_tensor_by_name('concat_19:0')
    output_scores = sess.graph.get_tensor_by_name('concat_18:0')
    output_boxes = sess.graph.get_tensor_by_name('concat_17:0')

    output_fetch = [output_classes, output_scores, output_boxes]
    return output_fetch


def postprocess(model_output):
    all_classes, all_scores, all_bboxes = model_output
    bboxes = []
    for c, s, bbox in zip(all_classes, all_scores, all_bboxes):
        bbox[0], bbox[1], bbox[2], bbox[3] = bbox[1], bbox[0], bbox[3], bbox[2]
        bboxes.append(bbox.tolist() + [s, c])

    return bboxes


def output_deal(inference_result: InferenceResult, nframe, img_rgb):
    # save and show image
    img_rgb = np.array(img_rgb)
    img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    collaboration_frame = draw_boxes(img_rgb, inference_result.final_result,
                                     colors="green,blue,yellow,red",
                                     text_thickness=None,
                                     box_thickness=None)

    cv2.imwrite(f"{all_output_path}/{nframe}.jpeg", collaboration_frame)

    # save hard example image to dir
    if not inference_result.is_hard_example:
        return

    if inference_result.hard_example_cloud_result is not None:
        cv2.imwrite(f"{hard_example_cloud_output_path}/{nframe}.jpeg",
                    collaboration_frame)
    edge_collaboration_frame = draw_boxes(
        img_rgb,
        inference_result.hard_example_edge_result,
        colors="green,blue,yellow,red",
        text_thickness=None,
        box_thickness=None)
    cv2.imwrite(f"{hard_example_edge_output_path}/{nframe}.jpeg",
                edge_collaboration_frame)


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

    mkdir(all_output_path)
    mkdir(hard_example_edge_output_path)
    mkdir(hard_example_cloud_output_path)

    # create little model object
    model = sedna.joint_inference.TSLittleModel(
        preprocess=preprocess,
        postprocess=postprocess,
        input_shape=input_shape,
        create_input_feed=create_input_feed,
        create_output_fetch=create_output_fetch
    )
    # create hard example algorithm
    threshold_box = float(sedna.context.get_hem_parameters(
        "threshold_box", 0.5
    ))
    threshold_img = float(sedna.context.get_hem_parameters(
        "threshold_img", 0.5
    ))
    hard_example_mining_algorithm = IBTFilter(threshold_img, threshold_box)

    # create joint inference object
    inference_instance = sedna.joint_inference.JointInference(
        little_model=model,
        hard_example_mining_algorithm=hard_example_mining_algorithm
    )

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
