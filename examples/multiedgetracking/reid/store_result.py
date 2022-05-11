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

from pathlib import Path
import os
import cv2
from PIL import Image

from sedna.common.log import LOGGER
from sedna.core.multi_edge_tracking.data_classes import DetTrackResult
from sedna.algorithms.reid.close_contact_estimation import ContactTracker

def create_results_folder(folder):
    """Creates result folder if it doesn't exist"""
    Path(folder).mkdir(parents=False, exist_ok=True)

def write_text_on_image(img, text, textX, textY, color=(0, 255, 0)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    height, width, _ = img.shape
    tl = round(0.002 * (height + width) * 0.1) + 1

    # add text centered on image
    cv2.putText(img, text, (textX, textY), font, 0.5, color, tl)

def draw_bbox(image, bbox, text="CARRIER", color=(255,0,0), tracking_id=None):
    height, width, _ = image.shape
    tl = round(0.002 * (height + width) * 0.1) + 1  # line thickness
    
    target_centroid = (int((bbox[0]+bbox[2])/2), int((bbox[1]+bbox[3])/2))
    write_text_on_image(image, text, int(bbox[0]), int(bbox[1])-10, color)

    if tracking_id:
        write_text_on_image(image, str(tracking_id), int(bbox[0]), int(bbox[3])-10, color)

    cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, tl)
    cv2.circle(image, target_centroid, radius=3, color=color, thickness=tl)

    return target_centroid

def save_image(data : DetTrackResult, CT : ContactTracker, folder):
    image_name = data.image_key.rsplit("_")[0] + ".png"
    image = cv2.imdecode(data.scene, cv2.IMREAD_COLOR)
    
    # We need the original image to avoid drawing artifacts when performing some operations
    subjects_at_risk = [data.targetID[data.is_target]]

    # TARGET 
    idx = data.is_target
                
    target_centroid = draw_bbox(image, data.bbox_coord[idx], tracking_id=data.tracklets[idx])
    CT.prep_homography(image.shape, data.bbox_coord[idx])

    # NON-TARGETS
    for lidx, bbox in enumerate(data.bbox_coord):
        # Draw bounding box
        if lidx != idx:
            other_centroid = (int((bbox[0]+bbox[2])/2), int((bbox[1]+bbox[3])/2))

            in_risk = CT.in_risk_zone(image, bbox)

            if in_risk:
                draw_bbox(image, bbox, "P-RISK", (255,69,0), tracking_id=data.tracklets[lidx])
                cv2.line(image, target_centroid, other_centroid, (0, 255, 0), thickness=1)
                cv2.circle(image, other_centroid, radius=3, color=(255, 0, 0), thickness=1)
                subjects_at_risk.append(data.targetID[lidx])

    # Add camera
    write_text_on_image(image, f"Camera:{data.camera}", 0, 30)
    
    # Save result in the specified folder
    create_results_folder(folder)
    base_image = Image.fromarray(image).resize((640, 480))
    base_image.save(os.path.join(folder, image_name), quality=80, optimize=True)

    return image_name
