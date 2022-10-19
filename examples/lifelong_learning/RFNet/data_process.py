import os
import rosbag
from datetime import datetime
import cv2
from cv_bridge import CvBridge

bag_file = '/home/lsq/prefix_2022-10-14-10-51-24.bag'
bag = rosbag.Bag(bag_file, "r")
info = bag.get_type_and_topic_info()
bag_data = bag.read_messages('/camera_front_up/color/image_raw')
file_url = "/home/lsq/e1_1f_collections_1010"

bridge = CvBridge()
for i, (topic, msg, t) in enumerate(bag_data):
    secs = msg.header.stamp.secs
    nsecs = msg.header.stamp.nsecs
    timestr = f"{secs}.{nsecs}"

    timestr = datetime.fromtimestamp(float(timestr))
    timestr = str(timestr).replace(" ", "-").replace(":", "-")

    cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")

    file_path = os.path.join(file_url, f"{timestr}.png")
    cv2.imwrite(file_path, cv_image)