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
import base64
import json
import numpy as np

from robosdk.utils.class_factory import ClassType
from robosdk.utils.class_factory import ClassFactory
from robosdk.utils.exceptions import SensorError
from .base import DataCollectorBase
from .base import DataCollector


__all__ = ("RosDataCollector", "RosImages", "RosTopicCollector")


@ClassFactory.register(ClassType.GENERAL, alias="rospy/AnyMsg")
class RosDataCollector(DataCollectorBase):  # noqa
    ros_time_types = ['time', 'duration']
    ros_primitive_types = ['bool', 'byte', 'char', 'int8', 'uint8', 'int16',
                           'uint16', 'int32', 'uint32', 'int64', 'uint64',
                           'float32', 'float64', 'string']
    ros_header_types = ['Header', 'std_msgs/Header', 'roslib/Header']

    def __init__(self, name: str, msg_type: str, publisher: str = ""):
        super(RosDataCollector, self).__init__()

        import message_filters
        import roslib.message

        self.message_class = roslib.message.get_message_class(msg_type)
        if not self.message_class:
            raise SensorError(f"message class - {name} were unable to load.")
        self.sub = message_filters.Subscriber(self.topic, self.message_class)
        self.topic = name
        self.publisher = publisher

    def start(self):
        self.sub.registerCallback(self.parse)

    def close(self):
        self.sub.sub.unregister()

    def __del__(self):
        self.close()

    def parse(self, data):
        if data is None:
            return
        self.data_sub_lock.acquire()
        self.curr_frame_id += 1
        self.curr_frame = data
        json_data = self.convert_ros_message_to_dictionary(message=data)
        self.data_sub_lock.release()
        return json_data

    @staticmethod
    def _get_message_fields(message):
        return zip(message.__slots__, message._slot_types)

    @staticmethod
    def _convert_from_ros_binary(field_value):
        field_value = base64.b64encode(field_value).decode('utf-8')
        return field_value

    @staticmethod
    def _convert_from_ros_time(field_value):
        field_value = {
            'secs': field_value.secs,
            'nsecs': field_value.nsecs
        }
        return field_value

    def _convert_from_ros_array(self, field_type, field_value,
                                binary_array_as_bytes=True):
        # use index to raise ValueError if '[' not present
        list_type = field_type[:field_type.index('[')]
        return [self._convert_from_ros_type(
            list_type, value, binary_array_as_bytes) for value in field_value]

    @staticmethod
    def _is_ros_binary_type(field_type):
        """ Checks if the field is a binary array one, fixed size or not"""
        return field_type.startswith('uint8[') or field_type.startswith('char[')

    @staticmethod
    def _is_field_type_an_array(field_type):
        return field_type.find('[') >= 0

    def _is_field_type_a_primitive_array(self, field_type):
        bracket_index = field_type.find('[')
        if bracket_index < 0:
            return False
        else:
            list_type = field_type[:bracket_index]
            return list_type in self.ros_primitive_types

    def _convert_from_ros_type(self, field_type, field_value,
                               binary_array_as_bytes=True):
        if field_type in self.ros_primitive_types:
            field_value = str(field_value)
        elif field_type in self.ros_time_types:
            field_value = self._convert_from_ros_time(field_value)
        elif self._is_ros_binary_type(field_type):
            if binary_array_as_bytes:
                field_value = self._convert_from_ros_binary(field_value)
            elif type(field_value) == str:
                field_value = [ord(v) for v in field_value]
            else:
                field_value = list(field_value)
        elif self._is_field_type_a_primitive_array(field_type):
            field_value = list(field_value)
        elif self._is_field_type_an_array(field_type):
            field_value = self._convert_from_ros_array(
                field_type, field_value, binary_array_as_bytes)
        else:
            field_value = self.convert_ros_message_to_dictionary(
                field_value, binary_array_as_bytes)
        return field_value

    def convert_ros_message_to_dictionary(self, message,
                                          binary_array_as_bytes=False):
        """
        Takes in a ROS message and returns a Python dictionary.
        """

        dictionary = {}
        message_fields = self._get_message_fields(message)
        for field_name, field_type in message_fields:
            field_value = getattr(message, field_name)
            dictionary[field_name] = self._convert_from_ros_type(
                field_type, field_value, binary_array_as_bytes)
        return dictionary


@ClassFactory.register(ClassType.GENERAL, alias="sensor_msgs/Image")
class RosImages(RosDataCollector):
    def __init__(self, name: str, publisher: str = ""):
        super(RosImages, self).__init__(name=name, publisher=publisher,
                                        msg_type="sensor_msgs/Image")
        from cv_bridge import CvBridge
        self.cv_bridge = CvBridge()

    def parse(self, data):
        if data is None:
            return
        self.data_sub_lock.acquire()
        self.curr_frame_id += 1
        self.curr_frame = self.cv_bridge.imgmsg_to_cv2(data, data.encoding)
        json_data = dict(
            header={
                'seq': data.header.seq,
                'stamp': {
                    'secs': data.header.stamp.secs,
                    'nsecs': data.header.stamp.nsecs
                },
                'frame_id': data.header.frame_id
            },
            data=np.array(self.curr_frame).tolist(),
            height=data.height,
            width=data.width,
            encoding=data.encoding,
            is_bigendian=data.is_bigendian,
            step=data.step
        )
        self.data_sub_lock.release()
        return json_data


@ClassFactory.register(ClassType.GENERAL)
class RosTopicCollector(DataCollector):  # noqa

    def __init__(self):
        super(RosTopicCollector, self).__init__()

        import rostopic

        pubs, _ = rostopic.get_topic_list()
        self.logger.info(f"{len(pubs)} topics has found.")

        self.all_topics = []

        filter_topics = set()
        filter_topic_types = set()
        keep_topics = set()
        keep_topic_types = set()
        if self.config.IgnoreData:
            filter_topics = self.config.IgnoreData.get(
                "topic_name", "").split(",")
            filter_topic_types = self.config.IgnoreData.get(
                "topic_type", "").split(",")
        if self.config.ForcusData:
            keep_topics = self.config.ForcusData.get(
                "topic_name", "").split(",")
            keep_topic_types = self.config.ForcusData.get(
                "topic_type", "").split(",")
        for name, _type, publisher in pubs:
            if name in filter_topics or _type in filter_topic_types:
                self.logger.info(f"Skip sensor data of {name} - {_type} ...")
                continue
            if len(keep_topics) and name not in keep_topics:
                self.logger.info(f"Skip sensor data of {name} - {_type} ...")
                continue
            if len(keep_topic_types) and _type not in keep_topic_types:
                self.logger.info(f"Skip sensor data of {name} - {_type} ...")
                continue
            try:
                sub_cls = ClassFactory.get_cls(ClassType.GENERAL, _type)
            except ValueError:
                sub_cls = ClassFactory.get_cls(ClassType.GENERAL,
                                               "rospy/AnyMsg")
            try:
                sub = sub_cls(name=name, publisher=publisher)
            except (ValueError, SensorError):
                self.logger.warning(f"Sensor data of {name} "
                                    f"were unable to record.")
                continue
            self.all_topics.append(sub)

    def connect(self):
        import message_filters

        subs = [i.sub for i in self.all_topics]
        if not len(subs):
            raise SensorError("No data available.")
        sync = message_filters.ApproximateTimeSynchronizer(
            subs, queue_size=10, slop=0.2
        )
        sync.registerCallback(self.convert)
        self.logger.info(f"DataCollector connect successfully")

    def stop(self):
        self.logger.warning("trying to unsubscribe")
        map(lambda sub: sub.close(), self.all_topics)

    def convert(self, *all_data):
        data_collect = {}
        for inx, msg_data in enumerate(all_data):
            topic = self.all_topics[inx]
            data = topic.parse(msg_data)
            data_collect[topic.name] = data
        self.message_channel.add_data(json.dumps(data_collect))
