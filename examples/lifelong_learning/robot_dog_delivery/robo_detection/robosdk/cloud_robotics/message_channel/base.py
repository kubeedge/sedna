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
import ssl
import queue
import threading

from robosdk.utils.logger import logging
from robosdk.utils.context import MessageConfig
from robosdk.utils.constant import MsgChannelItem


class MSGChannelBase(threading.Thread):
    endpoint = MessageConfig.ENDPOINT

    def __init__(self):
        threading.Thread.__init__(self)
        self.logger = logging.bind(instance="message_channel", system=True)
        self.mq = queue.Queue(maxsize=MessageConfig.QUEUE_MAXSIZE)

    @staticmethod
    def build_ssl():
        context = ssl.SSLContext(
            protocol=MessageConfig.SSL_VERSION
        )
        if MessageConfig.CA_CERTS:
            context.load_verify_locations(MessageConfig.CA_CERTS)
        if MessageConfig.KEY_FILE and MessageConfig.CERT_FILE:
            context.load_cert_chain(MessageConfig.CERT_FILE,
                                    MessageConfig.KEY_FILE)
        if MessageConfig.CERT_REQS:
            context.verify_mode = MessageConfig.CERT_REQS
        if MessageConfig.CHECK_HOSTNAME:
            context.check_hostname = MessageConfig.CHECK_HOSTNAME
        return context

    def get_data(self):
        try:
            data = self.mq.get(timeout=MsgChannelItem.GET_MESSAGE_TIMEOUT.value)
        except queue.Empty:
            data = None
        return data

    def add_data(self, data: str):
        try:
            self.mq.put_nowait(data)
        except queue.Full:
            self.logger.warning('mq full, drop the message')
        except Exception as e:
            self.logger.error(f'mq add message fail: {e}')

    def run(self):
        raise NotImplementedError
