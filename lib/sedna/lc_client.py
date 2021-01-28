import logging
import os
import time

import requests

LOG = logging.getLogger(__name__)


class LCClientConfig:
    def __init__(self):
        self.lc_server = os.getenv("LC_SERVER", "http://127.0.0.1:9100")


class LCClient:
    _retry = 3
    _retry_interval_seconds = 0.5
    config = LCClientConfig()

    @classmethod
    def send(cls, worker_name, message: dict):

        url = '{0}/sedna/workers/{1}/info'.format(
            cls.config.lc_server,
            worker_name
        )
        error = None
        for i in range(cls._retry):
            try:
                res = requests.post(url=url, json=message)
                LOG.info(
                    f"send to lc, url={url}, data={message},"
                    f"state={res.status_code}")
                return res.status_code < 300
            except Exception as e:
                error = e
                time.sleep(cls._retry_interval_seconds)

        LOG.warning(
            f"can't connect to lc[{cls.config.lc_server}] "
            f"data={message}, error={error}, "
            f"retry times: {cls._retry}")
        return False
