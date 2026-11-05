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

import os
import tempfile
import unittest
from unittest import mock

from sedna.service import client
from sedna.service.client import KBClient


class KBClientWithoutServerTest(unittest.TestCase):
    """KBClient must not issue any request when no KB server is set."""

    def setUp(self):
        patcher = mock.patch.object(client, "http_request")
        self.http_request = patcher.start()
        self.addCleanup(patcher.stop)

    def test_no_request_when_server_unset(self):
        for server in (None, ""):
            kb = KBClient(kbserver=server)
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                path = tmp.name
            self.addCleanup(lambda p=path: os.path.exists(p) and os.remove(p))

            self.assertEqual(kb.upload_file(path), path)
            self.assertTrue(os.path.exists(path))
            self.assertIsNone(kb.update_db(path))
            self.assertIsNone(kb.update_task_status("task"))
        self.http_request.assert_not_called()

    def test_request_when_server_set(self):
        self.http_request.return_value = "http://kb/knowledgebase/x"
        kb = KBClient(kbserver="http://kb")
        self.assertEqual(kb.update_task_status("task"),
                         "http://kb/knowledgebase/x")
        self.http_request.assert_called_once()


if __name__ == "__main__":
    unittest.main()
