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
import re

from sedna.service.server.knowledgeBase.model import init_db
from sedna.service.server.knowledgeBase.server import KBServer


def main():
    init_db()
    server = os.getenv("KnowledgeBaseServer", "")
    kb_dir = os.getenv("KnowledgeBasePath", "")
    match = re.compile(
        "(https?)://([0-9]{1,3}(?:\\.[0-9]{1,3}){3}):([0-9]+)").match(server)
    if match:
        _, host, port = match.groups()
    else:
        host, port = '0.0.0.0', 9020
    KBServer(host=host, http_port=int(port), save_dir=kb_dir).start()


if __name__ == '__main__':
    main()
