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

FROM python:3.6

RUN mkdir /code
COPY ./lib/requirements.txt /code
RUN pip install -r /code/requirements.txt
RUN pip install python-multipart
RUN pip install SQLAlchemy
RUN pip install aiofiles
ENV PYTHONPATH "/code/lib"

COPY ./lib /code/lib


ENTRYPOINT ["python", "/code/lib/sedna/service/run_kb.py"]
