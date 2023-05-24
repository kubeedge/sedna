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

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from sedna.common.config import Context


SQLALCHEMY_DATABASE_URL = Context.get_parameters(
    "KB_URL", "sqlite:///lifelong_kb.sqlite3")

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, encoding='utf-8',
    echo=True, connect_args={'check_same_thread': False}
)


SessionLocal = sessionmaker(
    bind=engine, autoflush=False, autocommit=False, expire_on_commit=True)

Base = declarative_base(bind=engine, name='Base')
