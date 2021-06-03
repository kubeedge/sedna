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

from sqlalchemy import Column, String, Integer, Boolean, Text
from sqlalchemy import DateTime, Float, SmallInteger, ForeignKey, func
from sqlalchemy.orm import relationship

from .database import Base, engine

__all__ = ('TaskGrp', 'Tasks', 'TaskModel', 'TaskRelation', 'Samples',
           'TaskSample', 'get_or_create', 'engine', 'init_db')


class TaskGrp(Base):
    """Task groups"""
    __tablename__ = 'll_task_grp'

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    name = Column(String(100), unique=True, nullable=False,
                  comment='task group name')
    deploy = Column(Boolean(create_constraint=False), default=False)

    sample_num = Column(Integer, default=0, comment='int of sample number')
    task_num = Column(Integer, default=0, comment='int of task number')


class Tasks(Base):
    """Task table"""
    __tablename__ = 'll_tasks'

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    name = Column(String(100), unique=True,
                  nullable=False, comment='task name')
    task_attr = Column(Text, default="{}", comment='task attribute, json')
    created_at = Column(DateTime, server_default=func.now(),
                        comment='task create time')
    updated_at = Column(DateTime, server_default=func.now(),
                        onupdate=func.now(), comment='task update time')

    def __repr__(self):
        return self.name


class TaskModel(Base):
    """model belong tasks"""
    __tablename__ = 'll_task_models'

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    task_id = Column(Integer, ForeignKey('ll_task_grp.id'),
                     index=True, nullable=False)
    task = relationship('TaskGrp')
    model_url = Column(Text, default="", comment='model save url/path')
    is_current = Column(Boolean(create_constraint=False), default=False)
    created_at = Column(DateTime, server_default=func.now(),
                        comment='model create time')


class TaskRelation(Base):
    """ relation between two tasks"""
    __tablename__ = 'll_task_relation'

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    grp_id = Column(Integer, ForeignKey('ll_task_grp.id'),
                    index=True, nullable=False)
    grp = relationship('TaskGrp')
    task_id = Column(Integer, ForeignKey('ll_tasks.id'),
                     index=True, nullable=False)
    task = relationship('Tasks')
    transfer_radio = Column(Float, default=0.0, comment="task transfer radio")


class Samples(Base):
    """ Sample storage """
    __tablename__ = 'll_samples'

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    data_url = Column(Text, default="", comment='dataset save url/path')
    descr = Column(Text, default="", comment='dataset description')
    data_type = Column(
        SmallInteger,
        default=0,
        index=True,
        comment='type of dataset, 0: train, 1: evaluation, 2: hard sample')
    updated_at = Column(DateTime, server_default=func.now(),
                        onupdate=func.now(), comment='dataset update time')
    sample_num = Column(Integer, default=0, comment='int of sample number')


class TaskSample(Base):
    """ Sample of tasks"""
    __tablename__ = 'll_task_sample'

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    sample_id = Column(Integer, ForeignKey(
        'll_samples.id'), index=True, nullable=False)
    sample = relationship('Samples')
    task_id = Column(Integer, ForeignKey('ll_tasks.id'),
                     index=True, nullable=False)
    task = relationship('Tasks')


def get_or_create(session, model, **kwargs):
    instance = session.query(model).filter_by(**kwargs).first()
    if instance:
        return instance, False
    else:
        instance = model(**kwargs)
        return instance, True


def init_db():
    Base.metadata.create_all(bind=engine)
    TaskGrp.__table__.create(bind=engine, checkfirst=True)
    Tasks.__table__.create(bind=engine, checkfirst=True)
    TaskModel.__table__.create(bind=engine, checkfirst=True)
    TaskRelation.__table__.create(bind=engine, checkfirst=True)
    Samples.__table__.create(bind=engine, checkfirst=True)
    TaskSample.__table__.create(bind=engine, checkfirst=True)
