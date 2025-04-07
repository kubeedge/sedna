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

from abc import ABC

import json
import numpy as np
import pandas as pd
from pycocotools.coco import COCO

from pathlib import Path
from sedna.common.file_ops import FileOps
from sedna.common.class_factory import ClassFactory, ClassType

__all__ = (
    'BaseDataSource',
    'TxtDataParse',
    'CSVDataParse',
    'JSONDataParse',
    'JsonlDataParse',
    'JSONMetaDataParse'
)


class BaseDataSource:
    """
    An abstract class representing a :class:`BaseDataSource`.

    All datasets that represent a map from keys to data samples should subclass
    it. All subclasses should overwrite parse`, supporting get train/eval/infer
    data by a function. Subclasses could also optionally overwrite `__len__`,
    which is expected to return the size of the dataset.overwrite `x` for the
    feature-embedding, `y` for the target label.

    Parameters
    ----------
    data_type : str
        define the datasource is train/eval/test
    func: function
        function use to parse an iter object batch by batch
    """

    def __init__(self, data_type="train", func=None):
        self.data_type = data_type  # sample type: train/eval/test
        self.process_func = None
        if callable(func):
            self.process_func = func
        elif func:
            self.process_func = ClassFactory.get_cls(
                ClassType.CALLBACK, func)()
        self.x = None  # sample feature
        self.y = None  # sample label
        self.meta_attr = None  # special in lifelong learning

    def num_examples(self) -> int:
        return len(self.x)

    def __len__(self):
        return self.num_examples()

    def parse(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def is_test_data(self):
        return self.data_type == "test"

    def save(self, output=""):
        return FileOps.dump(self, output)


class TxtDataParse(BaseDataSource, ABC):
    """
    txt file which contain image list parser
    """

    def __init__(self, data_type, func=None):
        super(TxtDataParse, self).__init__(data_type=data_type, func=func)

    def parse(self, *args, **kwargs):
        x_data = []
        y_data = []
        use_raw = kwargs.get("use_raw")
        for f in args:
            if not (f and FileOps.exists(f)):
                continue
            with open(f) as fin:
                if self.process_func:
                    res = list(map(self.process_func, [
                        line.strip() for line in fin.readlines()]))
                else:
                    res = [line.strip().split() for line in fin.readlines()]
            for tup in res:
                if not len(tup):
                    continue
                if use_raw:
                    x_data.append(tup)
                else:
                    x_data.append(tup[0])
                    if not self.is_test_data:
                        if len(tup) > 1:
                            y_data.append(tup[1])
                        else:
                            y_data.append(0)
        self.x = np.array(x_data)
        self.y = np.array(y_data)


class CSVDataParse(BaseDataSource, ABC):
    """
    csv file which contain Structured Data parser
    """

    def __init__(self, data_type, func=None):
        super(CSVDataParse, self).__init__(data_type=data_type, func=func)

    @staticmethod
    def parse_json(lines: dict, **kwargs) -> pd.DataFrame:
        return pd.DataFrame.from_dict([lines], **kwargs)

    def parse(self, *args, **kwargs):
        x_data = []
        y_data = []
        label = kwargs.pop("label") if "label" in kwargs else ""
        usecols = kwargs.get("usecols", "")
        if usecols and isinstance(usecols, str):
            usecols = usecols.split(",")
        if len(usecols):
            if label and label not in usecols:
                usecols.append(label)
            kwargs["usecols"] = usecols
        for f in args:
            if isinstance(f, (dict, list)):
                res = self.parse_json(f, **kwargs)
            else:
                if not (f and FileOps.exists(f)):
                    continue
                res = pd.read_csv(f, **kwargs)
            if self.process_func and callable(self.process_func):
                res = self.process_func(res)
            if label:
                if label not in res.columns:
                    continue
                y = res[label]
                y_data.append(y)
                res.drop(label, axis=1, inplace=True)
            x_data.append(res)
        if not x_data:
            return
        self.x = pd.concat(x_data)
        self.y = pd.concat(y_data)


class JSONDataParse(BaseDataSource, ABC):
    """
    json file which contain Structured Data parser
    """

    def __init__(self, data_type, func=None):
        super(JSONDataParse, self).__init__(data_type=data_type, func=func)

    def parse(self, *args, **kwargs):
        DIRECTORY = "train"
        LABEL_PATH = "*/gt/gt_val_half.txt"
        filepath = Path(*args)
        self.data_dir = Path(Path(filepath).parents[1], DIRECTORY)
        self.coco = COCO(filepath)
        self.ids = self.coco.getImgIds()
        self.class_ids = sorted(self.coco.getCatIds())
        self.annotations = [self.load_anno_from_ids(_ids) for _ids in self.ids]
        self.x = {
            "data_dir": self.data_dir,
            "coco": self.coco,
            "ids": self.ids,
            "class_ids": self.class_ids,
            "annotations": self.annotations,
        }
        self.y = [f for f in self.data_dir.glob(LABEL_PATH)]

    def load_anno_from_ids(self, id_):
        im_ann = self.coco.loadImgs(id_)[0]
        width = im_ann["width"]
        height = im_ann["height"]
        frame_id = im_ann["frame_id"]
        video_id = im_ann["video_id"]
        anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=False)
        annotations = self.coco.loadAnns(anno_ids)
        objs = []
        for obj in annotations:
            x1 = obj["bbox"][0]
            y1 = obj["bbox"][1]
            x2 = x1 + obj["bbox"][2]
            y2 = y1 + obj["bbox"][3]
            if obj["area"] > 0 and x2 >= x1 and y2 >= y1:
                obj["clean_bbox"] = [x1, y1, x2, y2]
                objs.append(obj)

        num_objs = len(objs)
        res = np.zeros((num_objs, 6))

        for ix, obj in enumerate(objs):
            cls = self.class_ids.index(obj["category_id"])
            res[ix, 0:4] = obj["clean_bbox"]
            res[ix, 4] = cls
            res[ix, 5] = obj["track_id"]

        file_name = (
            im_ann["file_name"]
            if "file_name" in im_ann
            else f"{id_:012}.jpg"
        )
        img_info = (height, width, frame_id, video_id, file_name)

        del im_ann, annotations

        return (res, img_info, file_name)


class JsonlDataParse(BaseDataSource, ABC):
    """
    jsonl file which contain Structured Data parser
    """

    def __init__(self, data_type, func=None):
        super(JsonlDataParse, self).__init__(data_type=data_type, func=func)

    def parse(self, *args, **kwargs):
        x_data = []
        y_data = []
        for f in args:
            if not (f and FileOps.exists(f)):
                continue
            with open(f, 'r', encoding='utf-8') as file:
                for line in file:
                    line = json.loads(line)
                    x_data.append(line['question'])
                    y_data.append(line['answer'])
        self.x = np.array(x_data)
        self.y = np.array(y_data)


class JSONMetaDataParse(BaseDataSource, ABC):
    """
    parse data_info.json file
    """

    def __init__(self, data_type, func=None):
        super(JSONMetaDataParse, self).__init__(data_type=data_type, func=func)
        self.need_other_info = True

    def parse(self, *args, **kwargs):
        for f in args:
            if not (f and FileOps.exists(f)):
                continue
            with open(f, 'r', encoding='utf-8') as file:
                json_data = json.load(file)
                self.dataset_name = json_data['dataset']
                self.description = json_data['description']
                self.level_1_dim = json_data['level_1_dim']
                self.level_2_dim = json_data['level_2_dim']
                if 'level_3_dim' in json_data:
                    self.level_3_dim = json_data['level_3_dim']
                if 'level_4_dim' in json_data:
                    self.level_4_dim = json_data['level_4_dim']

            data_f = f.replace('metadata.json', 'data.jsonl')
            x_data = []
            y_data = []
            explanation_data = []
            judge_prompts = []
            level_3_data = []
            level_4_data = []
            with open(data_f, 'r', encoding='utf-8') as file:
                for line in file:
                    line = json.loads(line)
                    cur_x = ""
                    # "prompt" is optional
                    if 'prompt' in line:
                        cur_x += line['prompt']
                    cur_x += line['query']
                    x_data.append(cur_x)
                    y_data.append(line['response'])
                    # "explanation" is optional
                    cur_exp = ""
                    if 'explanation' in line:
                        cur_exp = line['explanation']
                    explanation_data.append(cur_exp)
                    # "judge_prompt" is optional
                    cur_jp = ""
                    if "judge_prompt" in line:
                        cur_jp += line['judge_prompt']
                    judge_prompts.append(cur_jp)
                    level_3_data.append(line['level_3_dim'])
                    level_4_data.append(line['level_4_dim'])

            self.x = np.array(x_data)
            self.y = np.array(y_data)
            self.explanation = np.array(explanation_data)
            self.judge_prompts = np.array(judge_prompts)
            self.level_3 = np.array(level_3_data)
            self.level_4 = np.array(level_4_data)
