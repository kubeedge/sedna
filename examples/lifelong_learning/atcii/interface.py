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
import pandas as pd
import numpy as np
import xgboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score

os.environ['BACKEND_TYPE'] = 'SKLEARN'

DATACONF = {
    "ATTRIBUTES": ["Season", "Cooling startegy_building level"],
    "LABEL": "Thermal preference",
}


def feature_process(df: pd.DataFrame):
    if "City" in df.columns:
        df.drop(["City"], axis=1, inplace=True)
    for feature in df.columns:
        if feature in ["Season", ]:
            continue
        df[feature] = df[feature].apply(lambda x: float(x) if x else 0.0)
    df['Thermal preference'] = df['Thermal preference'].apply(
        lambda x: int(float(x)) if x else 1)
    return df


class Estimator:
    def __init__(self):
        """Model init"""
        self.model = xgboost.XGBClassifier(
            learning_rate=0.1,
            n_estimators=600,
            max_depth=2,
            min_child_weight=1,
            gamma=0,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softmax",
            num_class=3,
            nthread=4,
            seed=27)

    def train(self, train_data, valid_data=None,
              save_best=True,
              metric_name="mlogloss",
              early_stopping_rounds=100
              ):
        es = [
            xgboost.callback.EarlyStopping(
                metric_name=metric_name,
                rounds=early_stopping_rounds,
                save_best=save_best
            )
        ]
        x, y = train_data.x, train_data.y
        if valid_data:
            x1, y1 = valid_data.x, valid_data.y
        else:
            x, x1, y, y1 = train_test_split(
                x, y, test_size=0.1, random_state=42)
        history = self.model.fit(x, y, eval_set=[(x1, y1), ], callbacks=es)
        d = {}
        for k, v in history.evals_result().items():
            for k1, v1, in v.items():
                m = np.mean(v1)
                if k1 not in d:
                    d[k1] = []
                d[k1].append(m)
        for k, v in d.items():
            d[k] = np.mean(v)
        return d

    def predict(self, datas, **kwargs):
        """ Model inference """
        return self.model.predict(datas)

    def predict_proba(self, datas, **kwargs):
        return self.model.predict_proba(datas)

    def evaluate(self, test_data, **kwargs):
        """ Model evaluate """
        y_pred = self.predict(test_data.x)
        return precision_score(test_data.y, y_pred, average="micro")

    def load(self, model_url):
        self.model.load_model(model_url)
        return self

    def save(self, model_path=None):
        """
        save model as a single pb file from checkpoint
        """
        return self.model.save_model(model_path)


if __name__ == '__main__':
    from sedna.datasources import CSVDataParse
    from sedna.common.config import BaseConfig

    train_dataset_url = BaseConfig.train_dataset_url
    train_data = CSVDataParse(data_type="train", func=feature_process)
    train_data.parse(train_dataset_url, label=DATACONF["LABEL"])

    test_dataset_url = BaseConfig.test_dataset_url
    valid_data = CSVDataParse(data_type="valid", func=feature_process)
    valid_data.parse(test_dataset_url, label=DATACONF["LABEL"])

    model = Estimator()
    print(model.train(train_data))
    print(model.evaluate(test_data=valid_data))
