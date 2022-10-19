import argparse
import torchvision
import torch
import pickle
import xgboost
from PIL import Image

from sklearn.metrics import mean_squared_error
from sedna.common.config import Context, BaseConfig
from sedna.datasources import TxtDataParse

from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_args(**kwargs):
    parser = argparse.ArgumentParser(description="Regression Training")
    parser.add_argument("--regression-method", type=str,
                        default=kwargs.get("regression_method", "decision_tree"))

    args = parser.parse_args()
    return args

class ResnetModel:
    resnet_model = torchvision.models.resnet18(pretrained=True).to(device)
    resnet_model.eval()

    @classmethod
    def get_resnet_output(cls, data):
        outputs = []
        for x in data:
            output = cls.resnet_model(x)
            output = output.data.cpu().numpy()[0]
            outputs.append(output)

        return outputs

class MLRegression:
    def __init__(self, args):
        self.args = args

        if self.args.regression_method == "decision_tree":
            from sklearn.tree import DecisionTreeRegressor

            self.model = DecisionTreeRegressor()
        elif self.args.regression_method == "svr_regress":
            from sklearn.svm import SVR

            self.model = SVR()
        elif self.args.regression_method == "knn_regress":
            from sklearn.neighbors import KNeighborsRegressor

            self.model = KNeighborsRegressor()
        elif self.args.regression_method == "rf_regress":
            from sklearn.ensemble import RandomForestRegressor

            self.model = RandomForestRegressor()
        elif self.args.regression_method == "adaboost_regress":
            from sklearn.ensemble import AdaBoostRegressor

            self.model = AdaBoostRegressor()
        elif self.args.regression_method == "gb_regress":
            from sklearn.ensemble import GradientBoostingRegressor

            self.model = GradientBoostingRegressor()
        elif self.args.regression_method == "xgb_regress":
            from xgboost import XGBRegressor

            self.model = XGBRegressor()
        
    def train(self, train_data, valid_data=None, **kwargs):
        x_data, y_data = self._preprocess(train_data)

        self.model.fit(x_data, y_data)

    def evaluate(self, data, metric=None):
        x_data, y_data = self._preprocess(data)
        prediction = self.predict(x_data)
        score = metric(y_data, prediction)
        return score

    def predict(self, data):
        return self.model.predict(data)

    def load(self, model_url=None):
        if not model_url:
            raise Exception
        with open(model_url, "rb") as fin:
            self.model = pickle.load(fin)

    def save(self, model_name=None):
        with open(model_name, 'wb') as fin:
            pickle.dump(self.model, fin)

        return model_name

    def _preprocess(self, data):
        x_data, y_data = data.x, data.y
        y_data = list(map(lambda y: float(y), y_data))

        transformed_images = []
        for img_url in x_data:
            img = Image.open(img_url).convert('RGB')
            transformation = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                     std=(0.229, 0.224, 0.225))
            ])
            img = transformation(img).unsqueeze(0).to(device)

            transformed_images.append(img)

        x_data = ResnetModel.get_resnet_output(transformed_images)
        return x_data, y_data


def train():
    train_dataset_url = "./data_txt/regression_train.txt"
    train_data = TxtDataParse(data_type="train")
    train_data.parse(train_dataset_url, use_raw=False)
    regression_methods = ["decision_tree",
                         "svr_regress",
                         "knn_regress",
                         "rf_regress",
                         "adaboost_regress",
                         "gb_regress",
                         "xgb_regress"]

    for regression_method in regression_methods:
        args = set_args(regression_method=regression_method)

        regressor = MLRegression(args)
        regressor.train(train_data)
        regressor.save(f"./models/{regression_method}.pkl")


def eval():
    test_dataset_url = "./data_txt/regression_test.txt"
    test_data = TxtDataParse(data_type="eval")
    test_data.parse(test_dataset_url, use_raw=False)
    regression_methods = ["decision_tree",
                          "svr_regress",
                          "knn_regress",
                          "rf_regress",
                          "adaboost_regress",
                          "gb_regress",
                          "xgb_regress"]
    eval_result = {}
    for regression_method in regression_methods:
        args = set_args(regression_method=regression_method)

        regressor = MLRegression(args)
        regressor.load(f"./models/{regression_method}.pkl")
        score = regressor.evaluate(test_data, metric=mean_squared_error)
        eval_result.update({regression_method: score})

    print("Evaluation results")
    print(sorted(eval_result.items(), key=lambda kv: (kv[1], kv[0])))

if __name__ == '__main__':
    eval()
