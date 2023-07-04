from PIL import Image
from tqdm import tqdm
import argparse

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import mean_squared_error
from sedna.common.config import Context, BaseConfig
from sedna.datasources import TxtDataParse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess(train_data):
    x_data, y_data = train_data.x, train_data.y
    # preprocess label
    y_data = list(map(lambda y: [float(y)], y_data))
    y_data = torch.tensor(y_data)
    # preprocess images
    transformed_images = []
    for img_url in x_data:
        img = Image.open(img_url).convert('RGB')
        transformation = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225))
        ])
        img = transformation(img).unsqueeze(0).to(device)

        transformed_images.append(img[0])

    return transformed_images, y_data

def set_args(**kwargs):
    parser = argparse.ArgumentParser(description="NeuralNetworkRegression Training")
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--test-batch-size", type=int, default=1)
    parser.add_argument("--num-epoch", type=int, default=200)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--cuda", action="store_true", default=torch.cuda.is_available())

    args = parser.parse_args()
    return args

class NNRregressionNet(nn.Module):
    def __init__(self, backbone, hidden_size):
        super(NNRregressionNet, self).__init__()
        self.backbone = backbone
        self.fc1 = nn.Linear(1000, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)
        self.relu2 = nn.ReLU()
        
    def forward(self, x):
        out = self.backbone(x)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out
    
class NNRegression():
    def __init__(self, args):
        backbone = torchvision.models.resnet18(pretrained=True).to(device)
        self.args = args
        self.model = NNRregressionNet(backbone, args.hidden_size).to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        
    def train(self, train_data, valid_data=None):
        x_data, y_data = preprocess(train_data)
        train_data = list(zip(x_data, y_data))

        train_data_loader = DataLoader(train_data, batch_size=self.args.batch_size, shuffle=True)

        for epoch in range(self.args.num_epoch):
            self.model.train()
            for i, (images, labels) in enumerate(tqdm(train_data_loader)):
                if self.args.cuda:
                    images = images.cuda()
                    labels = labels.cuda()

                # forward pass
                # 完整的模型为self.model
                # 假设forward只涉及第n+1层以后
                new_model = nn.Sequential(*list(self.model.children())[n+1:]) # 将模型切片，不确定这样切片对不对，可以尝试其他
                outputs = self.model(images)

                loss = self.criterion(outputs, labels)

                # backward and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def predict(self, data):
        self.model.eval()
        prediction = []

        test_data_loader = DataLoader(data, batch_size=self.args.test_batch_size, shuffle=False)
        for image in tqdm(test_data_loader):
            if self.args.cuda:
                image = image.cuda()
            with torch.no_grad():
                output = self.model(image)
            output = output.data.cpu().numpy()[0]
            prediction.append(output)

        return prediction

    def eval(self, data, metric=None):
        x_data, y_data = preprocess(data)
        y_data = y_data.data.cpu().numpy()
        prediction = self.predict(x_data)
        score = metric(y_data, prediction)
        return score

    def save(self, model_name):
        torch.save(self.model.state_dict(), model_name)
        return model_name

    def load(self, model_path):
        self.model.load_state_dict(torch.load(model_path))

def train():
    train_dataset_url = "./data_txt/regression_train.txt"
    train_data = TxtDataParse(data_type="train")
    train_data.parse(train_dataset_url, use_raw=False)
    args = set_args()
    regressor = NNRegression(args)
    regressor.train(train_data)
    regressor.save("./models/nn_regression.pth")

def eval():
    test_dataset_url = "./data_txt/regression_test.txt"
    test_data = TxtDataParse(data_type="eval")
    test_data.parse(test_dataset_url, use_raw=False)
    args = set_args()
    regressor = NNRegression(args)
    regressor.load("./models/nn_regression.pth")
    eval_result = regressor.eval(test_data, metric=mean_squared_error)

    print("MSE:", eval_result)

if __name__ == '__main__':
    eval()






        
        
        