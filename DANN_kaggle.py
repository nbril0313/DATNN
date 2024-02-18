import numpy as np
import pandas as pd
import plotly.express as px
from sklearn import datasets

moon_train = datasets.make_moons(n_samples=1000, shuffle=True, noise=0.05, random_state=None)
moon_test = datasets.make_moons(n_samples=1000, shuffle=True, noise=0.05, random_state=None)

import math


def rotate(origin, points, angle):
    """
    origin - tuple
    points - numpy array
    angle - number in radian
    """

    ox, oy = origin  # 기준점
    px, py = (points[:, 0], points[:, 1])  # 회전시킬 좌표

    # 회전 후 좌표
    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return np.stack((qx, qy), axis=1)


points_train, labels_train = (moon_train[0], moon_train[1])
points_test, labels_test = (moon_test[0], moon_test[1])

points_test = rotate((0, 1), points_test, 0.5)

x_train, y_train = (points_train[:, 0], points_train[:, 1])
domain_train = np.full((x_train.shape), 0)

x_test, y_test = (points_test[:, 0], points_test[:, 1])
domain_test = np.full((x_train.shape), 1)

# visulalization을 위한 label을 따로 생성
plabel_test = np.full((labels_test.shape), -1)

x = np.concatenate((x_train, x_test))  # train과 test x 좌표 모음
y = np.concatenate((y_train, y_test))  # train과 test y 좌표 모음

# real labels
labels = np.concatenate((labels_train, labels_test))

# train, test 구분 열
domain = np.concatenate((domain_train, domain_test))

# labels for train, plot 할 때 데이터를 구분하기 위한 열
plabels = np.concatenate((labels_train, plabel_test))

# dictionary 생성
d = {'x': x, 'y': y, 'labels': labels, 'domain': domain, 'plabels': plabels}
df = pd.DataFrame(data=d)

df.loc[(df['domain'] == 0) & (df['labels'] == 0), 'x'] *= 0.8
df.loc[(df['domain'] == 0) & (df['labels'] == 1), 'x'] *= 0.8

df.loc[(df['domain'] == 1) & (df['labels'] == 0), 'y'] -= 0.2
df.loc[(df['domain'] == 1) & (df['labels'] == 1), 'y'] -= 0.2

fig = px.scatter(df, x='x', y='y', color='plabels')
# fig.show()

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import torch.nn.functional as F
import torch.optim as optim

from plotly.subplots import make_subplots
import plotly.graph_objects as go


class MoonDataset(Dataset):
    def __init__(self, data, onehot=False):
        self.X = torch.Tensor(data.iloc[:, 0:2].to_numpy())
        self.y = torch.LongTensor(data.iloc[:, 2].to_numpy())
        if onehot:
            self.y = F.one_hot(self.y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X = self.X[idx, :]
        y = self.y[idx]
        return X, y


batch_size = 64

label_dataset = MoonDataset(df.loc[df['domain'] == 0, ['x', 'y', 'labels']])
domain_dataset = MoonDataset(df.loc[:, ['x', 'y', 'domain']])
test_dataset = MoonDataset(df.loc[df['domain'] == 1, ['x', 'y', 'labels']])

# Create data loaders.
label_dataloader = DataLoader(label_dataset, shuffle=True, batch_size=batch_size)
domain_dataloader = DataLoader(domain_dataset, shuffle=True, batch_size=batch_size * 2)
test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=batch_size * 2)


# Define model
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 4)
        )

    def forward(self, x):
        feature = self.linear_relu_stack(x)
        return feature


# Define model
class LabelPredictor(nn.Module):
    def __init__(self):
        super(LabelPredictor, self).__init__()

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 2)
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits


# Define model
class DomainClassifier(nn.Module):
    def __init__(self):
        super(DomainClassifier, self).__init__()

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits


Gf = FeatureExtractor()
Gy = LabelPredictor()

label_criteria = nn.NLLLoss()
label_optimizer = optim.Adam(list(Gf.parameters()) + list(Gy.parameters()), lr=1e-3)

epochs = 100

loss_log = []
acc_log = []
epoch_log = []

for epoch in range(epochs):
    acc_sum, loss_sum, cnt = (0, 0, 0)

    for data in label_dataloader:
        X, y = data

        sample_num = len(X)

        Gf.zero_grad()
        Gy.zero_grad()

        feature = Gf(X)
        pred = Gy(feature)

        pred = F.softmax(pred, dim=1)

        loss = label_criteria(pred, y)

        label_optimizer.zero_grad()
        loss.backward()
        label_optimizer.step()

        loss_sum += np.exp(loss.item())

        with torch.no_grad():
            feature = Gf(X)
            pred = Gy(feature)
            pred = F.softmax(pred, dim=1)
            acc_sum += torch.sum(torch.argmax(pred, dim=1) == y) / sample_num
        cnt += 1

    acc_log.append(acc_sum / cnt)
    loss_log.append(loss_sum / cnt)
    epoch_log.append(epoch)

