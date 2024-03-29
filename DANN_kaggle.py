import math
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn import datasets
from torch.autograd import Function

moon_train = datasets.make_moons(n_samples=1000, shuffle=True, noise=0.05, random_state=None)
moon_test = datasets.make_moons(n_samples=1000, shuffle=True, noise=0.05, random_state=None)


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


# Gf = FeatureExtractor()
# Gy = LabelPredictor()
#
# label_criteria = nn.NLLLoss()
# label_optimizer = optim.Adam(list(Gf.parameters()) + list(Gy.parameters()), lr=1e-3)
#
# epochs = 100
#
# loss_log = []
# acc_log = []
# epoch_log = []
#
# for epoch in range(epochs):
#     acc_sum, loss_sum, cnt = (0, 0, 0)
#
#     for data in label_dataloader:
#         X, y = data
#
#         sample_num = len(X)
#
#         Gf.zero_grad()
#         Gy.zero_grad()
#
#         feature = Gf(X)
#         pred = Gy(feature)
#
#         pred = F.softmax(pred, dim=1)
#
#         loss = label_criteria(pred, y)
#
#         label_optimizer.zero_grad()
#         loss.backward()
#         label_optimizer.step()
#
#         loss_sum += np.exp(loss.item())
#
#         with torch.no_grad():
#             feature = Gf(X)
#             pred = Gy(feature)
#             pred = F.softmax(pred, dim=1)
#             acc_sum += torch.sum(torch.argmax(pred, dim=1) == y) / sample_num
#         cnt += 1
#
#     acc_log.append(acc_sum / cnt)
#     loss_log.append(loss_sum / cnt)
#     epoch_log.append(epoch)
#
# fig = make_subplots(rows=1, cols=2)
#
# fig.add_trace(
#     go.Scatter(x=epoch_log, y=loss_log, name='loss'),
#     row=1, col=1
# )
#
# fig.add_trace(
#     go.Scatter(x=epoch_log, y=acc_log, name='accuracy'),
#     row=1, col=2,
# )
#
# fig.update_layout(height=450, width=900, title_text="Train label Results")
# # fig.show()
#
# Gf = FeatureExtractor()
# Gd = DomainClassifier()
#
# domain_criteria = nn.CrossEntropyLoss()
# domain_optimizer = optim.Adam(list(Gf.parameters()) + list(Gd.parameters()), lr=1e-3)
#
# epochs = 100
#
# loss_log = []
# acc_log = []
# epoch_log = []
#
# for epoch in range(epochs):
#     acc_sum, loss_sum, cnt = (0, 0, 0)
#     for data in domain_dataloader:
#         X, y = data
#
#         sample_num = len(X)
#
#         Gf.zero_grad()
#         Gd.zero_grad()
#
#         feature = Gf(X)
#         pred = Gd(feature)
#
#         loss = domain_criteria(pred, y)
#
#         domain_optimizer.zero_grad()
#         loss.backward()
#         domain_optimizer.step()
#
#         loss_sum += loss.item()
#
#         with torch.no_grad():
#             feature = Gf(X)
#             pred = Gd(feature)
#             acc_sum += torch.sum(torch.argmax(pred, dim=1) == y) / sample_num
#         cnt += 1
#
#     acc_log.append(acc_sum / cnt)
#     loss_log.append(loss_sum / cnt)
#     epoch_log.append(epoch)
#
# fig = make_subplots(rows=1, cols=2)
#
# fig.add_trace(
#     go.Scatter(x=epoch_log, y=loss_log, name='loss'),
#     row=1, col=1
# )
#
# fig.add_trace(
#     go.Scatter(x=epoch_log, y=acc_log, name='accuracy'),
#     row=1, col=2
# )
#
# fig.update_layout(height=450, width=900, title_text="Train domain Results")
# # fig.show()
#
# Gf = FeatureExtractor()
# Gy = LabelPredictor()
#
# label_criteria = nn.NLLLoss()
#
# epochs = 200
#
# mu = torch.FloatTensor([0.01])
#
# loss_log = []
# acc_log = []
# epoch_log = []
#
# for epoch in range(epochs):
#     acc_sum, loss_sum, cnt = (0, 0, 0)
#
#     for data in label_dataloader:
#         X, y = data
#
#         sample_num = len(X)
#
#         Gf.zero_grad()
#         Gy.zero_grad()
#
#         feature = Gf(X)
#         pred = Gy(feature)
#
#         pred = F.softmax(pred, dim=1)
#         loss = label_criteria(pred, y)
#
#         loss.backward()
#         loss_sum += np.exp(loss.item())
#
#         with torch.no_grad():
#             # update Gf
#             for param in Gf.parameters():
#                 param -= mu * param.grad
#
#             # update Gy
#             for param in Gy.parameters():
#                 param -= mu * param.grad
#
#         with torch.no_grad():
#             feature = Gf(X)
#             pred = Gy(feature)
#             pred = F.softmax(pred, dim=1)
#             acc_sum += torch.sum(torch.argmax(pred, dim=1) == y) / sample_num
#         cnt += 1
#
#     acc_log.append(acc_sum / cnt)
#     loss_log.append(loss_sum / cnt)
#     epoch_log.append(epoch)
#
# fig = make_subplots(rows=1, cols=2)
#
# fig.add_trace(
#     go.Scatter(x=epoch_log, y=loss_log, name='loss'),
#     row=1, col=1
# )
#
# fig.add_trace(
#     go.Scatter(x=epoch_log, y=acc_log, name='acc'),
#     row=1, col=2
# )
#
# fig.update_layout(height=450, width=900, title_text="Manual weights update, label loss and acc")
# # fig.show()


class GradientReversalFn(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha  # Reverse gradient
        return output, None


# Gf = FeatureExtractor()
# Gd = DomainClassifier()
#
# domain_criteria = nn.CrossEntropyLoss()
#
# # Define optimizers for Gf and Gd
# optimizer_Gf = optim.Adam(Gf.parameters(), lr=1e-3)
# optimizer_Gd = optim.Adam(Gd.parameters(), lr=1e-3)
#
# epochs = 200
#
# loss_log = []
# acc_log = []
# epoch_log = []
#
# lmbda = torch.FloatTensor([0.05])
#
# for epoch in range(epochs):
#     acc_sum, loss_sum, cnt = (0, 0, 0)
#     for data in domain_dataloader:
#         X, y = data
#         sample_num = len(X)
#
#         optimizer_Gf.zero_grad()
#         optimizer_Gd.zero_grad()
#
#         feature = Gf(X)
#         reversed_feature = GradientReversalFn.apply(feature, lmbda)
#         pred = Gd(reversed_feature)
#
#         loss = domain_criteria(pred, y)
#         loss.backward()
#
#         optimizer_Gf.step()
#         optimizer_Gd.step()
#
#         loss_sum += loss.item()
#
#         with torch.no_grad():
#             feature = Gf(X)
#             pred = Gd(feature)
#             pred = torch.argmax(pred, dim=1)
#             acc = (pred == y).float().mean().item()
#             acc_sum += acc
#         cnt += 1
#
#     acc_log.append(acc_sum / cnt)
#     loss_log.append(loss_sum / cnt)
#     epoch_log.append(epoch)
#
# fig = make_subplots(rows=1, cols=2)
#
# fig.add_trace(
#     go.Scatter(x=epoch_log, y=loss_log, name='loss'),
#     row=1, col=1
# )
#
# fig.add_trace(
#     go.Scatter(x=epoch_log, y=acc_log, name='acc'),
#     row=1, col=2
# )
#
# fig.update_layout(height=450, width=900, title_text="Manual weights update, domain loss and acc")
# # fig.show()

# Gf = FeatureExtractor()
# Gy = LabelPredictor()
# Gd = DomainClassifier()
#
# label_criteria = nn.NLLLoss()
# domain_criteria = nn.CrossEntropyLoss()
#
# epochs = 1000
#
# mu = torch.FloatTensor([0.01])
# lmbda = torch.FloatTensor([0.001])
#
# epoch_log = []
# label_loss_log, label_acc_log = [], []
# domain_loss_log, domain_acc_log = [], []
# test_acc_log = []
#
# # Early stopping parameters
# best_test_acc = 0.0
# best_epoch = 0
# patience = 10
# patience_counter = 0
#
# for epoch in range(epochs):
#
#     label_loss_sum, label_acc_sum = 0, 0
#     domain_loss_sum, domain_acc_sum = 0, 0
#     test_acc_sum, cnt = 0, 0
#
#     for label_data, domain_data, test_data \
#             in zip(label_dataloader, domain_dataloader, test_dataloader):
#
#         # if early_stop:
#         #     break
#
#         label_X, label_y = label_data
#         domain_X, domain_y = domain_data
#         test_X, test_y = test_data
#
#         # train label of training data
#         Gf.zero_grad()
#         Gy.zero_grad()
#
#         feature = Gf(label_X)
#         pred = Gy(feature)
#         pred = F.softmax(pred, dim=1)
#
#         loss = label_criteria(pred, label_y)
#         loss.backward()
#
#         label_loss_sum += np.exp(loss.item())
#
#         # get grades and manually update Gy
#         with torch.no_grad():
#             # get grads of Gf
#             # Gf_label_grads = [param.grad for param in Gf.parameters()]
#
#             # Update Gf
#             for param in Gf.parameters():
#                 param -= mu * param.grad
#
#             # update Gy
#             for param in Gy.parameters():
#                 param -= mu * param.grad
#
#         with torch.no_grad():
#             sample_num = len(label_X)
#             feature = Gf(label_X)
#             pred = Gd(feature)
#             pred = F.softmax(pred, dim=1)
#
#             label_acc_sum += torch.sum(torch.argmax(pred, dim=1) == label_y) / sample_num
#
#         # train domain classifier
#         Gf.zero_grad()
#         Gd.zero_grad()
#
#         feature = Gf(domain_X)
#         reversed_feature = GradientReversalFn.apply(feature, lmbda)
#         domain_pred = Gd(reversed_feature)
#
#         domain_loss = domain_criteria(domain_pred, domain_y)
#         domain_loss.backward()
#         domain_loss_sum += domain_loss.item()
#
#         # get grades and manually update Gf, Gy
#         with torch.no_grad():
#             # update Gf
#             for param in Gf.parameters():
#                 param -= mu * param.grad
#
#             # update Gd
#             for param in Gd.parameters():
#                 param -= mu * param.grad
#
#         # domain accuracy
#         with torch.no_grad():
#             sample_num = len(domain_X)
#             feature = Gf(domain_X)
#             pred = Gd(feature)
#             domain_acc_sum += torch.sum(torch.argmax(pred, dim=1) == domain_y) / sample_num
#
#         with torch.no_grad():
#             sample_num = len(test_X)
#
#             feature = Gf(test_X)
#             pred = Gy(feature)
#             pred = F.softmax(pred, dim=1)
#
#             test_acc_sum += torch.sum(torch.argmax(pred, dim=1) == test_y) / sample_num
#
#         cnt += 1
#
#         # Check early stopping criteria
#         current_test_acc = test_acc_sum / cnt
#         test_acc_log.append(current_test_acc)
#
#         if current_test_acc > best_test_acc:
#             best_test_acc = current_test_acc
#             best_epoch = epoch  # Save the epoch at which we have the best test accuracy
#             patience_counter = 0
#             # Save the best model
#             torch.save(Gf.state_dict(), 'best_Gf.pth')
#             torch.save(Gy.state_dict(), 'best_Gy.pth')
#             print(f"Epoch {epoch}: New best test accuracy: {best_test_acc}. Model saved.")
#         else:
#             patience_counter += 1
#
#         if patience_counter >= patience:
#             # print(f"Stopping early at epoch {epoch} due to no improvement in test accuracy.")
#             break
#
#     epoch_log.append(epoch)
#
#     label_loss_log.append(label_loss_sum / cnt)
#     label_acc_log.append(label_acc_sum / cnt)
#
#     domain_loss_log.append(domain_loss_sum / cnt)
#     domain_acc_log.append(domain_acc_sum / cnt)
#
#     test_acc_log.append(test_acc_sum / cnt)
#
# # At the end of training, adjust the logs to only go up to the best epoch
# label_loss_log = label_loss_log[:best_epoch + 30]
# label_acc_log = label_acc_log[:best_epoch + 30]
# domain_loss_log = domain_loss_log[:best_epoch + 30]
# domain_acc_log = domain_acc_log[:best_epoch + 30]
# test_acc_log = test_acc_log[:best_epoch + 30]
# epoch_log = epoch_log[:best_epoch + 30]
#
# fig = make_subplots(rows=3, cols=2)
#
# fig.add_trace(
#     go.Scatter(x=epoch_log, y=label_loss_log),
#     row=1, col=1
# )
#
# fig.add_trace(
#     go.Scatter(x=epoch_log, y=label_acc_log),
#     row=1, col=2
# )
#
# fig.add_trace(
#     go.Scatter(x=epoch_log, y=domain_loss_log),
#     row=2, col=1
# )
#
# fig.add_trace(
#     go.Scatter(x=epoch_log, y=domain_acc_log),
#     row=2, col=2
# )
#
# fig.add_trace(
#     go.Scatter(x=epoch_log, y=test_acc_log),
#     row=3, col=1
# )
#
# fig.update_layout(height=1350, width=900, title_text="Manual weights update, domain loss and acc")
# # fig.show()
#
# print(best_epoch)
#
# # Load the best model
# Gf.load_state_dict(torch.load('best_Gf.pth'))
# Gy.load_state_dict(torch.load('best_Gy.pth'))
#
# df['test_result'] = df['labels']
#
# with torch.no_grad():
#     feature = Gf(test_dataset.X)
#     pred = Gy(feature)
#     pred = F.softmax(pred, dim=1)
#     df.loc[1000:, 'test_result'] = torch.argmax(pred, dim=1).numpy() + 2
#
# fig = px.scatter(df, x='x', y='y', color='test_result')
# # fig.show()

from pytorch_revgrad import RevGrad


# Define model
class DomainClassifier(nn.Module):
    def __init__(self):
        super(DomainClassifier, self).__init__()

        self.linear_relu_stack = nn.Sequential(
            RevGrad(),
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
Gd = DomainClassifier()

mu = 1e-3
lmda = 1e-4

label_criteria = nn.NLLLoss()
label_optimizer = optim.Adam(list(Gf.parameters()) + list(Gd.parameters()), lr=mu)

domain_criteria = nn.CrossEntropyLoss()
domain_optimizer = optim.Adam(list(Gf.parameters()) + list(Gd.parameters()), lr=mu * lmda)

epochs = 1000

mu = torch.FloatTensor([0.01])
lmbda = torch.FloatTensor([0.5])

epoch_log = []
label_loss_log, label_acc_log = [], []
domain_loss_log, domain_acc_log = [], []
test_acc_log = []

for epoch in range(epochs):

    label_loss_sum, label_acc_sum = 0, 0
    domain_loss_sum, domain_acc_sum = 0, 0
    test_acc_sum, cnt = 0, 0

    for label_data, domain_data, test_data in zip(label_dataloader, domain_dataloader, test_dataloader):
        label_X, label_y = label_data
        domain_X, domain_y = domain_data
        test_X, test_y = test_data

        # train label of training data
        Gf.zero_grad()
        Gy.zero_grad()

        feature = Gf(label_X)
        pred = Gy(feature)
        pred = F.softmax(pred, dim=1)

        loss = label_criteria(pred, label_y)
        label_optimizer.zero_grad()
        loss.backward()
        label_optimizer.step()

        label_loss_sum += np.exp(loss.item())

        with torch.no_grad():
            sample_num = len(label_X)

            feature = Gf(label_X)
            pred = Gd(feature)
            pred = F.softmax(pred, dim=1)

            label_acc_sum += torch.sum(torch.argmax(pred, dim=1) == label_y) / sample_num

        # train domain classifier
        Gf.zero_grad()
        Gd.zero_grad()

        feature = Gf(domain_X)
        pred = Gd(feature)

        loss = domain_criteria(pred, domain_y)
        domain_optimizer.zero_grad()
        loss.backward()
        domain_optimizer.step()
        domain_loss_sum += loss.item()

        # domain accuracy
        with torch.no_grad():
            sample_num = len(domain_X)
            feature = Gf(domain_X)
            pred = Gd(feature)
            domain_acc_sum += torch.sum(torch.argmax(pred, dim=1) == domain_y) / sample_num

        with torch.no_grad():
            sample_num = len(test_X)

            feature = Gf(test_X)
            pred = Gy(feature)
            pred = F.softmax(pred, dim=1)

            test_acc_sum += torch.sum(torch.argmax(pred, dim=1) == test_y) / sample_num

        cnt += 1

    epoch_log.append(epoch)

    label_loss_log.append(label_loss_sum / cnt)
    label_acc_log.append(label_acc_sum / cnt)

    domain_loss_log.append(domain_loss_sum / cnt)
    domain_acc_log.append(domain_acc_sum / cnt)

    test_acc_log.append(test_acc_sum / cnt)

fig = make_subplots(rows=3, cols=2)

fig.add_trace(
    go.Scatter(x=epoch_log, y=label_loss_log, name='label loss'),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(x=epoch_log, y=label_acc_log, name='label acc'),
    row=1, col=2
)

fig.add_trace(
    go.Scatter(x=epoch_log, y=domain_loss_log, name='domain loss'),
    row=2, col=1
)

fig.add_trace(
    go.Scatter(x=epoch_log, y=domain_acc_log, name='domain acc'),
    row=2, col=2
)

fig.add_trace(
    go.Scatter(x=epoch_log, y=test_acc_log, name='test acc'),
    row=3, col=1
)

fig.update_layout(height=1350, width=900, title_text="Manual weights update, domain loss and acc")
fig.show()

df['test_result'] = df['labels']

with torch.no_grad():
    feature = Gf(test_dataset.X)
    pred = Gy(feature)
    pred = F.softmax(pred, dim=1)
    df.loc[1000:, 'test_result'] = torch.argmax(pred, dim=1).numpy() + 2

fig = px.scatter(df, x='x', y='y', color='test_result')
fig.show()