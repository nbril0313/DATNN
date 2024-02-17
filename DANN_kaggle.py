from sklearn import datasets
import plotly.express as px
import numpy as np
import pandas as pd

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

x = np.concatenate((x_train, x_test)) # train과 test x 좌표 모음
y = np.concatenate((y_train, y_test)) # train과 test y 좌표 모음

# real labels
labels = np.concatenate((labels_train, labels_test))

# train, test 구분 열
domain = np.concatenate((domain_train, domain_test))

# labels for train, plot 할 때 데이터를 구분하기 위한 열
plabels = np.concatenate((labels_train, plabel_test))

# dictionary 생성
d = {'x': x, 'y': y, 'labels': labels, 'domain': domain, 'plabels': plabels}
print(d)
df = pd.DataFrame(data=d)

df.loc[(df['domain'] == 0) & (df['labels'] == 0), 'x'] *= 0.8
df.loc[(df['domain'] == 0) & (df['labels'] == 1), 'x'] *= 0.8

df.loc[(df['domain'] == 1) & (df['labels'] == 0), 'y'] -= 0.2
df.loc[(df['domain'] == 1) & (df['labels'] == 1), 'y'] -= 0.2

