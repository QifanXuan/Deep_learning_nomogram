import glob
from collections import Counter

import PIL
import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.decomposition import PCA
from torch.utils.data import dataloader

from Utils.data_transform import lung_transform
from Utils.get_dataset import getdataset

mode_select = 9
train_dataset = getdataset("../dataset/GA_third_data.csv", "../dataset/roi/train", "../dataset/fat_intrathoracic",
                           lung_transform['train'], mode_select=mode_select, is_augment=False,
                           use_fat_index=False)
test1_dataset = getdataset("../dataset/GA_first_data.csv", "../dataset/roi/test", "../dataset/fat_intrathoracic",
                           lung_transform['train'], mode_select=mode_select, is_augment=False,
                           use_fat_index=False)
test2_dataset = getdataset("../dataset/GA_second_data.csv", "../dataset/roi/test2", "../dataset/fat_intrathoracic",
                           lung_transform['train'], mode_select=mode_select, is_augment=False,
                           use_fat_index=False)
print(len(train_dataset), len(test1_dataset), len(test2_dataset))
# 计算每个类别的样本数量
train_labels = [label for (_, label, _) in train_dataset]
label_count = dict(Counter(train_labels))
print(f'训练集类别比例是:{label_count}')
sexs = [sex for (_, _, sex) in train_dataset]
sex_count = dict(Counter(sexs))
print(f'训练集男女比例是:{sex_count}, 男0, 女1')

# 计算每个类别的样本数量
train_labels = [label for (_, label, _) in test1_dataset]
label_count = dict(Counter(train_labels))
print(f'一院类别比例是:{label_count}')
sexs = [sex for (_, _, sex) in test1_dataset]
sex_count = dict(Counter(sexs))
print(f'一院男女比例是:{sex_count}, 男2, 女1')

# 计算每个类别的样本数量
train_labels = [label for (_, label, _) in test2_dataset]
label_count = dict(Counter(train_labels))
print(f'二院类别比例是:{label_count}')
sexs = [sex for (_, _, sex) in test2_dataset]
sex_count = dict(Counter(sexs))
print(f'二院男女比例是:{sex_count}, 男0, 女1')

train_dataloader = dataloader.DataLoader(dataset=train_dataset, batch_size=1,
                                         shuffle=False, drop_last=False)
test1_dataloader = dataloader.DataLoader(dataset=test1_dataset, batch_size=1,
                                       shuffle=False, drop_last=False)
test2_dataloader = dataloader.DataLoader(dataset=test2_dataset, batch_size=1,
                                       shuffle=False, drop_last=False)

train = []
test1 = []
test2 = []
# 获取标签
train_labels = []
test1_labels = []
test2_labels = []
for data, label, sex in train_dataloader:
    data = data[0]
    train.append(data.numpy().reshape(-1))
    train_labels.append(label.item())
train = np.array(train)

for data, label, sex in test1_dataloader:
    data = data[0]
    test1.append(data.numpy().reshape(-1))
    test1_labels.append(label.item())

test1 = np.array(test1)

for data, label, sex in test2_dataloader:
    data = data[0]
    test2.append(data.numpy().reshape(-1))
    test2_labels.append(label.item())

test2 = np.array(test2)

train_labels = np.array(train_labels)
test1_labels = np.array(test1_labels)
test2_labels = np.array(test2_labels)
# 使用PCA将数据降到3维
pca = PCA(n_components=2)
train_3d = pca.fit_transform(train)
test1_3d = pca.fit_transform(test1)
test2_3d = pca.fit_transform(test2)
data_3d = np.concatenate((train_3d, test1_3d, test2_3d), axis=0)
labels = np.concatenate((train_labels, test1_labels, test2_labels), axis=0)
clf = svm.SVC()
clf.fit(train_3d, train_labels)

# # 绘制三维散点图
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(train_3d[:, 0], train_3d[:, 1], c=train_labels, marker='^', cmap='jet')
# ax.scatter(test1_3d[:, 0], test1_3d[:, 1], c=test1_labels, marker='o', cmap='cool')
# ax.scatter(test2_3d[:, 0], test2_3d[:, 1], c=test2_labels, marker='s', cmap='cool')
#
# ax.set_xlabel('Principal Component 1')
# ax.set_ylabel('Principal Component 2')
# ax.set_title('Scatter plot of PCA')
#
# # 绘制支持向量
# support_vectors = clf.support_vectors_
# ax.scatter(support_vectors[:, 0], support_vectors[:, 1], c='black', marker='x')
#
# # 绘制分割曲线
# xmin, xmax = train_3d[:, 0].min(), train_3d[:, 0].max()
# ymin, ymax = train_3d[:, 1].min(), train_3d[:, 1].max()
# xx, yy = np.meshgrid(np.linspace(xmin, xmax, 100), np.linspace(ymin, ymax, 100))
# Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
# Z = Z.reshape(xx.shape)
# ax.contour(xx, yy, Z, levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'], colors='k')
#
# plt.show()
numbers = list(range(-50, -101, -1))
result = []
# for i in range(len(train_3d)):
#     data = train_3d[i]
#     x = data[0]
#     if (x <= -50) and (x >= -100):
#         result.append(i)
#
# print(result)
# print(len(result))