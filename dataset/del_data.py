import os
import random
from collections import Counter

import torch

from Utils.data_transform import lung_transform
from Utils.get_dataset import getdataset

del_num = [0, 1, 2, 3, 5, 6, 8, 9, 10, 14, 21, 22, 26, 33, 45, 46, 52, 58, 68, 75, 77, 85, 89
    , 90, 94, 96, 97, 98, 105, 107, 108, 110, 111, 113, 123, 124, 129, 131, 134, 137, 143, 144,
           150, 155, 158, 165, 170, 177, 178, 181, 186, 187, 188, 189, 190, 192, 193, 195, 197, 204,
           205, 215, 216, 220, 224, 231, 245, 247, 248, 249, 252, 257, 259, 260, 263, 265, 267, 269, 276,
           277, 282, 283, 285, 287, 289, 292, 296, 297, 300, 307, 308, 312, 313, 318, 322, 325, 326, 327, 328,
           329, 331, 332, 337, 342, 344, 346, 351, 352, 355, 358, 360, 361, 364, 367, 368, 372, 375, 376, 377, 385,
           388, 390, 395, 396, 402, 407, 409, 414, 417, 418, 419, 420, 421, 422, 426, 428, 429, 433, 434, 435, 437,
           438, 439, 443, 445, 447, 449, 450, 451, 452, 453, 460, 461, 462, 464, 466, 467, 468, 471, 474, 475, 479,
           480, 486, 487, 489, 490, 493, 495, 497, 498, 500, 501, 503, 507, 511, 517, 519, 524, 528, 534, 536, 544,
           545, 547, 550, 553, 555, 559, 560, 563, 564, 565, 566, 569, 571, 573, 574, 576, 581, 583, 584, 585, 586,
           587, 589, 591, 592, 595, 599, 601, 602, 604, 605, 606, 609, 610, 611, 612, 614, 615, 617, 618, 620, 623,
           627, 629, 631, 632, 633, 634, 637, 641, 643, 645, 646, 647, 648, 653, 654, 656, 658, 661, 663, 666, 667,
           668, 671, 675, 680, 681, 682, 684, 685, 686, 687, 690, 692, 693, 694, 698, 702, 708, 709, 710, 711, 717,
           719, 720, 722, 724, 726, 727, 728, 732, 734, 736, 737, 741, 743, 744, 748, 750, 757, 760, 761, 762, 764,
           766, 770, 771, 778, 779, 783, 784, 786, 787, 790, 793, 796, 797, 798, 800, 801, 803, 805, 807, 810, 813,
           814, 816, 818]

mode_select = 4
train_dataset = getdataset("../dataset/GA_third_data.csv", "../dataset/roi/img_train", "../dataset/fat_intrathoracic",
                           lung_transform['train'], mode_select=mode_select, is_augment=False,
                           use_fat_index=False)
# # 获得数据集的长度
# dataset_len = len(train_dataset)
#
# # 创建一个索引列表
# index_list = list(range(dataset_len))
#
# # 随机化索引列表
# random.shuffle(index_list)
# # 创建一个新的随机化的数据集
# shuffled_dataset = torch.utils.data.Subset(train_dataset, index_list)

# 计算每个类别的样本数量
train_labels = [label for (_, label, _) in train_dataset]
label_count = dict(Counter(train_labels))
print(f'训练集类别比例是:{label_count}')
sexs = [sex for (_, _, sex) in train_dataset]
sex_count = dict(Counter(sexs))
print(f'训练集男女比例是:{sex_count}, 男0, 女1')
desired_class_ratio = {1: 302, 0: 70}
desired_gender_ratio = {0: 124, 1: 248}
# need_del = {0: 181 - 124, 1: 0}
# for j, (data, label, sex) in enumerate(train_dataset):
#     if label == 1:
#         if random.random() < 0.5:
#             if (sex == 0) and (need_del[0] != 0):
#                 path = train_dataset.img_path_list[j]
#                 print(path)
#                 os.remove(path)
#                 need_del[0] = need_del[0] - 1
#             elif (sex == 1) and (need_del[1] != 0):
#                 path = train_dataset.img_path_list[j]
#                 print(path)
#                 os.remove(path)
#                 need_del[1] = need_del[1] - 1
#     else:
#         pass

# count = 0
# for i in range(len(train_dataset)):
#        img, label, sex = train_dataset[i]
#        path = train_dataset.img_path_list
#        if i in del_num and sex == 0 and label==1:
#               os.remove(path[i])
#               count +=1
# print(count)
