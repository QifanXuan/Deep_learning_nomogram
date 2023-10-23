import math
import scipy
import torch
import numpy as np
import torchvision
from scipy.stats import sem, t, stats
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
from torch import nn
from tqdm import tqdm
from Utils.get_dataset import getdataset
# from models.swin_transfomer import swin_small_patch4_window7_224
from Utils.data_transform import lung_transform
import os
import sys
from models.cla_models import swin_small_patch4_window7_224, resnet101, resnet50

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
if __name__ == '__main__':
    # mode_select = int(input("please select data fuse mode(range is 0-5)："))
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    # 加载模型参数
    mode_select = 5
    model = swin_small_patch4_window7_224(num_classes=2)
    # model = resnet101()
    # model = resnet50()
    # model = torchvision.models.vgg16()
    # num_features = model.classifier[-1].in_features
    # model.classifier[-1] = nn.Linear(num_features, 2)
    # model = torchvision.models.densenet121()
    # model.classifier = nn.Linear(in_features=1024, out_features=2, bias=True)
    '''
    '''
    '''
    '''

    model.load_state_dict(torch.load('../runner/Model_Dict/SwinTransformerfat 0.841 0.805_nn.pth'))
    model.eval()
    model.cuda()
    data_transform = lung_transform
    # # 加载测试数据集
    total_dataset = getdataset("../dataset/lung.csv", "../dataset/roi/img_test2", "../dataset/fat_intrathoracic",
                               lung_transform['original'], mode_select=mode_select, is_augment=False)
    total_loader = torch.utils.data.DataLoader(total_dataset, batch_size=32)
    imgs, labels = next(iter(total_loader))
    # plt.figure(figsize=(16, 8))
    # for i in range(len(imgs[:8])):
    #     img = imgs[:8][i]
    #     lable = labels[:8][i]
    #     img = img.numpy()
    #     img = np.transpose(img, (1, 2, 0))
    #     plt.subplot(2, 4, i + 1)
    #     plt.imshow(img)
    #     plt.title(lable)
    # plt.show()
    # print(len(total_loader.dataset))

    # 输出测试结果
    y_true = []
    y_score = []
    with torch.no_grad():
        for data, labels in tqdm(total_loader):
            imgs = data
            imgs = imgs.to('cuda:0')
            # 预测概率
            y_pred = model(imgs).softmax(dim=1)[:, 1]
            y_true.append(labels.numpy())
            y_score.append(y_pred.cpu().numpy())
    # 把每个 batch 的结果合并成一个数组
    y_true = np.concatenate(y_true)
    y_score = np.concatenate(y_score)

    # 计算 ROC 曲线和 AUC
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    print("ROC is {:.3f}".format(roc_auc))
    # 找到最佳阈值
    best_threshold = thresholds[np.argmax(tpr - fpr)]
    print(best_threshold)
    # # 画 ROC 曲线图
    # plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc)
    # plt.plot([0, 1], [0, 1], 'k--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver Operating Characteristic')
    # plt.legend(loc="lower right")
    # plt.show()

    # 计算spe，sens 阈值越小spe越高，阈值越大sens越高
    for j in range(len(y_score)):
        if y_score[j] > 0.5:
            y_score[j] = 1
        else:
            y_score[j] = 0
    y_pred_argmax = y_score
    tp, tn, fp, fn = 0, 0, 0, 0
    for i in range(len(y_pred_argmax)):
        if y_pred_argmax[i] == y_true[i]:
            if y_true[i] == 1:
                tp += 1
            else:
                tn += 1
        else:
            if y_true[i] == 0:
                fp += 1
            else:
                fn += 1
    epoch_sens = tp / (tp + fn + 0.00001)
    epoch_spe = tn / (tn + fp + 0.00001)
    epoch_acc = (tp+tn)/(tp+tn+fp+fn)
    print("sens={:.3f}, spe={:.3f}".format(epoch_sens, epoch_spe))
    print("acc={:.3f}".format(epoch_acc))