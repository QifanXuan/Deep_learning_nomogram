import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from tqdm import tqdm

from Utils.data_transform import lung_transform
from Utils.get_dataset import getdataset

from runner.Runner_utils.evaluation_index import ClassificationMetric


def cal_auc_sens_spe_with95(y_true, y_score):
    # 设置bootstrap抽样次数
    n_bootstrap = 1000  # 您可以根据需要选择抽样次数

    # 初始化空数组以存储每次抽样的AUC、sens和spe
    bootstrap_aucs = []
    bootstrap_sens = []
    bootstrap_spe = []
    bootstrap_acc = []
    for _ in range(n_bootstrap):
        # 随机抽样数据集（可以根据需要调整抽样大小）
        sampled_indices = resample(range(len(y_true)))
        y_true_sampled = y_true[sampled_indices]
        y_score_sampled = y_score[sampled_indices]
        metric = ClassificationMetric(2)  # 2表示有2个分类，有几个分类就填几
        # 计算AUC
        fpr, tpr, thresholds = roc_curve(y_true_sampled, y_score_sampled)
        auc_value = auc(fpr, tpr)
        bootstrap_aucs.append(auc_value)
        # 计算敏感性和特异性
        threshold = thresholds[np.argmax(tpr - fpr)]
        y_pred_sampled = (y_score_sampled > threshold).astype(int)
        hist = metric.addBatch(y_pred_sampled, y_true_sampled)
        sens_value = metric.sensitivity()
        spe_value = metric.specificity()
        bootstrap_sens.append(sens_value)
        bootstrap_spe.append(spe_value)
        # 计算ACC
        acc_value = metric.accuracy()
        bootstrap_acc.append(acc_value)

    # 计算AUC、sens和spe的置信区间
    confidence_level = 0.95
    alpha = (1 - confidence_level) / 2
    lower_percentile = alpha * 100
    upper_percentile = (1 - alpha) * 100

    conf_interval_auc = np.percentile(bootstrap_aucs, [lower_percentile, upper_percentile])
    conf_interval_sens = np.percentile(bootstrap_sens, [lower_percentile, upper_percentile])
    conf_interval_spe = np.percentile(bootstrap_spe, [lower_percentile, upper_percentile])
    conf_interval_acc = np.percentile(bootstrap_acc, [lower_percentile, upper_percentile])
    mean_auc = np.mean(bootstrap_aucs)
    mean_sens = np.mean(bootstrap_sens)
    mean_spe = np.mean(bootstrap_spe)
    mean_acc = np.mean(bootstrap_acc)
    print("95% AUC 置信区间: {:.3f}[{:.3f},{:.3f}]".format(mean_auc, conf_interval_auc[0], conf_interval_auc[1]))
    print("95% ACC 置信区间: {:.3f}[{:.3f},{:.3f}]".format(mean_acc, conf_interval_acc[0], conf_interval_acc[1]))
    print("95% sens 置信区间: {:.3f}[{:.3f},{:.3f}]".format(mean_sens, conf_interval_sens[0], conf_interval_sens[1]))
    print("95% spe 置信区间: {:.3f}[{:.3f},{:.3f}]".format(mean_spe, conf_interval_spe[0], conf_interval_spe[1]))

if __name__ == '__main__':
    # mode_select = int(input("please select data fuse mode(range is 0-5)："))
    # if torch.cuda.is_available():
    #     device = 'cuda:0'
    # else:
    #     device = 'cpu'
    # # 加载模型参数
    # model = swin_small_patch4_window7_224(num_classes=2)
    # # model = resnet101()
    # # model = resnet50()
    # # model = torchvision.models.vgg16()
    # # num_features = model.classifier[-1].in_features
    # # model.classifier[-1] = nn.Linear(num_features, 2)
    # # model = torchvision.models.densenet121()
    # # model.classifier = nn.Linear(in_features=1024, out_features=2, bias=True)
    # model.load_state_dict(torch.load('../../runner/Running_Dict/SwinTransformer_fat 第1折 0.8709 0.6821.pth'))
    # model.eval()
    # model.cuda()
    # data_transform = lung_transform
    # # # 加载测试数据集
    # total_dataset = getdataset("../../dataset/lung.csv", "../../dataset/roi/img_test", "../../dataset/fat_intrathoracic",
    #                            lung_transform['train'], mode_select=mode_select)
    # total_loader = torch.utils.data.DataLoader(total_dataset, batch_size=32)
    # imgs, labels = next(iter(total_loader))
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
    #
    # # 输出测试结果
    # y_true = []
    # y_score = []
    # with torch.no_grad():
    #     for data, labels in tqdm(total_loader):
    #         imgs = data
    #         imgs = imgs.to('cuda:0')
    #         # 预测概率
    #         y_pred = model(imgs).softmax(dim=1)[:, 1]
    #         y_true.append(labels.numpy())
    #         y_score.append(y_pred.cpu().numpy())
    # # 把每个 batch 的结果合并成一个数组
    data = pd.read_csv('../data_cleaning/first.csv')
    y_true = np.array(data.iloc[:, 1])
    y_score = np.array(data.iloc[:, 4])
    # cal_auc_sens_spe_with95(y_true, y_score)
    train_pred, val_pred, train_label, val_label = train_test_split(y_score, y_true, test_size=0.2,
                                                                    random_state=120)
    cal_auc_sens_spe_with95(train_label, train_pred)
    cal_auc_sens_spe_with95(val_label, val_pred)
