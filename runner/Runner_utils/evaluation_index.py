
"""
refer to https://github.com/jfzhang95/pytorch-deeplab-xception/blob/master/utils/metrics.py
"""
import numpy as np
import cv2

__all__ = ['SegmentationMetric']

from sklearn.metrics import confusion_matrix, roc_auc_score

"""
confusionMetric  # 注意：此处横着代表预测值，竖着代表真实值，与之前介绍的相反
P\L     P    N
P      TP    FP
N      FN    TN
"""

# 语义分割模型评价
class SegmentationMetric(object):
    def __init__(self, numClass):
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass,) * 2)  # 混淆矩阵（空）

    def pixelAccuracy(self):
        # return all class overall pixel accuracy 正确的像素占总像素的比例
        #  PA = acc = (TP + TN) / (TP + TN + FP + TN)
        acc = np.diag(self.confusionMatrix).sum() / self.confusionMatrix.sum()
        return acc

    def classPixelAccuracy(self):
        # return each category pixel accuracy(A more accurate way to call it precision)
        # acc = (TP) / TP + FP
        classAcc = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=1)
        return classAcc  # 返回的是一个列表值，如：[0.90, 0.80, 0.96]，表示类别1 2 3各类别的预测准确率

    def meanPixelAccuracy(self):
        """
        Mean Pixel Accuracy(MPA，均像素精度)：是PA的一种简单提升，计算每个类内被正确分类像素数的比例，之后求所有类的平均。
        :return:
        """
        classAcc = self.classPixelAccuracy()
        meanAcc = np.nanmean(classAcc)  # np.nanmean 求平均值，nan表示遇到Nan类型，其值取为0
        return meanAcc  # 返回单个值，如：np.nanmean([0.90, 0.80, 0.96, nan, nan]) = (0.90 + 0.80 + 0.96） / 3 =  0.89

    def IntersectionOverUnion(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = np.diag(self.confusionMatrix)  # 取对角元素的值，返回列表
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(
            self.confusionMatrix)  # axis = 1表示混淆矩阵行的值，返回列表； axis = 0表示取混淆矩阵列的值，返回列表
        IoU = intersection / union  # 返回列表，其值为各个类别的IoU
        return IoU

    def meanIntersectionOverUnion(self):
        mIoU = np.nanmean(self.IntersectionOverUnion())  # 求各类别IoU的平均
        return mIoU

    def genConfusionMatrix(self, imgPredict, imgLabel):  #
        """
        同FCN中score.py的fast_hist()函数,计算混淆矩阵
        :param imgPredict:
        :param imgLabel:
        :return: 混淆矩阵
        """
        # remove classes from unlabeled pixels in gt image and predict
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        count = np.bincount(label, minlength=self.numClass ** 2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        # print(confusionMatrix)
        return confusionMatrix

    def Frequency_Weighted_Intersection_over_Union(self):
        """
        FWIoU，频权交并比:为MIoU的一种提升，这种方法根据每个类出现的频率为其设置权重。
        FWIOU =     [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
        """
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def addBatch(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)  # 得到混淆矩阵
        return self.confusionMatrix

    def reset(self):
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))

# 分类任务模型评价
class ClassificationMetric(object):
    def __init__(self, numClass):
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass,) * 2)  # 混淆矩阵（空）
    def genConfusionMatrix(self, imgPredict, imgLabel):  #
        confusionMatrix = confusion_matrix(imgLabel, imgPredict)
        return confusionMatrix
    def accuracy(self):
        # 预测正确数/总数
        #  PA = acc = (TP + TN) / (TP + TN + FP + TN)
        acc = np.diag(self.confusionMatrix).sum() / self.confusionMatrix.sum()
        return acc
    def sensitivity(self):
        # 敏感性，癌症患者中有多少检测出来了
        # tp / (tp + fn)
        sens = np.diag(self.confusionMatrix)[1]/(np.sum(self.confusionMatrix[1, :])+0.0000001)
        return sens
    def specificity(self):
        # 特异性，非癌症患者中有多少检测出来了
        # tn / (tn + fp)
        spec = np.diag(self.confusionMatrix)[0]/(np.sum(self.confusionMatrix[0, :])+0.0000001)
        return spec
    def ppv(self):
        # PPV = TP / (TP + FP)，模型的预测正例中，有多少是预测对了(评判误诊的能力)
        ppv = np.diag(self.confusionMatrix)[1]/(np.sum(self.confusionMatrix[:, 1])+0.0000001)
        return ppv
    def npv(self):
        # NPV = TN / (TN + FN)，模型的预测负例中，有多少是预测对了(评判漏检的能力)
        npv = np.diag(self.confusionMatrix)[0]/(np.sum(self.confusionMatrix[:, 0])+0.0000001)
        return npv
    def addBatch(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)  # 得到混淆矩阵
        return self.confusionMatrix
# 测试内容
if __name__ == '__main__':
    imgPredict = np.array([0, 0, 1, 0, 0, 0])  # 可直接换成预测图片
    imgLabel = np.array([0, 0, 1, 1, 1, 1])  # 可直接换成标注图片
    metric = ClassificationMetric(2)  # 2表示有2个分类，有几个分类就填几
    hist = metric.addBatch(imgPredict, imgLabel)
    acc = metric.accuracy()
    sens = metric.sensitivity()
    spec = metric.specificity()
    ppv = metric.ppv()
    npv = metric.npv()
    # print(acc)
    print(sens)
    print(spec)
    print(ppv)
    print(npv)
    # hist = metric.addBatch(imgPredict, imgLabel)
    # pa = metric.pixelAccuracy()
    # cpa = metric.classPixelAccuracy()
    # mpa = metric.meanPixelAccuracy()
    # IoU = metric.IntersectionOverUnion()
    # mIoU = metric.meanIntersectionOverUnion()
    # print('hist is :\n', hist)
    # print('PA is : %f' % pa)
    # print('cPA is :', cpa)  # 列表
    # print('mPA is : %f' % mpa)
    # print('IoU is : ', IoU)
    # print('mIoU is : ', mIoU)

