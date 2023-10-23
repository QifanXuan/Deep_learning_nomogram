import numpy as np
import os
import sys

from sklearn.metrics import roc_curve, auc

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import torch
from matplotlib import pyplot as plt
from torch.utils.data import dataloader
from tqdm import tqdm
from .evaluation_index import SegmentationMetric, ClassificationMetric

def train_cla_epoch(model, loss_fn, optimizer, data_loader, device):
    metric = ClassificationMetric(2)
    model.train()
    running_loss = 0
    y_argmax = []
    y_true = []
    y_score = []
    for data in tqdm(data_loader):
        imgs, labels = data
        imgs = imgs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        y_pred = model(imgs)
        loss = loss_fn(y_pred, labels)
        y_true.append(labels.cpu().numpy())
        y_score.append(y_pred[:, 1].cpu().detach().numpy())
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            running_loss += loss.item()
            y_pred = torch.argmax(y_pred, dim=1)
            y_argmax.append(y_pred.cpu().detach().numpy())
    y_true = np.concatenate(y_true)
    y_score = np.concatenate(y_score)
    y_argmax = np.concatenate(y_argmax)
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    metric.addBatch(y_argmax, y_true)
    epoch_acc = metric.accuracy()
    epoch_sens = metric.sensitivity()
    epoch_spe = metric.specificity()
    epoch_ppv = metric.ppv()
    epoch_npv = metric.npv()
    epoch_loss = running_loss / len(data_loader.dataset)
    return epoch_acc, epoch_loss, epoch_sens, epoch_spe, epoch_ppv, epoch_npv, roc_auc
def val_cla_epoch(model, loss_fn, optimizer, data_loader, device):
    metric = ClassificationMetric(2)
    model.eval()
    running_loss = 0
    y_true = []
    y_score = []
    y_argmax = []
    for data in tqdm(data_loader):
        imgs, labels = data
        imgs = imgs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        y_pred = model(imgs)
        loss = loss_fn(y_pred, labels)
        y_true.append(labels.cpu().numpy())
        y_score.append(y_pred[:, 1].cpu().detach().numpy())
        with torch.no_grad():
            running_loss += loss.item()
            y_pred = torch.argmax(y_pred, dim=1)
            y_argmax.append(y_pred.cpu().detach().numpy())
    y_true = np.concatenate(y_true)
    y_score = np.concatenate(y_score)
    y_argmax = np.concatenate(y_argmax)
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    metric.addBatch(y_argmax, y_true)
    epoch_acc = metric.accuracy()
    epoch_sens = metric.sensitivity()
    epoch_spe = metric.specificity()
    epoch_ppv = metric.ppv()
    epoch_npv = metric.npv()
    epoch_loss = running_loss / len(data_loader.dataset)
    return epoch_acc, epoch_loss, epoch_sens, epoch_spe, epoch_ppv, epoch_npv, roc_auc


def img_pre_visualization(data_loader, is_seg=False):
    if is_seg:
        imgs, labels = next(iter(data_loader))
        plt.figure(figsize=(16, 8))
        for i in range(len(imgs[:4])):
            img = imgs[:4][i]
            lable = labels[:4][i]
            print(img.shape, lable.shape)
            img = img.numpy()
            img = np.transpose(img, (1, 2, 0))
            plt.subplot(2, 4, i + 1)
            plt.imshow(img, cmap="gray")
            plt.subplot(2, 4, i + 5)
            plt.imshow(lable)
        plt.show()
    else:
        imgs, labels = next(iter(data_loader))
        plt.figure(figsize=(16, 8))
        for i in range(len(imgs[:8])):
            img = imgs[:8][i]
            lable = labels[:8][i]
            img = img.numpy()
            img = np.transpose(img, (1, 2, 0))
            plt.subplot(2, 4, i + 1)
            plt.imshow(img)
            plt.title(lable)
        plt.show()


def train_seg_epoch(model, loss_fn, optimizer, data_loader, device, img_size):
    metric = SegmentationMetric(2)
    model.train()
    epoch_right_num = 0
    epoch_total_num = 0
    epoch_loss = 0
    epoch_iou = []
    epoch_pa = []
    epoch_mpa = []
    for data in tqdm(data_loader):
        imgs, labels = data
        # imgs = imgs.half()
        imgs = imgs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        y_pred = model(imgs)
        # y_pred = y_pred.to(torch.float32)
        loss = loss_fn(y_pred, labels)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            y_pred = torch.argmax(y_pred, dim=1)
            epoch_right_num += (y_pred == labels).sum().item()
            epoch_total_num += labels.size(0)
            epoch_loss += loss.item()
            y_pred, labels = y_pred.to('cpu'), labels.to('cpu')
            y_pred, labels = np.array(y_pred), np.array(labels)

            for i in range(y_pred.shape[0]):
                metric.addBatch(y_pred[i], labels[i])
                pa = metric.pixelAccuracy()
                mpa = metric.meanPixelAccuracy()
                mIoU = metric.meanIntersectionOverUnion()
                epoch_iou.append(mIoU)
                epoch_pa.append(pa)
                epoch_mpa.append(mpa)
    epoch_loss = epoch_loss / len(data_loader.dataset)
    epoch_acc = epoch_right_num / (epoch_total_num * img_size * img_size)
    epoch_iou = torch.tensor(epoch_iou)
    mean_iou = torch.mean(epoch_iou)
    epoch_pa = torch.tensor(epoch_pa)
    mean_pa = torch.mean(epoch_pa)
    epoch_mpa = torch.tensor(epoch_mpa)
    mean_mpa = torch.mean(epoch_mpa)
    return epoch_acc, epoch_loss, mean_iou, mean_pa, mean_mpa

def val_seg_epoch(model, loss_fn, optimizer, data_loader, device, img_size):
    metric = SegmentationMetric(2)
    model.eval()
    epoch_right_num = 0
    epoch_total_num = 0
    epoch_loss = 0
    epoch_iou = []
    epoch_pa = []
    epoch_mpa = []
    for data in tqdm(data_loader):
        imgs, labels = data
        # imgs = imgs.half()
        imgs = imgs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        y_pred = model(imgs)
        # y_pred = y_pred.to(torch.float32)
        loss = loss_fn(y_pred, labels)
        with torch.no_grad():
            y_pred = torch.argmax(y_pred, dim=1)
            epoch_right_num += (y_pred == labels).sum().item()
            epoch_total_num += labels.size(0)
            epoch_loss += loss.item()
            y_pred, labels = y_pred.to('cpu'), labels.to('cpu')
            y_pred, labels = np.array(y_pred), np.array(labels)
            for i in range(y_pred.shape[0]):
                metric.addBatch(y_pred[i], labels[i])
                pa = metric.pixelAccuracy()
                mpa = metric.meanPixelAccuracy()
                mIoU = metric.meanIntersectionOverUnion()
                epoch_iou.append(mIoU)
                epoch_pa.append(pa)
                epoch_mpa.append(mpa)

    epoch_loss = epoch_loss / len(data_loader.dataset)
    epoch_acc = epoch_right_num / (epoch_total_num * img_size * img_size)
    epoch_iou = torch.tensor(epoch_iou)
    mean_iou = torch.mean(epoch_iou)
    epoch_pa = torch.tensor(epoch_pa)
    mean_pa = torch.mean(epoch_pa)
    epoch_mpa = torch.tensor(epoch_mpa)
    mean_mpa = torch.mean(epoch_mpa)
    return epoch_acc, epoch_loss, mean_iou, mean_pa, mean_mpa