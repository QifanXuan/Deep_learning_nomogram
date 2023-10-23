import argparse
import json
import os
import time

from sklearn.model_selection import KFold
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torch import optim
from torch import nn
from models.CAUnet import CAUnet
from models.Unet_pp import NestedUNet
from models.Unet import U_Net
import torch
from Utils.data_transform import seg_transform
from Utils import lung_seg_dataset
from runner.Runner_utils import train_val_epoch


def train(args):
    if args.weight_path != "":
        print(f'load pretrained weights from {args.weight_path}')
        # state_dict = torch.load(args.finetune_from, map_location='cpu')
        # msg = net.load_state_dict(state_dict, strict=False)
        net_dict = args.model.state_dict()
        predict_model = torch.load(args.weight_path, map_location='cpu')
        # and ('detail.S' not in k)
        state_dict = {k: v for k, v in predict_model.items() if k in net_dict.keys()}  # 寻找网络中公共层，并保留预训练参数
        net_dict.update(state_dict)  # 将预训练参数更新到新的网络层
        msg = args.model.load_state_dict(net_dict)
        print('\tmissing keys: ' + json.dumps(msg.missing_keys))
        print('\tunexpected keys: ' + json.dumps(msg.unexpected_keys))

    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除head外，其他权重全部冻结
            if "head" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))
    pg = [p for p in model.parameters() if p.requires_grad]
    # 定义损失函数、优化器、步长衰减
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=5E-1)
    sche_lr = lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0.5, total_iters=int(args.epochs * 0.75))

    model_name = type(model).__name__ + "_fine"
    best_iou = 0
    best_val_iou = 0
    model.to(device)
    if os.path.exists("./Model_Dict") is False:
        os.makedirs("./Model_Dict")

    if os.path.exists("./logs") is False:
        os.makedirs("./logs")

    if args.record_loss:
        writer = SummaryWriter("./logs/coarse_seg{:s}".format(model_name))
    else:
        writer = None
    data_transform = seg_transform
    train_dataset = lung_seg_dataset.get_dataset(data_transform, "../dataset/roi/img_train/*.img",
                                                 "../dataset/roi/mask_train/*.img")
    val_dataset = lung_seg_dataset.get_val_test_dataset(data_transform, "../dataset/roi/img_test/*.img",
                                                        "../dataset/roi/mask_test/*.img")
    # total_dataset = lung_seg_dataset.get_dataset(data_transform, "../dataset/roi/img_train/*.img",
    #                                              "../dataset/roi/mask_train/*.img")
    batch_size = args.batch_size

    nw = 0  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    # # 划分数据集的索引序列
    # indices = list(range(len(total_dataset)))
    # 定义五折交叉验证的折数和训练轮数
    n_splits = 5

    # 定义 k-fold 交叉验证生成器
    kfold = KFold(n_splits=n_splits, shuffle=True)

    # for fold, (train_indices, val_indices) in enumerate(kfold.split(indices)):
    #     print(f"Fold {fold + 1}")
    #     train_dataset = [total_dataset[i] for i in train_indices]
    #     val_dataset = [total_dataset[i] for i in val_indices]
    #
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   pin_memory=True,
                                                   num_workers=nw, )

    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=4,
                                                 shuffle=False,
                                                 pin_memory=True,
                                                 num_workers=nw, )

    imgs, lables = next(iter(train_dataloader))
    img_size = imgs.size(2)
    train_val_epoch.img_pre_visualization(train_dataloader, True)
    train_val_epoch.img_pre_visualization(val_dataloader, True)
    for epoch in range(args.epochs):
        print(f"第{epoch + 1}轮学习开始")

        start = time.time()
        train_acc, train_loss, train_iou, train_pa, train_mpa = \
            train_val_epoch.train_seg_epoch(model, loss_fn, optimizer, train_dataloader, device, img_size)
        end = time.time()
        sche_lr.step()
        print("train_acc:{:.5f}, train_loss:{:.5f}, train_pa:{:.5f}, train_mpa:{:.5f}, train_time:{:.0f}m, {:.0f}s"
              .format(train_acc, train_loss, train_pa, train_mpa, (end - start) // 60, (end - start) % 60))
        print("train_iou------------lr")
        print(train_iou, sche_lr.get_lr())

        start = time.time()
        val_acc, val_loss, val_iou, val_pa, val_mpa = \
            train_val_epoch.val_seg_epoch(model, loss_fn, optimizer, val_dataloader, device, img_size)
        end = time.time()

        print("val_acc:{:.5f}, val_loss:{:.5f}, val_pa:{:.5f}, val_mpa:{:.5f},val_time:{:.0f}m, {:.0f}s"
              .format(val_acc, val_loss, val_pa, val_mpa, (end - start) // 60, (end - start) % 60))
        print("val_iou------------")
        print(val_iou)

        if (train_acc > 0.95) & (train_iou > 0.90) & (val_iou > 0.90) & (
                (best_iou < train_iou) | (best_val_iou < val_iou)):
            best_iou = train_iou
            best_val_iou = val_iou
            torch.save(model.state_dict(),
                       "./Model_Dict/{:.5f} {:.5f} {:s}.pth".format(train_iou.float(), val_iou.float(), model_name))
            print("模型保存中最好iou为{:.5f}, {:.5f}".format(train_iou, val_iou))
        print(f"第{epoch + 1}轮学习结束")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'

    # model = U_Net(in_ch=3, out_ch=2)
    model = CAUnet(3, 2)
    # model = NestedUNet(2, 3)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--lr', type=float, default=0.00001)
    parser.add_argument('--device', type=str, default=device)
    # 数据集路径
    parser.add_argument('--data_path', type=str, default="")
    # 预训练参数路径
    parser.add_argument('--weight_path', type=str, default="")
    # 是否冻结权重
    parser.add_argument('--freeze_layers', type=bool, default=False)
    # 是否记录loss变化
    parser.add_argument('--record_loss', type=bool, default=False)
    parser.add_argument('--model', default=model)

    args = parser.parse_args()

    train(args)
