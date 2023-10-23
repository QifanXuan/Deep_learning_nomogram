import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch



"""
    构造上采样模块--左边特征提取基础模块    
"""
class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1):
        super(ResBlock, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channel, out_channel, 3, stride, 1)
        self.bn = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, 3, stride, 1)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channel != out_channel * ResBlock.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel * ResBlock.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channel * ResBlock.expansion)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = identity + x
        return x

class conv_block(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            # 在卷积神经网络的卷积层之后总会添加BatchNorm2d进行数据的归一化处理，这使得数据在进行Relu之前不会因为数据过大而导致网络性能的不稳定
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


"""
    构造下采样模块--右边特征融合基础模块    
"""

class up_conv(nn.Module):
    """
    Up Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

"""
    模型主架构
"""

class U_Net(nn.Module):
    def __init__(self, block=conv_block, in_ch=3, out_ch=2,):
        super(U_Net, self).__init__()

        # 卷积参数设置
        filters = [64, 128, 256, 512, 1024]

        # 最大池化层
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 左边特征提取卷积层
        self.Conv1 = block(in_ch, filters[0])
        self.Conv2 = block(filters[0], filters[1])
        self.Conv3 = block(filters[1], filters[2])
        self.Conv4 = block(filters[2], filters[3])
        self.Conv5 = block(filters[3], filters[4])

        # 右边特征融合反卷积层
        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        e1 = self.Conv1(x)

        e2_pool = self.Maxpool1(e1)
        e2 = self.Conv2(e2_pool)

        e3_pool = self.Maxpool2(e2)
        e3 = self.Conv3(e3_pool)

        e4_pool = self.Maxpool3(e3)
        e4 = self.Conv4(e4_pool)

        e5_pool = self.Maxpool4(e4)
        e5 = self.Conv5(e5_pool)

        d5_up= self.Up5(e5)
        d5_cat = torch.cat((e4, d5_up), dim=1)  # 将e4特征图与d5特征图横向拼接

        d5 = self.Up_conv5(d5_cat)

        d4_up = self.Up4(d5)
        d4_cat = torch.cat((e3, d4_up), dim=1)  # 将e3特征图与d4特征图横向拼接

        d4 = self.Up_conv4(d4_cat)

        d3_up = self.Up3(d4)
        d3_cat = torch.cat((e2, d3_up), dim=1)  # 将e2特征图与d3特征图横向拼接

        d3 = self.Up_conv3(d3_cat)

        d2_up = self.Up2(d3)
        d2_cat = torch.cat((e1, d2_up), dim=1)  # 将e1特征图与d1特征图横向拼接

        d2 = self.Up_conv2(d2_cat)

        out = self.Conv(d2)

        return out

if __name__ == '__main__':
    x = torch.randn((10, 3, 400, 400))

    # print(x.shape)
    # mymodule = U_Net(block=ResBlock)
    # y = mymodule(x)
    # print(y.shape)
    module = U_Net(ResBlock, 3, 2)
    y_pred = module(x)
    print(y_pred.shape)