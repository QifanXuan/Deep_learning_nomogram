import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch

class ConvAttBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernelsize=1, stride=1):
        super(ConvAttBlock, self).__init__()
        pad = int((kernelsize-1)/2)
        self.gelu = nn.GELU()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=(1, kernelsize), stride=stride, padding=(0, pad))
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=(kernelsize, 1), stride=stride, padding=(pad, 0))
        self.bn = nn.BatchNorm2d(out_channel)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.gelu(x)

        x = self.conv2(x)
        x = self.bn(x)
        x = self.gelu(x)
        return x

class ConvAttention(nn.Module):
    def __init__(self, in_channel, out_channel, kernellist):
        super().__init__()
        self.left = ConvAttBlock(out_channel, out_channel, kernellist[0])
        self.mid = ConvAttBlock(out_channel, out_channel, kernellist[1])
        self.right = ConvAttBlock(out_channel, out_channel, kernellist[2])
        self.conv1 = nn.Conv2d(out_channel*4, out_channel, 3, padding=1)
        # self.conv1 = nn.Conv2d(in_channel, out_channel, 1)
    def forward(self, x):
        pre = x
        left = self.left(pre)
        mid = self.mid(pre)
        right = self.right(pre)
        x = torch.cat((left, mid, right, x), dim=1)
        # attn = pre+left+mid+right
        # x = self.conv1(x)
        x = self.conv1(x)
        return x
class Up(nn.Module):
    """Upscaling and concat"""

    def __init__(self):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [torch.div(diffX, 2, rounding_mode='trunc'), diffX - torch.div(diffX, 2, rounding_mode='trunc'),
                        torch.div(diffY, 2, rounding_mode='trunc'), diffY - torch.div(diffX, 2, rounding_mode='trunc')])
        # x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
        #                 diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return x

class up_conv(nn.Module):
    """
    Up Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        x = self.conv(x)
        return x

class CAUnet(nn.Module):
    def __init__(self, in_channel=3, num_classes=2):
        super().__init__()
        nb_filter = [32, 64, 128, 256]
        kernel_list = [7, 11, 21]
        self.pool = nn.MaxPool2d(2)
        self.up = Up()

        self.conv0_0 = ResBlock(in_channel, nb_filter[0])
        self.conv1_0 = ResBlock(nb_filter[0], nb_filter[1])
        self.conv2_0 = ResBlock(nb_filter[1], nb_filter[2])
        self.conv3_0 = ResBlock(nb_filter[2], nb_filter[3])

        self.conv0_1 = ResBlock(nb_filter[0] + nb_filter[1], nb_filter[0])
        self.conv1_1 = ResBlock(nb_filter[1] + nb_filter[2], nb_filter[1])
        self.conv2_1 = ResBlock(nb_filter[2] + nb_filter[3], nb_filter[2])

        self.conv0_2 = ResBlock(nb_filter[0] * 2 + nb_filter[1], nb_filter[0])
        self.conv1_2 = ResBlock(nb_filter[1] * 2 + nb_filter[2], nb_filter[1])

        self.conv0_3 = ResBlock(nb_filter[0] * 3 + nb_filter[1], nb_filter[0])

        self.conv_att3_0 = ConvAttention(nb_filter[3], nb_filter[3], kernel_list)
        self.conv_att2_1 = ConvAttention(nb_filter[2], nb_filter[2], kernel_list)
        self.conv_att1_2 = ConvAttention(nb_filter[1], nb_filter[1], kernel_list)

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(self.up(x1_0, x0_0))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(self.up(x2_0, x1_0))
        x0_2 = self.conv0_2(self.up(x1_1, torch.cat([x0_0, x0_1], 1)))

        x3_0 = self.conv_att3_0(self.conv3_0(self.pool(x2_0)))
        x2_1 = self.conv_att2_1(self.conv2_1(self.up(x3_0, x2_0)))
        x1_2 = self.conv_att1_2(self.conv1_2(self.up(x2_1, torch.cat([x1_0, x1_1], 1))))
        x0_3 = self.conv0_3(self.up(x1_2, torch.cat([x0_0, x0_1, x0_2], 1)))

        out = self.final(x0_3)
        return out

    def make_stage(self, block, in_channel, out_channel, block_num, stride=1, dilate=False):
        if dilate:
            self.dilation *= stride
            stride = 1
        layers = []
        layers.append(block(in_channel, out_channel, stride))
        in_channel = out_channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(in_channel, out_channel, stride))

        return nn.Sequential(*layers)

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

if __name__ == '__main__':
    x = torch.randn((1, 3, 256, 256))
    module = CAUnet()
    y_pred = module(x)
    print(y_pred.shape)
    # print(x.shape)
    # mymodule = U_Net(block=ResBlock)
    # y = mymodule(x)
    # print(y.shape)
    # module = U_Net(ResBlock, 3, 2)
    # y_pred = module(x)
    # print(y_pred.shape)