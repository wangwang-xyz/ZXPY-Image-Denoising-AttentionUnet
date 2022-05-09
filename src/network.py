import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.tensorboard import SummaryWriter

# Channel Attention Module in CBAM Model
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.f1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.f2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.f2(self.relu(self.f1(self.avg_pool(x))))
        max_out = self.f2(self.relu(self.f1(self.max_pool(x))))
        out = self.sigmoid(avg_out + max_out)
        return torch.mul(x, out)

# Spatial Attention Module in CBAM Model
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv(out))
        return torch.mul(x, out)

# CBAM Model
class CBAM(nn.Module):
    # 
    def __init__(self, channel):  
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channel)
        self.spatial_attention = SpatialAttention(7)

    def forward(self, x):
   		# 将最后的标准卷积模块改为了注意力机制提取特征
        return self.spatial_attention(self.channel_attention(x))

# Modefied Unet with CBAM Architecture
class ModefiedUnet(nn.Module):
    def __init__(self, in_channels=4, out_channels=4):
        super(ModefiedUnet, self).__init__()

        # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.conv1_1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.attention1 = CBAM(32)

        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.attention2 = CBAM(64)

        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.attention3 = CBAM(128)

        self.conv4_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2)
        self.attention4 = CBAM(256)

        self.conv5_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.attention5 = CBAM(512)

        self.upv6 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv6_1 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.conv6_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.upv7 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv7_1 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv7_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.upv8 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv8_1 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.conv8_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.upv9 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv9_1 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv9_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)

        self.conv10_1 = nn.Conv2d(32, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        n, c, h, w = x.shape
        h_pad = 32 - h % 32 if not h % 32 == 0 else 0
        w_pad = 32 - w % 32 if not w % 32 == 0 else 0
        padded_image = F.pad(x, (0, w_pad, 0, h_pad), 'replicate')

        conv1 = self.leaky_relu(self.conv1_1(padded_image))
        conv1 = self.leaky_relu(self.conv1_2(conv1))
        pool1 = self.pool1(conv1)
        conv1 = self.attention1(conv1)

        conv2 = self.leaky_relu(self.conv2_1(pool1))
        conv2 = self.leaky_relu(self.conv2_2(conv2))
        pool2 = self.pool1(conv2)
        conv2 = self.attention2(conv2)

        conv3 = self.leaky_relu(self.conv3_1(pool2))
        conv3 = self.leaky_relu(self.conv3_2(conv3))
        pool3 = self.pool1(conv3)
        conv3 = self.attention3(conv3)

        conv4 = self.leaky_relu(self.conv4_1(pool3))
        conv4 = self.leaky_relu(self.conv4_2(conv4))
        pool4 = self.pool1(conv4)
        conv4 = self.attention4(conv4)

        conv5 = self.leaky_relu(self.conv5_1(pool4))
        conv5 = self.leaky_relu(self.conv5_2(conv5))
        conv5 = self.attention5(conv5)

        up6 = self.upv6(conv5)
        up6 = torch.cat([up6, conv4], 1)
        conv6 = self.leaky_relu(self.conv6_1(up6))
        conv6 = self.leaky_relu(self.conv6_2(conv6))

        up7 = self.upv7(conv6)
        up7 = torch.cat([up7, conv3], 1)
        conv7 = self.leaky_relu(self.conv7_1(up7))
        conv7 = self.leaky_relu(self.conv7_2(conv7))

        up8 = self.upv8(conv7)
        up8 = torch.cat([up8, conv2], 1)
        conv8 = self.leaky_relu(self.conv8_1(up8))
        conv8 = self.leaky_relu(self.conv8_2(conv8))

        up9 = self.upv9(conv8)
        up9 = torch.cat([up9, conv1], 1)
        conv9 = self.leaky_relu(self.conv9_1(up9))
        conv9 = self.leaky_relu(self.conv9_2(conv9))

        conv10 = self.conv10_1(conv9)
        out = conv10[:, :, :h, :w]

        return out

    def leaky_relu(self, x):
        out = torch.max(0.2 * x, x)
        return out


if __name__ == "__main__":
    test_input = torch.from_numpy(np.random.randn(1, 4, 512, 512)).float()
    net = ModefiedUnet()
    output = net(test_input)

    log = SummaryWriter("logs/model")
    log.add_graph(net, test_input)
    log.close()

    print("========================= Summary ==============================")
    total_num = sum(p.numel() for p in net.parameters())
    print("Number of parameter: %.2fM" % (total_num / 1e6))
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("Number of parameter need to be trained: %.2fM" % (trainable_num / 1e6))

    print("test over")
