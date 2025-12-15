
import torch
import torch.nn as nn

from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from functools import partial

import math
from einops import rearrange, repeat
import matplotlib.pyplot as plt


# Stem层  作用于模型最开始，作用是将通道数为3尺寸大小为256的图像  卷积为   通道数为32，尺寸大小为256的图
class StemLayer(nn.Module):
    def __init__(self,
                 in_channels=3,
                 out_channels=96,
                 act_layer=nn.GELU,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels,
                               out_channels // 2,
                               kernel_size=3,
                               stride=(1,1),  # 修改 stride 为 1
                               padding=1)  # 确保 padding 为 1
        self.norm1 = norm_layer(out_channels // 2)
        self.act = act_layer()
        self.conv2 = nn.Conv2d(out_channels // 2,
                               out_channels,
                               kernel_size=3,
                               stride=1,  # 修改 stride 为 1
                               padding=1)  # 确保 padding 为 1
        self.norm2 = norm_layer(out_channels)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.conv1(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm1(x)
        x = x.permute(0, 3, 1, 2)
        x = self.act(x)
        x = self.conv2(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm2(x)
        return x


# PatchMerging层（下采样）
class PatchMerging2D(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        B, H, W, C = x.shape

        SHAPE_FIX = [-1, -1]
        if (W % 2 != 0) or (H % 2 != 0):
            print(f"Warning, x.shape {x.shape} is not match even ===========", flush=True)
            SHAPE_FIX[0] = H // 2
            SHAPE_FIX[1] = W // 2

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C

        if SHAPE_FIX[0] > 0:
            x0 = x0[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x1 = x1[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x2 = x2[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x3 = x3[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]

        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, H // 2, W // 2, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

# PatchExpand （上采样层）
class PatchExpand(nn.Module):
    """
    Reference: https://arxiv.org/pdf/2105.05537.pdf
    """
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.expand = nn.Linear(dim, 2*dim, bias=False) if dim_scale==2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        x = x.to(self.expand.weight.device)
        # x = x.permute(0, 2, 3, 1)  # B, C, H, W ==> B, H, W, C
        x = self.expand(x)
        B, H, W, C = x.shape

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C//4)
        x = x.view(B,-1,C//4)
        x = self.norm(x)
        x = x.reshape(B, H*2, W*2, C//4)
        return x

# 按照通道划分
def split_features(feature_list, split_ratio=(1 / 4, 3 / 4)):
    """
    Splits each feature tensor in the list by channel ratio.

    Args:
    - feature_list (list of torch.Tensor): List of feature tensors from VSSMEncoder output.
    - split_ratio (tuple of float): The ratio to split channels per feature tensor (default is (1/4, 3/4)).

    Returns:
    - list of torch.Tensor: Features with channels according to the first ratio for each scale.
    - list of torch.Tensor: Features with channels according to the second ratio for each scale.
    """
    split_list_1 = []  # List to hold features split by the first ratio
    split_list_2 = []  # List to hold features split by the second ratio

    for features in feature_list:
        num_channels = features.shape[1]  # Get the number of channels in the feature tensor
        split_idx_1 = max(1, int(math.floor(num_channels * split_ratio[0])))  # Calculate split index for 1/4
        split_idx_2 = min(num_channels, int(math.ceil(num_channels * split_ratio[1])))  # Calculate split index for 3/4

        # Split the feature tensor into two parts according to the calculated indices
        split_features_1 = features[:, :split_idx_1, :, :]  # Channels for 1/4
        split_features_2 = features[:, split_idx_1:, :, :]  # Channels for 3/4 (remaining)

        # Append the split features to the corresponding lists
        split_list_1.append(split_features_1)
        split_list_2.append(split_features_2)

    return split_list_1, split_list_2

# 降通道操作
class ChannelReductionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ChannelReductionBlock, self).__init__()
        # 定义1x1卷积层以减少通道数
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        # 定义批量归一化层
        self.norm = nn.BatchNorm2d(out_channels)
        # 定义激活函数
        self.relu = nn.ReLU()

    def forward(self, x):
        # 应用1x1卷积层减少通道数
        x = self.conv(x)
        # 应用批量归一化
        x = self.norm(x)
        # # 应用激活函数
        x = self.relu(x)
        return x


# 将所得的结果输出在画布上
class printpic(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x1,x2,x3,x4):
        '''
        :param x1: 解码器最后得到的通道为3 尺寸大小为256的图片
        :param x2: 解码器最后得到的通道为3 尺寸大小为256的图片
        :param x3: 解码器最后得到的通道为3 尺寸大小为256的图片
        :param x4: 解码器最后得到的通道为3 尺寸大小为256的图片
        :return:
        '''

        # 将图片转换为可以输出在屏幕上的形式
        # 因为numpy操作只能在CPU进行，但是我在主函数中，将张量转移到了GPU，因此得先转回到CPU，但是进行.numpy操作，最后再转回GPU
        x1 = x1.cpu()
        x2 = x2.cpu()
        x3 = x3.cpu()
        x4 = x4.cpu()
        image_np1 = x1[0].permute(1, 2, 0).detach().numpy()
        image_np2 = x2[0].permute(1, 2, 0).detach().numpy()
        image_np3 = x3[0].permute(1, 2, 0).detach().numpy()
        image_np4 = x4[0].permute(1, 2, 0).detach().numpy()

        # image_np1 = image_np1.to('cuda:0')
        # image_np2 = image_np2.to('cuda:0')
        # image_np3 = image_np3.to('cuda:0')
        # image_np4 = image_np4.to('cuda:0')
        # 设置Matplotlib的字体，以支持中文显示
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题

        # 创建一个2行2列的子图
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))

        # 显示第一张图像
        axs[0, 0].imshow(image_np1)
        axs[0, 0].axis('off')  # 不显示坐标轴
        axs[0, 0].set_title('T0的独有和T0的公有')

        # 显示第二张图像
        axs[0, 1].imshow(image_np2)
        axs[0, 1].axis('off')  # 不显示坐标轴
        axs[0, 1].set_title('T0的独有和T1的公有')

        axs[1, 0].imshow(image_np3)
        axs[1, 0].axis('off')  # 不显示坐标轴
        axs[1, 0].set_title('T1的独有和T1的公有')

        # 显示第二张图像
        axs[1, 1].imshow(image_np4)
        axs[1, 1].axis('off')  # 不显示坐标轴
        axs[1, 1].set_title('T1的独有和T0的公有')
        plt.axis('off')
        plt.show()
        
        
class ConvBNReLUBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3), stride=(1,1), padding=1):
        super(ConvBNReLUBlock, self).__init__()
        # 卷积层
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        # 批量归一化层
        self.bn = nn.BatchNorm2d(out_channels)
        # ReLU激活函数
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class resblock(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=(3,3), stride=(1,1),padding=1):
        super(resblock,self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,kernel_size,stride,padding)
        self.IN = nn.InstanceNorm2d(out_channels)
        self.relu = nn.ReLU()
    def forward(self,x):
        x_res = self.conv(x)
        x_res = self.IN(x_res)
        x_res = self.relu(x_res)
        x_res = self.conv(x_res)
        x_res = self.IN(x_res)
        return x+x_res

