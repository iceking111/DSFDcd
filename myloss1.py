import torch
import torch.nn as nn
import torch.nn.functional as F



class BCE_loss(nn.Module):
    def __init__(self):
        super(BCE_loss, self).__init__()

    def forward(self, input, target):
        input = torch.argmax(input,dim=1)
        input = input.float()
        input = input.unsqueeze((1))
        #s = 1 - input
        # input = torch.cat((s.reshape(-1, 1), input.reshape(-1, 1)), dim=1)
        # print(input.shape)

        target = target.float()
        loss1 = F.binary_cross_entropy_with_logits(input,target)
        loss2 = F.binary_cross_entropy_with_logits(input,1-target)
      

        return loss1 + loss2
class BCE_loss1(nn.Module):
    def __init__(self):
        super(BCE_loss, self).__init__()

    def forward(self, input, target):
        # 确保 input 是二维张量，并且使用正确的维度来找到最大值
        # 使用 torch.where 来创建一个与 input 形状相同的 0 和 1 的张量
        # 这可以被看作是将 logits 转换为概率
        input = torch.argmax(input, dim=1).long
        target = target.long()
        if target.dim() == 4:
            target = torch.squeeze(target, dim=1)

        return F.cross_entropy(input=input,target=target,weight=None,ignore_index=255,reduction='mean') * 15

        # 将两个损失相加



class MyEntropyLoss(nn.Module): #创新
    def __init__(self):
        super(MyEntropyLoss, self).__init__()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, outputs, labels):#自己写损失函数

        outputs = self.softmax(outputs)
        outputs = outputs[:, 1:2, :, :]              # 先切片得到第二个变化的图片

        nc = torch.sum((labels == 1).float())                       # 加和就是1的数量   1就是变化，0就是没变化
        nu = torch.sum((labels == 0).float())

        loss1 = 0
        loss2 = 0
        if nc != 0:                                  # 写了这个就不用写加一个特别小的数字了
            loss1 = torch.sum(labels * torch.clamp(3.5 - outputs, min=0.0)) / nc

        if nu != 0:
            loss2 = torch.sum((1 - labels) * outputs) / nu

        loss = loss1 + loss2

        return loss



class CombinedLoss(nn.Module):

    """
    logSoftmax_with_loss
    :param input: torch.Tensor, N*C*H*W
    :param target: torch.Tensor, N*1*H*W,/ N*H*W
    :param weight: torch.Tensor, C
    :return: torch.Tensor [0]
    """
    def __init__(self):
        super(CombinedLoss, self).__init__()
    def forward(self, input, target, weight=None, reduction='mean',ignore_index=255):
        target = target.long()
        if target.dim() == 4:
            target = torch.squeeze(target, dim=1)
        if input.shape[-1] != target.shape[-1]:
            input = F.interpolate(input, size=target.shape[1:], mode='bilinear',align_corners=True)

        return F.cross_entropy(input=input, target=target, weight=weight,
                               ignore_index=ignore_index, reduction=reduction)




if __name__ == '__main__':
    net = CombinedLoss()
    # input = torch.tensor([[3,0.3, 0.2, 0.6], [3,0.8, 0.4, 0.9]])
    batch = 3
    input = torch.randn([batch,2,256,256])

    target = torch.randn([batch,1,256,256])

    BCE_loss_example = BCE_loss()
    loss1 = BCE_loss_example(input, target)

    print(loss1.item())
