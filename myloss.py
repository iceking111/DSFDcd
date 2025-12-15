import torch
import torch.nn as nn
import torch.nn.functional as F


# 交叉熵损失函数 重构图比较
class CustomCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CustomCrossEntropyLoss,self).__init__()

    def forward(self,input,target):
        input1 = input[:,0,:,:].unsqueeze(1)
        input2 = input[:,1,:,:].unsqueeze(1)
        loss1 = F.binary_cross_entropy_with_logits(input1, target)
        loss2 = F.binary_cross_entropy_with_logits(input2, 1 - target)
        return loss1 + loss2

# 重构图进行比较
class mse_loss(nn.Module):
    def __init__(self):
        super(mse_loss, self).__init__()
    def forward(self, predictions, targets):
        predictions = predictions.to('cuda:0')
        targets = targets.to('cuda:0')
        return F.mse_loss(predictions, targets)

# 争对于二范数损失函数
class L2NormLossForBCHW(nn.Module):
    def __init__(self):
        super(L2NormLossForBCHW, self).__init__()

    def forward(self, predictions, targets):
        # 确保 predictions 和 targets 是张量
        if not (isinstance(predictions, torch.Tensor) and isinstance(targets, torch.Tensor)):
            raise ValueError("要做损失的两个东西都是张量[B,3,256,256]")

        # 确保预测值和目标值是浮点数，并且都在 GPU 上
        predictions = predictions.float().to('cuda:0')
        targets = targets.float().to('cuda:0')

        # 计算预测值和目标值之间的二范数，这里 dim 参数设置为 (1, 2, 3) 表示沿着通道、高度和宽度维度计算范数
        norm = torch.norm(predictions - targets, p=2, dim=(2, 3))

        # 计算平均二范数损失
        loss = norm.mean()

        # 返回平均二范数损失
        return loss

class Loss(nn.Module):
    def __init__(self):
        super(Loss,self).__init__()

    def forward(self,prediction,target):

        prediction = torch.mean(prediction,dim=1,keepdim=True)
        prediction = F.softmax(prediction,dim=1)


        print(prediction.shape)

# L1范数和L2范数
def L1_loss(x,y):
    x = x.to('cuda')
    y = y.to('cuda')
    loss = torch.mean(torch.abs(x - y))
    return loss

def L2_loss(x,y):
    x = x.to('cuda')
    y = y.to('cuda')
    loss = torch.mean((x - y) ** 2)
    return loss

def L2_regularize(x):
    loss = torch.mean(torch.pow(x,2))
    return loss


# 余弦相似度
class CosineSimilarityLoss(nn.Module):
    def __init__(self, margin=0.0, reduction='mean'):
        """
        初始化余弦相似度损失类。
        参数:
        margin -- 边界值，用于调整损失函数的敏感度。
        reduction -- 指定应用于输出的减少方式，'mean' | 'sum' | 'none'。
        """
        super(CosineSimilarityLoss, self).__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(self, input1, input2,target):
        """
        计算余弦相似度损失。
        参数:
        input1 -- 第一个输入张量。
        input2 -- 第二个输入张量。
        target -- 目标张量，表示两个输入张量是否应该相似（1 表示相似，-1 表示不相似）。
        返回:
        余弦相似度损失。
        """
        # 计算输入张量的余弦相似度
        cosine_similarity = F.cosine_similarity(input1, input2, dim=1)
        # 根据目标值调整损失
        if target == 1:  # 相似
            loss = 1 - cosine_similarity
        else:  # 不相似
            # loss = F.relu(self.margin - cosine_similarity)
            loss = - cosine_similarity
        # 根据reduction参数调整损失的维度
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss




def compute_kl_loss(p_logits, q_logits):

    p_logits = p_logits.to('cuda')
    q_logits = q_logits.to('cuda')

    p = F.log_softmax(p_logits, dim=-1, dtype=torch.float32)
    p_tec = F.softmax(p_logits, dim=-1, dtype=torch.float32)
    q = F.log_softmax(q_logits, dim=-1, dtype=torch.float32)
    q_tec = F.softmax(q_logits, dim=-1, dtype=torch.float32)

    p_loss = F.kl_div(p, q_tec, reduction='none').sum()
    q_loss = F.kl_div(q, p_tec, reduction='none').sum()

    # loss = (p_loss + q_loss) / 2 * 0.01
    loss = q_loss * 0.02
    return loss


def compute_kl_loss1(p_logits, q_logits):
    # 确保输入在GPU上
    p_logits = p_logits.to('cuda')
    q_logits = q_logits.to('cuda')

    # 计算log_softmax和softmax
    p = F.log_softmax(p_logits, dim=1)  # 对通道维度应用log_softmax
    q = F.softmax(q_logits, dim=1)  # 对通道维度应用softmax

    # 计算KL散度，并对通道、高度和宽度维度求和
    kl_loss = F.kl_div(p, q, reduction='none').sum(dim=(1, 2, 3))

    # 取批量中所有样本的平均损失
    loss = kl_loss.mean() * 0.005
    return loss
    
    
    

if __name__ == '__main__':
    # 定义logits和目标
    # 注意：logits 应该是未经softmax的预测值
    logits = torch.randn(1, 2, 256, 256)  # 所得2通道的图
    targets = torch.randn(1, 1, 256, 256)  # 目标变化图
    #

    input1 = torch.ones(1, 3, 256, 256)
    input2 = torch.zeros(1, 3, 256, 256)

    print(input1.shape)
    LOSS = L2NormLossForBCHW()
    print("L2损失函数为:", L2_loss(input1,input2))
    print("L2损失函数为:", LOSS(input1, input2))
    #print("二范数损失函数为:", loss2_1.item())

    # 余弦相似度示例使用
    # 假设有两个输入张量 input1 和 input2，以及目标标签 target
    # input1 = torch.randn(1,2,256,256)  # 假设有10个样本，每个样本有5个特征
    # input2 = torch.randn(1,2,256,256)
    # 创建余弦相似度损失函数实例
    csl_loss = CosineSimilarityLoss(margin=0.5)
    # 计算余弦相似度损失
    loss1 = csl_loss(input1, input2,1)
    print(loss1)

    loss2 = csl_loss(input1, input2,-1)
    print(loss2)


