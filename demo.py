import torch
import torchvision.transforms as transforms
from PIL import Image
from models.model import Net  # 确保这行代码对应你的模型定义
from dataset import load_img  # 如果这是加载数据的方式
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
# 假设CDTrainer类已经定义，并且包含了_load_checkpoint方法来加载模型参数
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np

# 假设misc.metric_tool, models.myloss等路径正确，并且所有必要的类都已定义

class CDTrainer():
    def __init__(self, model_path, device='cuda'):
        self.model_path = model_path
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.net_G = Net().to(self.device)
        self.load_model()

    def load_model(self):
        """
        加载保存的模型参数。
        """
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.net_G.load_state_dict(checkpoint['model_G_state_dict'])

    def predict(self, imageA_path, imageB_path):
        """
        对两张照片进行预测。

        参数:
        imageA_path, imageB_path - 两张照片的路径。
        """
        # 定义图像预处理步骤
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # 加载并预处理图像
        imageA = Image.open(imageA_path).convert('RGB')
        imageB = Image.open(imageB_path).convert('RGB')
        imageA = transform(imageA)
        imageB = transform(imageB)

        # 增加批次维度
        imageA = imageA.unsqueeze(0)
        imageB = imageB.unsqueeze(0)

        # 确保数据在正确的设备上
        imageA, imageB = imageA.to(self.device), imageB.to(self.device)

        with torch.no_grad():
            # 前向传播以获得输出
            #G1_12,G2_12,G1_22,G2_22,G1_32,G2_32,G1_42,G2_42,G1_11,G2_11,G1_21,G2_21,G1_31,G2_31,G1_41,G2_41,out1, out2, out3, out4,change_map = self.net_G(imageA, imageB)
            out1, out2, out3, out4,change_map = self.net_G(imageA, imageB)

        print(1)
        # 返回模型的输出
        return out1, out2, out3, out4,change_map


class printpic(nn.Module):
    def __init__(self):
        super(printpic, self).__init__()

    def forward(self, input1,input2,x1, x2, x3, x4,changemap,ture):
        # 将图片转换为PIL图像格式以便于显示
        x1 = x1[0].permute(1, 2, 0).detach().cpu().numpy()
        x2 = x2[0].permute(1, 2, 0).detach().cpu().numpy()
        x3 = x3[0].permute(1, 2, 0).detach().cpu().numpy()
        x4 = x4[0].permute(1, 2, 0).detach().cpu().numpy()

        # 将NumPy数组转换为PIL图像
        from numpy import newaxis

        image1 = Image.fromarray((x1 * 255).astype('uint8'))
        image2 = Image.fromarray((x2 * 255).astype('uint8'))
        image3 = Image.fromarray((x3 * 255).astype('uint8'))
        image4 = Image.fromarray((x4 * 255).astype('uint8'))

        # 显示图像
        images = [input1,image1, image2,changemap,ture,input2, image3, image4,changemap,ture]
        titles = ['Input1','Image 1','Image 2','changemap','GT','Input2','Image 3', 'Image 4','changemap','GT']

        plt.figure(figsize=(10, 10))
        for i, (img, title) in enumerate(zip(images, titles), 1):
            plt.subplot(2, 5, i)
            if title == 'changemap':
                plt.imshow(changemap, cmap='gray')
                plt.title('Change Map')
                plt.axis('off')  # 不显示坐标轴
            else:
                if title == 'GT':
                    plt.imshow(ture, cmap='gray')
                    plt.title('GT')
                    plt.axis('off')  # 不显示坐标轴
                else:
                    plt.imshow(img)
                    plt.title(title)
                    plt.axis('off')

        plt.show()

# 使用示例
if __name__ == '__main__':
    # 模型参数文件的路径
    model_path = 'checkpoints_attention_cdd(11.25 lr=0.01 batch=24)/best.pt'

    # 创建CDTrainer实例
    trainer = CDTrainer(model_path)

    # 图片路径，需要替换为实际的图片路径
    imageA_path = 'cdd B1.jpg'
    imageB_path = 'cdd B2.jpg'
    imageC_path = 'cdd B3.jpg'

    imageA = Image.open(imageA_path).convert('RGB')
    imageA_np = np.array(imageA)

    imageB = Image.open(imageB_path).convert('RGB')
    imageB_np = np.array(imageB)

    imageC = Image.open(imageC_path).convert('L')
    imageC_np = np.array(imageC)

    # 使用模型进行预测
    out1, out2, out3, out4,change_map= trainer.predict(imageA_path, imageB_path)

    change_map = change_map.detach()
    argmax_map = torch.argmax(change_map, dim=1)

    # 将结果张量转换为二维的NumPy数组
    # 我们首先使用squeeze去除单维度，然后使用cpu()将张量转换为NumPy数组
    argmax_np = argmax_map.squeeze().cpu().numpy()

    # 使用matplotlib显示图像，这里使用'gray' colormap来显示黑白图
    # plt.figure(figsize=(10,10))
    # plt.imshow(argmax_np, cmap='gray')
    # plt.title('Change Map')
    # plt.axis('off')  # 不显示坐标轴
    # plt.show()

    output = printpic()
    output.forward(imageA_np,imageB_np,out1,out2,out3,out4,argmax_np,imageC_np)
    #output.forward(change_map,change_map,change_map,change_map,change_map,change_map)


    print(out1.shape)
