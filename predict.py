import torch
from PIL import Image
from torchvision import transforms
import numpy as np
from  models.model import Net
import os
from torchvision import utils as vutils

def pre(img_path,threshold = 0.5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    img_pathA = os.path.join('/home/zzkk/xiaotong/LEVIRCD_FC\A',img_path)
    img_pathB = os.path.join('/home/zzkk/xiaotong/LEVIRCD_FC\B',img_path)
    label = os.path.join('/home/zzkk/xiaotong/LEVIRCD_FC\label',img_path)
    model = Net()
    model = model.to(device)
    model.load_state_dict(torch.load('checkpoints/best.pt'))
    model.eval()
    imageA,imageB,label = Image.open(img_pathA),Image.open(img_pathB),Image.open(label)

    loader1 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    loader = transforms.Compose([
            transforms.ToTensor()
    ])

    imgA, imgB = loader(imageA).unsqueeze(0), loader(imageB).unsqueeze(0)
    imageA,imageB = loader1(imageA).unsqueeze(0),loader1(imageB).unsqueeze(0)

    label = loader(label)
    imageA,imageB = imageA.to(device),imageB.to(device)
    model.eval()
    # imageA, imageB, labels = imageA.to(device), imageB.to(device), labels.to(device)
    _, outputs = model(imageA, imageB)
    outputs = outputs > threshold
    outputs = outputs.float()

    label = label.float()
    print(outputs.type())
    vutils.save_image(imgA, '预测图片/imageA.jpg')
    vutils.save_image(imgB, '预测图片/imageB.jpg')
    vutils.save_image(outputs, '预测图片/test.jpg')
    vutils.save_image(label, '预测图片/label.jpg')
    # 再加一个比较的
    # 如果多预测了那么就是红色 label=0,output=1
    # 如果少预测了那么就是绿色 label=1,output=0
    # 如果都是1 那么是白色
    # 如果都是0 那么是黑色

    outputs = outputs.squeeze()
    print(outputs.shape)
    print(label.shape)

    ans = torch.zeros(3, 256, 256)
    for i in range(256):
        for j in range(256):
            if outputs[i][j] == 1.0 and label[0][i][j] == 1:   #白色
                ans[0][i][j] = 255
                ans[1][i][j] = 255
                ans[2][i][j] = 255
            elif outputs[i][j] == 0.0 and label[0][i][j] == 0: #黑色
                ans[0][i][j] = 0
                ans[1][i][j] = 0
                ans[2][i][j] = 0
            elif outputs[i][j] == 0.0 and label[0][i][j] == 1: #绿色
                ans[0][i][j] = 0
                ans[1][i][j] = 255
                ans[2][i][j] = 0
            elif outputs[i][j] == 1.0 and label[0][i][j] == 0: #红色
                ans[0][i][j] = 255
                ans[1][i][j] = 0
                ans[2][i][j] = 0
    ans /= 255
    vutils.save_image(ans, '预测图片/bijiao.jpg')

    print(outputs.shape)

if __name__ == '__main__':
    pre('test_2_1.png')  #在这里改图片

