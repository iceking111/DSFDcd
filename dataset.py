import os
import torch
import torch.utils.data as D
from torchvision import transforms
import random
from torchvision.transforms import functional as F
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

# 每次迭代喂进网络的图像都是数据增强的图片，
# 如果增强方法比较多，那么每次给网络的图像都是不一样的，间接增加了训练的数据量

# 以指定的概率水平翻转图像。
class RandomHorizontalFlip1(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, imageA,imageB,label):
        if random.random() < self.flip_prob:
            imageA = F.hflip(imageA)
            imageB = F.hflip(imageB)
            label = F.hflip(label)
        return imageA,imageB,label

# 以指定的概率垂直翻转图像。
class RandomVerticalFlip1(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, imageA,imageB,label):
        if random.random() < self.flip_prob:
            imageA = F.vflip(imageA)
            imageB = F.vflip(imageB)
            label = F.vflip(label)
        return imageA,imageB,label
# 将图像数据标准化到特定的均值和标准差。
class Normalize111(object):
    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self,imageA,imageB,label):
        imageA = F.normalize(imageA, self.mean, self.std, self.inplace)
        imageB = F.normalize(imageB, self.mean, self.std, self.inplace)
        return imageA,imageB,label
# 将 PIL 图像转换为 PyTorch 张量。
class ToTensor1(object):
    def __call__(self, imageA,imageB,label):
        imgA = F.to_tensor(imageA)
        imgB = F.to_tensor(imageB)
        label = F.to_tensor(label)
        return imgA,imgB,label

# 组合多个数据增强操作为一个操作。
class Compose1(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, imageA,imageB,label):
        for t in self.transforms:
            imageA,imageB,label = t(imageA,imageB,label)

        return imageA,imageB,label
# 按给定的信噪比和概率向图像添加椒噪声。
class AddPepperNoise(object):
    """"
    Args:
        snr (float): Signal Noise Rate
        p (float): 概率值， 依概率执行
    """

    def __init__(self, snr, p=0.9):
        assert isinstance(snr, float) and (isinstance(p, float))
        self.snr = snr
        self.p = p

    def addZao(self,img):
        # 把img转化成ndarry的形式
        img_ = np.array(img).copy()
        h, w, c = img_.shape
        # 原始图像的概率（这里为0.9）
        signal_pct = self.snr
        # 噪声概率共0.1
        noise_pct = (1 - self.snr)
        mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[signal_pct, noise_pct / 2., noise_pct / 2.])
        # 将mask按列复制c遍
        mask = np.repeat(mask, c, axis=2)
        img_[mask == 1] = 255  # 盐噪声
        img_[mask == 2] = 0  # 椒噪声
        return Image.fromarray(img_.astype('uint8')).convert('RGB')  # 转化为PIL的形式

    def __call__(self, imageA,imageB,label):
        if random.uniform(0, 1) < self.p: # 按概率进行
            imageA = self.addZao(imageA)
            imageB = self.addZao(imageB)
            return imageA,imageB,label
        else:
            return imageA,imageB,label

# 配置了训练、测试和验证数据集的数据增强流程，使用 Compose1 包装了多个数据增强操作。
def load_img(root_dir):
    # 数据增强和标准化的配置
    data_transforms = {
        'train': Compose1([
            RandomHorizontalFlip1(0.5),
            RandomVerticalFlip1(0.5),
            ToTensor1(),
            Normalize111((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]),
        'test': Compose1([
            ToTensor1(),
            Normalize111((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]),
        'val': Compose1([
            ToTensor1(),
            Normalize111((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    }
    # 定义了 MyDataset 类，继承自 torch.utils.data.Dataset，用于加载图像和标签。
    class MyDataset(Dataset):
        def __init__(self, phase, transform=None):# __init__ 方法初始化数据集的路径，并列出所有图像路径。
            self.transform = transform
            self.phase_dir = os.path.join(root_dir, phase)  # 构建阶段目录路径
            self.A_dir = os.path.join(self.phase_dir, 'A')
            self.B_dir = os.path.join(self.phase_dir, 'B')
            self.label_dir = os.path.join(self.phase_dir, 'label')

            # 列出所有图片路径
            self.imgA_paths = [os.path.join(self.A_dir, f) for f in os.listdir(self.A_dir) if f.endswith('.jpg')]
            self.imgB_paths = [os.path.join(self.B_dir, f) for f in os.listdir(self.B_dir) if f.endswith('.jpg')]
            self.label_paths = [os.path.join(self.label_dir, f) for f in os.listdir(self.label_dir) if
                                f.endswith('.jpg')]

        def __len__(self):# __len__ 方法返回数据集中的样本数量。
            return min(len(self.imgA_paths), len(self.imgB_paths), len(self.label_paths))

        def __getitem__(self, idx):# __getitem__ 方法根据索引加载并返回一个批次的图像和标签。
            imgA_path, imgB_path, label_path = self.imgA_paths[idx], self.imgB_paths[idx], self.label_paths[idx]
            imgA = Image.open(imgA_path).convert('RGB')
            imgB = Image.open(imgB_path).convert('RGB')
            label = Image.open(label_path).convert('L')  # 标签通常使用灰度图

            if self.transform:
                imgA, imgB, label = self.transform(imgA, imgB, label)

            return imgA, imgB, label

    # 创建数据集和数据加载器
    image_datasets = {x: MyDataset(x, data_transforms[x]) for x in ['train', 'test', 'val']}
    batch_size = 1  # 根据显存大小和需求设置
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True) for x in
                   ['train', 'test', 'val']}

    return dataloaders['train'], dataloaders['test'], dataloaders['val'], batch_size


# 主函数测试代码不变
if __name__ == "__main__":
    root_dir = 'dataset1/levircd11'  # 确保路径正确
    train_loader, test_loader, val_loader, batch_size = load_img(root_dir)

    # 测试训练数据加载
    for i, (imagesA, imagesB, labels) in enumerate(train_loader):
        print(
            f"Batch {i + 1}: imagesA shape: {imagesA.size()}, imagesB shape: {imagesB.size()}, labels shape: {labels.size()}")
        if i == 0:
            break  # 只打印第一个批次的数据
