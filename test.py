import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from einops import rearrange

# 1. 修复后的PatchExpand模块（核心逻辑不变，确保输入通道数适配）
class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        # 通道扩展：dim_scale=2时，通道数翻倍（需确保 dim*dim_scale 能被 (dim_scale^2) 整除）
        self.expand = nn.Linear(dim, dim_scale * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)  # 重排后通道数 = 原通道数//dim_scale

    def forward(self, x):
        if hasattr(self.expand, 'weight'):
            x = x.to(self.expand.weight.device)
        
        # 通道扩展（此时 dim*dim_scale 需能被 (dim_scale^2) 整除）
        x = self.expand(x)
        B, H, W, C = x.shape
        
        # 维度重排：p1*p2=4，C必须是4的整数倍（修复后满足）
        x = rearrange(
            x, 
            'b h w (p1 p2 c) -> b (h p1) (w p2) c', 
            p1=self.dim_scale, 
            p2=self.dim_scale, 
            c=C // (self.dim_scale ** 2)  # 计算后为整数
        )
        
        # 归一化与形状恢复
        x = x.view(B, -1, C // (self.dim_scale ** 2))
        x = self.norm(x)
        x = x.reshape(B, H * self.dim_scale, W * self.dim_scale, C // (self.dim_scale ** 2))
        return x

# 2. 修复后的上采样模型（新增通道适配层）
class PatchExpandUpscaler(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        # 新增：通道适配层（将输入3通道→4通道，确保后续能被4整除）
        self.channel_adapt = nn.Conv2d(in_channels, 4, kernel_size=1, bias=False)
        
        # 三次上采样：通道数依次为 4→2→1→3（最终恢复3通道）
        self.expand1 = PatchExpand(input_resolution=(32, 32), dim=4, dim_scale=2)  # 32→64，通道4→2
        self.expand2 = PatchExpand(input_resolution=(64, 64), dim=2, dim_scale=2)  # 64→128，通道2→1
        self.expand3 = PatchExpand(input_resolution=(128, 128), dim=1, dim_scale=2) # 128→256，通道1→0.5？不，dim=1时dim_scale=2不触发线性层
        # 修复expand3：dim=1时，线性层为Identity，故重排后通道数=1（无需减半）
        self.expand3 = PatchExpand(input_resolution=(128, 128), dim=1, dim_scale=2)
        
        # 最终通道恢复：将1通道→3通道（匹配原始输入）
        self.final_conv = nn.Conv2d(1, 3, kernel_size=1, bias=False)

    def forward(self, x):
        # 步骤1：通道适配（3→4）
        x = self.channel_adapt(x)  # [1,3,32,32] → [1,4,32,32]
        
        # 步骤2：维度转换（适配PatchExpand的[B,H,W,C]格式）
        x = x.permute(0, 2, 3, 1)  # [1,4,32,32] → [1,32,32,4]
        
        # 步骤3：三次上采样
        x = self.expand1(x)  # [1,32,32,4] → [1,64,64,2]（通道4→2）
        x = self.expand2(x)  # [1,64,64,2] → [1,128,128,1]（通道2→1）
        x = self.expand3(x)  # [1,128,128,1] → [1,256,256,1]（通道保持1，无扩展）
        
        # 步骤4：恢复格式+通道（[B,H,W,C]→[B,C,H,W]，1→3）
        x = x.permute(0, 3, 1, 2)  # [1,256,256,1] → [1,1,256,256]
        x = self.final_conv(x)      # [1,1,256,256] → [1,3,256,256]
        
        return x

# 3. 本地生成32×32测试图（无修改，含彩色方块便于观察）
def generate_32x32_image():
    img_array = np.ones((32, 32, 3), dtype=np.uint8) * 255
    # 红色方块（左上）
    img_array[4:12, 4:12] = [255, 0, 0]
    # 绿色方块（右上）
    img_array[4:12, 20:28] = [0, 255, 0]
    # 蓝色方块（底部中间）
    img_array[20:28, 12:20] = [0, 0, 255]
    return Image.fromarray(img_array)

# 4. 图片-张量转换工具（无修改）
def img_to_tensor(img):
    # 归一化到0-1，形状[1,3,32,32]
    tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    return tensor

def tensor_to_img(tensor):
    # 恢复到0-255 uint8，去除批次维度
    img_np = (tensor.squeeze().permute(1, 2, 0).detach().numpy() * 255).astype(np.uint8)
    return Image.fromarray(img_np)

# 5. 主流程（无修改）
if __name__ == "__main__":
    # 初始化模型（CPU运行）
    model = PatchExpandUpscaler(in_channels=3)
    model.eval()  # 推理模式，关闭训练层
    
    # 生成32×32测试图
    print("正在生成32×32测试图...")
    img_32 = generate_32x32_image()
    print(f"原始图片尺寸：{img_32.size}")
    
    # 上采样
    print("正在执行上采样（32→256）...")
    tensor_32 = img_to_tensor(img_32)
    with torch.no_grad():  # 关闭梯度，加速
        tensor_256 = model(tensor_32)
    
    # 张量转图片
    img_256 = tensor_to_img(tensor_256)
    print(f"上采样后图片尺寸：{img_256.size}")
    
    # 对比显示
    plt.figure(figsize=(10, 5))
    # 原始图（放大显示）
    plt.subplot(1, 2, 1)
    plt.imshow(img_32.resize((128, 128)))
    plt.title("原始图（32×32，放大显示）")
    plt.axis("off")
    # 上采样图
    plt.subplot(1, 2, 2)
    plt.imshow(img_256)
    plt.title("PatchExpand上采样图（256×256）")
    plt.axis("off")
    plt.show()
    
    # 保存结果
    save_path = "fixed_patch_expand_upscaled_256x256.png"
    img_256.save(save_path)
    print(f"上采样图片已保存到：{save_path}")