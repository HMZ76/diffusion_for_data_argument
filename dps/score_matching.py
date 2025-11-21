import os
import math
import glob
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse
import torch_npu
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子和默认数据类型
def set_seed(seed=42):
    """设置全局随机种子以确保可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.npu.is_available():
        torch.npu.manual_seed_all(seed)
    torch.set_default_dtype(torch.float32)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

# ===================== 数据预处理工具函数 =====================
def normalize_img(x):
    """将图像归一化到[-1, 1]范围"""
    x = np.array(x, dtype=np.float32) if not isinstance(x, np.ndarray) else x.astype(np.float32)
    if x.max() > 1.5:  # 0~255范围
        x = x / 127.5 - 1.0
    else:  # 0~1范围
        x = x * 2.0 - 1.0
    return x

def denormalize_img(x):
    """将[-1, 1]范围的张量反归一化到[0, 255]"""
    if isinstance(x, np.ndarray):
        return np.clip(((x + 1) * 127.5), 0, 255).astype(np.uint8)
    return ((x + 1) * 127.5).clamp(0, 255).to(torch.uint8)

def save_image(tensor, path):
    """保存张量为图像文件"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = denormalize_img(tensor.squeeze(0).cpu())
    Image.fromarray(img.numpy() if isinstance(img, torch.Tensor) else img).save(path)

def create_sample_grid(samples, epoch, save_path, n_cols=4):
    """创建生成样本的网格图"""
    n_samples = len(samples)
    n_rows = (n_samples + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    for i in range(n_samples):
        row = i // n_cols
        col = i % n_cols
        img_np = denormalize_img(samples[i].squeeze(0).cpu())
        axes[row, col].imshow(img_np, cmap='gray', vmin=0, vmax=255)
        axes[row, col].axis('off')
    
    # 隐藏多余的子图
    for i in range(n_samples, n_rows*n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].axis('off')
    
    plt.suptitle(f'Epoch {epoch} - DDPM Score Matching Generation', fontsize=16, y=0.98)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

# ===================== 数据集（无条件生成专用） =====================
class UnconditionalImageDataset(Dataset):
    def __init__(self, data_root, target_size=32):
        """
        无条件图像数据集：仅加载目标图像（32x32）
        data_root: 数据根目录，包含 .npy 图像文件
        target_size: 图像尺寸（默认32x32）
        """
        self.target_size = target_size
        # 加载所有.npy图像文件（支持example_target_*.npy或直接的图像文件）
        self.image_files = sorted(glob.glob(os.path.join(data_root, "*.npy")))
        # 过滤出目标图像文件（如果是之前的数据集格式）
        self.image_files = [f for f in self.image_files if "target" in f.lower() or "image" in f.lower()]
        
        assert len(self.image_files) > 0, f"未找到数据文件！在 {data_root} 中寻找 .npy 文件"
        
        print(f"加载了 {len(self.image_files)} 个图像文件")
        print(f"目标图像尺寸: {target_size}x{target_size}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        
        try:
            # 加载图像（shape: [H, W] 或 [C, H, W]）
            img = np.load(image_path).squeeze().astype(np.float32)
            
            # 归一化到[-1, 1]
            img = normalize_img(img)
            
            # 确保通道维度存在（[H, W] -> [1, H, W]）
            if len(img.shape) == 2:
                img = img[np.newaxis, ...]
            
            # 调整尺寸（如果不匹配）
            if img.shape[1] != self.target_size or img.shape[2] != self.target_size:
                img = torch.from_numpy(img).unsqueeze(0)  # [1, 1, H, W]
                img = F.interpolate(img, size=(self.target_size, self.target_size), mode='bilinear', align_corners=False)
                img = img.squeeze(0).numpy()  # [1, 32, 32]
            
            return torch.from_numpy(img).float()
            
        except Exception as e:
            print(f"加载数据出错 {image_path}: {e}")
            return torch.zeros(1, self.target_size, self.target_size).float()

class MultiScaleDataset(Dataset):
    def __init__(self, data_root, target_size=32, template_size=128):
        self.target_size = target_size
        self.template_size = template_size
        
        input_files = sorted(glob.glob(os.path.join(data_root, "example_input_*.npy")))
        target_files = sorted(glob.glob(os.path.join(data_root, "example_target_*.npy")))
        
        assert len(input_files) == len(target_files), f"文件数量不匹配"
        assert len(input_files) > 0, f"未找到数据文件"
        
        self.data_pairs = list(zip(input_files, target_files))
        print(f"加载了 {len(self.data_pairs)} 个数据对")
        print(f"条件图像尺寸: {template_size}x{template_size}")
        print(f"目标图像尺寸: {target_size}x{target_size}")

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        input_path, target_path = self.data_pairs[idx]
        
        try:
            template = np.load(input_path).squeeze().astype(np.float32)
            target = np.load(target_path).squeeze().astype(np.float32)
            
            if template.max() > 1.5:
                template = template / 255.0
            if target.max() > 1.5:
                target = target / 255.0
            
            condition = normalize_img(template)
            target = normalize_img(target)
            
            if random.random() < 0.1:
                condition = np.zeros_like(condition)
            
            condition_tensor = torch.from_numpy(condition).float().unsqueeze(0)
            target_tensor = torch.from_numpy(target).float().unsqueeze(0)
            
            return condition_tensor, target_tensor
            
        except Exception as e:
            print(f"加载数据出错 {input_path}: {e}")
            return (torch.zeros(1, self.template_size, self.template_size), 
                   torch.zeros(1, self.target_size, self.target_size))


# ===================== DDPM分数匹配UNet（修复残差连接通道数） =====================
class DDPMScoreUNet(nn.Module):
    def __init__(self, target_size=32, base_ch=64, time_embed_dim=128):
        super().__init__()
        self.target_size = target_size
        self.time_embed_dim = time_embed_dim
        self.base_ch = base_ch
        
        # 时间嵌入（DDPM核心：将噪声步t编码为特征）
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )
        
        # -------------------------- 编码器（修复残差连接通道数） --------------------------
        # 输入：(B, 1, 32, 32) → 输出：(B, base_ch, 32, 32)
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, base_ch, 3, padding=1),
            nn.GroupNorm(8, base_ch),
            nn.SiLU()
        )
        # 残差块：输入通道=base_ch（无时间嵌入拼接，避免通道不匹配）
        self.enc1_res = nn.Sequential(
            nn.Conv2d(base_ch, base_ch, 3, padding=1),
            nn.GroupNorm(8, base_ch),
            nn.SiLU(),
            nn.Conv2d(base_ch, base_ch, 3, padding=1),
            nn.GroupNorm(8, base_ch),
            nn.SiLU()
        )
        
        # 下采样：(B, base_ch, 32, 32) → (B, base_ch*2, 16, 16)
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_ch, base_ch*2, 3, padding=1, stride=2),
            nn.GroupNorm(8, base_ch*2),
            nn.SiLU()
        )
        self.enc2_res = nn.Sequential(
            nn.Conv2d(base_ch*2, base_ch*2, 3, padding=1),
            nn.GroupNorm(8, base_ch*2),
            nn.SiLU(),
            nn.Conv2d(base_ch*2, base_ch*2, 3, padding=1),
            nn.GroupNorm(8, base_ch*2),
            nn.SiLU()
        )
        
        # 下采样：(B, base_ch*2, 16, 16) → (B, base_ch*4, 8, 8)
        self.enc3 = nn.Sequential(
            nn.Conv2d(base_ch*2, base_ch*4, 3, padding=1, stride=2),
            nn.GroupNorm(8, base_ch*4),
            nn.SiLU()
        )
        self.enc3_res = nn.Sequential(
            nn.Conv2d(base_ch*4, base_ch*4, 3, padding=1),
            nn.GroupNorm(8, base_ch*4),
            nn.SiLU(),
            nn.Conv2d(base_ch*4, base_ch*4, 3, padding=1),
            nn.GroupNorm(8, base_ch*4),
            nn.SiLU()
        )
        
        # -------------------------- 中间层 --------------------------
        self.mid = nn.Sequential(
            nn.Conv2d(base_ch*4, base_ch*8, 3, padding=1),
            nn.GroupNorm(8, base_ch*8),
            nn.SiLU(),
            nn.Conv2d(base_ch*8, base_ch*8, 3, padding=1),
            nn.GroupNorm(8, base_ch*8),
            nn.SiLU(),
            nn.Conv2d(base_ch*8, base_ch*4, 3, padding=1),
            nn.GroupNorm(8, base_ch*4),
            nn.SiLU()
        )
        
        # -------------------------- 解码器（修复残差连接通道数） --------------------------
        # 上采样：(B, base_ch*4, 8, 8) → (B, base_ch*2, 16, 16)
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(base_ch*4, base_ch*2, 4, stride=2, padding=1),
            nn.GroupNorm(8, base_ch*2),
            nn.SiLU()
        )
        # 残差块：输入=base_ch*2（跳跃连接+上采样输出，通道一致）
        self.dec3_res = nn.Sequential(
            nn.Conv2d(base_ch*2, base_ch*2, 3, padding=1),
            nn.GroupNorm(8, base_ch*2),
            nn.SiLU(),
            nn.Conv2d(base_ch*2, base_ch*2, 3, padding=1),
            nn.GroupNorm(8, base_ch*2),
            nn.SiLU()
        )
        
        # 上采样：(B, base_ch*2, 16, 16) → (B, base_ch, 32, 32)
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(base_ch*2, base_ch, 4, stride=2, padding=1),
            nn.GroupNorm(8, base_ch),
            nn.SiLU()
        )
        self.dec2_res = nn.Sequential(
            nn.Conv2d(base_ch, base_ch, 3, padding=1),
            nn.GroupNorm(8, base_ch),
            nn.SiLU(),
            nn.Conv2d(base_ch, base_ch, 3, padding=1),
            nn.GroupNorm(8, base_ch),
            nn.SiLU()
        )
        
        # 最终输出：(B, base_ch, 32, 32) → (B, 1, 32, 32)
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_ch, base_ch, 3, padding=1),
            nn.GroupNorm(8, base_ch),
            nn.SiLU(),
            nn.Conv2d(base_ch, 1, 3, padding=1)  # 输出分数（与输入同维度）
        )
        
        # 时间嵌入融合层（单独的1x1卷积，将时间特征融入主特征，避免通道冲突）
        self.time_fuse1 = nn.Conv2d(time_embed_dim//4, base_ch, 1, padding=0)  # 适配enc1
        self.time_fuse2 = nn.Conv2d(time_embed_dim//4, base_ch*2, 1, padding=0)  # 适配enc2/dec3
        self.time_fuse3 = nn.Conv2d(time_embed_dim//4, base_ch*4, 1, padding=0)  # 适配enc3/mid
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """初始化卷积层权重"""
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def _expand_time_embed(self, time_embed, x):
        """将时间嵌入扩展到图像空间维度（B, D）->（B, D//4, H, W）"""
        B, D = time_embed.shape
        H, W = x.shape[2], x.shape[3]
        # 拆分时间嵌入为3份，分别适配各层尺寸
        embed1 = time_embed[:, :D//4].unsqueeze(-1).unsqueeze(-1).repeat(1, 1, H, W)  # (B, 32, 32, 32)
        embed2 = time_embed[:, D//4:D//2].unsqueeze(-1).unsqueeze(-1).repeat(1, 1, H//2, W//2)  # (B, 32, 16, 16)
        embed3 = time_embed[:, D//2:3*D//4].unsqueeze(-1).unsqueeze(-1).repeat(1, 1, H//4, W//4)  # (B, 32, 8, 8)
        return embed1, embed2, embed3
    
    def forward(self, x, t):
        """
        DDPM风格分数预测：输入带噪声图像+噪声步t，输出分数
        x: 带噪声的图像 (B, 1, 32, 32)
        t: 噪声步（0~T-1）(B, 1) → float类型
        return: 分数 (B, 1, 32, 32)
        """
        # 1. 时间嵌入（t为float，适配Linear层）
        time_embed = self.time_embed(t)  # (B, 128)
        embed1, embed2, embed3 = self._expand_time_embed(time_embed, x)  # 扩展为图像维度
        
        # 2. 编码器（通过1x1卷积融合时间特征，避免通道冲突）
        # 第一层：(B, 1, 32, 32) → (B, base_ch, 32, 32)
        h1 = self.enc1(x)  # (B, 64, 32, 32)
        time_fused1 = self.time_fuse1(embed1)  # (B, 32, 32, 32) → (B, 64, 32, 32)
        h1 = h1 + time_fused1  # 时间特征融合（残差加法，通道一致）
        h1 = self.enc1_res(h1) + h1  # 残差连接（64=64，通道匹配）
        
        # 第二层（下采样）：(B, 64, 32, 32) → (B, 128, 16, 16)
        h2 = self.enc2(h1)  # (B, 128, 16, 16)
        time_fused2 = self.time_fuse2(embed2)  # (B, 32, 16, 16) → (B, 128, 16, 16)
        h2 = h2 + time_fused2
        h2 = self.enc2_res(h2) + h2  # 残差连接（128=128）
        
        # 第三层（下采样）：(B, 128, 16, 16) → (B, 256, 8, 8)
        h3 = self.enc3(h2)  # (B, 256, 8, 8)
        time_fused3 = self.time_fuse3(embed3)  # (B, 32, 8, 8) → (B, 256, 8, 8)
        h3 = h3 + time_fused3
        h3 = self.enc3_res(h3) + h3  # 残差连接（256=256）
        
        # 3. 中间层
        h_mid = self.mid(h3)  # (B, 256, 8, 8)
        h_mid = h_mid + time_fused3  # 融合时间特征（256=256）
        
        # 4. 解码器（跳跃连接+时间融合）
        # 第三层（上采样）：(B, 256, 8, 8) → (B, 128, 16, 16)
        h_dec3 = self.dec3(h_mid)  # (B, 128, 16, 16)
        h_dec3 = h_dec3 + h2  # 跳跃连接（128=128）
        h_dec3 = h_dec3 + time_fused2  # 融合时间特征
        h_dec3 = self.dec3_res(h_dec3) + h_dec3  # 残差连接（128=128）
        
        # 第二层（上采样）：(B, 128, 16, 16) → (B, 64, 32, 32)
        h_dec2 = self.dec2(h_dec3)  # (B, 64, 32, 32)
        h_dec2 = h_dec2 + h1  # 跳跃连接（64=64）
        h_dec2 = h_dec2 + time_fused1  # 融合时间特征
        h_dec2 = self.dec2_res(h_dec2) + h_dec2  # 残差连接（64=64）
        
        # 5. 最终输出分数
        score = self.dec1(h_dec2)  # (B, 1, 32, 32)
        
        return score

# ===================== DDPM分数匹配核心模块 =====================
class DDPMScoreMatching:
    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02, device='npu'):
        """
        DDPM核心配置（基于VP-SDE）
        T: 扩散总步数
        beta_start/beta_end: 噪声强度调度（线性增加）
        device: 设备
        """
        self.T = T
        self.device = device
        
        # 1. DDPM噪声调度参数（线性beta调度）
        self.beta = torch.linspace(beta_start, beta_end, T, device=device, dtype=torch.float32)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)  # 累积乘积 ᾱ_t = α_1*α_2*...*α_t
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - self.alpha_bar)
        
        # 2. 反向扩散参数
        self.sqrt_alpha = torch.sqrt(self.alpha)
        self.one_over_sqrt_alpha = 1.0 / self.sqrt_alpha
        self.beta_over_sqrt_one_minus_alpha_bar = self.beta / self.sqrt_one_minus_alpha_bar
        
        print(f"DDPM配置：总扩散步数T={T}，beta范围[{beta_start:.4f}, {beta_end:.4f}]")

    def forward_diffusion(self, x_clean, t):
        """
        前向扩散：给干净图像添加t步噪声（DDPM前向过程）
        x_clean: 干净图像 (B, 1, 32, 32)
        t: 噪声步（0~T-1）(B, 1) → 整数类型
        return: 带噪声图像x_t, 噪声epsilon
        """
        # 采样标准高斯噪声
        eps = torch.randn_like(x_clean, device=self.device)
        
        # 将t转换为long类型（索引必须是整数）
        t_long = t.long().squeeze(1)  # (B, 1) → (B,)
        # 提取对应t步的ᾱ_t（扩展维度适配图像）
        sqrt_alpha_bar_t = self.sqrt_alpha_bar[t_long].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1, 1)
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bar[t_long].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        
        # 前向扩散公式：x_t = sqrt(ᾱ_t)x_0 + sqrt(1-ᾱ_t)ε
        x_t = sqrt_alpha_bar_t * x_clean + sqrt_one_minus_alpha_bar_t * eps
        
        return x_t, eps

    def score_matching_loss(self, model, x_clean):
        """
        DDPM风格分数匹配损失（去偏版本）
        核心：模型预测分数 -> 转换为噪声预测，与真实噪声计算MSE
        分数与噪声的关系：score(x_t, t) = -∇log p(x_t|x_0) ≈ (x_0 - x_t)/(1-ᾱ_t) ≈ -ε / sqrt(1-ᾱ_t)
        """
        B = x_clean.shape[0]
        
        # 生成整数类型的噪声步t（0~T-1），维度为(B,1)
        t = torch.randint(0, self.T, (B, 1), device=self.device, dtype=torch.long)
        
        # 前向扩散生成x_t和真实噪声eps
        x_t, eps_true = self.forward_diffusion(x_clean, t)
        
        # 模型预测分数score_pred（t转换为float适配Linear层）
        score_pred = model(x_t, t.float())
        
        # 将分数转换为噪声预测（基于分数与噪声的解析关系）
        t_long = t.long().squeeze(1)
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bar[t_long].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        eps_pred = -score_pred * sqrt_one_minus_alpha_bar_t  # ε_pred = -score * sqrt(1-ᾱ_t)
        
        # 去偏分数匹配损失（等价于噪声预测MSE，天然去偏）
        loss = F.mse_loss(eps_pred, eps_true)
        
        # 可选L2正则化
        reg_loss = 1e-6 * torch.norm(score_pred, p=2)
        total_loss = loss + reg_loss
        
        return total_loss

    @torch.no_grad()
    def ddpm_sampling(self, model, batch_size):
        """
        DDPM反向扩散采样（无朗之万，纯DDPM采样）
        从纯噪声x_T开始，迭代T步生成x_0
        """
        # 1. 初始化：x_T ~ N(0, I)（纯噪声）
        x_t = torch.randn((batch_size, 1, 32, 32), device=self.device, dtype=torch.float32)
        
        # 2. 反向扩散迭代（从T-1步降到0步）
        pbar = tqdm(range(self.T-1, -1, -1), desc="DDPM Sampling", leave=False)
        for t in pbar:
            # 构造当前步的t张量（B, 1），整数类型转float适配模型
            t_tensor = torch.tensor([t], device=self.device, dtype=torch.long).repeat(batch_size, 1).float()
            
            # 3. 模型预测分数
            score_pred = model(x_t, t_tensor)
            
            # 4. 转换分数为噪声预测
            sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bar[t].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            eps_pred = -score_pred * sqrt_one_minus_alpha_bar_t
            
            # 5. DDPM反向更新公式
            # x_{t-1} = (1/√α_t)(x_t - (β_t/√(1-ᾱ_t))ε_pred) + σ_t * z
            x_t = self.one_over_sqrt_alpha[t] * (x_t - self.beta_over_sqrt_one_minus_alpha_bar[t] * eps_pred)
            
            # 6. 添加反向噪声（t>0时，t=0时不添加）
            if t > 0:
                sigma_t = torch.sqrt(self.beta[t])  # DDPM默认sigma_t=√β_t
                z = torch.randn_like(x_t)
                x_t = x_t + sigma_t * z
            
            # 数值稳定
            x_t = torch.clamp(x_t, -1.5, 1.5)
            
            # 更新进度条
            pbar.set_postfix({"step": t})
        
        # 3. 返回生成的干净图像x_0
        return x_t
    @torch.no_grad()
    def dps_sampling(self, model, batch_size):
        """
        DDPM反向扩散采样（无朗之万，纯DDPM采样）
        从纯噪声x_T开始，迭代T步生成x_0
        """
        # 1. 初始化：x_T ~ N(0, I)（纯噪声）
        x_t = torch.randn((batch_size, 1, 32, 32), device=self.device, dtype=torch.float32)
        cond_dataset = MultiScaleDataset(data_root="../generated_data/", target_size=32, template_size=128)
        cond_dataloader = DataLoader(cond_dataset, batch_size=batch_size, shuffle=True)
        y = next(iter(cond_dataloader))[0].to(self.device)
        def stride_sampling_downsample_4x(y):
                return y[:,:,::4, ::4]
        y = stride_sampling_downsample_4x(y)
        # 2. 反向扩散迭代（从T-1步降到0步）
        pbar = tqdm(range(self.T-1, -1, -1), desc="DPS", leave=False)
        for t in pbar:
            # 构造当前步的t张量（B, 1），整数类型转float适配模型
            t_tensor = torch.tensor([t], device=self.device, dtype=torch.long).repeat(batch_size, 1).float()
            
            # 3. 模型预测分数
            score_pred = model(x_t, t_tensor)
            
            # 4. 转换分数为噪声预测
            sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bar[t].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            eps_pred = -score_pred * sqrt_one_minus_alpha_bar_t
            
            # 5. DDPM反向更新公式
            # x_{t-1} = (1/√α_t)(x_t - (β_t/√(1-ᾱ_t))ε_pred) + σ_t * z
            x_t = self.one_over_sqrt_alpha[t] * (x_t - self.beta_over_sqrt_one_minus_alpha_bar[t] * eps_pred)
            x0_hat = (x_t - self.sqrt_one_minus_alpha_bar[t] * eps_pred) / self.sqrt_alpha_bar[t]
            x_t = x_t - 0.3/torch.norm(y-x0_hat,2)*(y-x0_hat)
            # 6. 添加反向噪声（t>0时，t=0时不添加）
            if t > 0:
                sigma_t = torch.sqrt(self.beta[t])  # DDPM默认sigma_t=√β_t
                z = torch.randn_like(x_t)
                x_t = x_t + sigma_t * z
            
            # 数值稳定
            x_t = torch.clamp(x_t, -1.5, 1.5)
            
            # 更新进度条
            pbar.set_postfix({"step": t})
        
        # 3. 返回生成的干净图像x_0
        return x_t
# ===================== 主训练函数 =====================
def main():
    # 配置参数
    parser = argparse.ArgumentParser(description="DDPM Style Unconditional Score Matching")
    parser.add_argument("--data_root", type=str, default="../generated_data/", help="数据根目录")
    parser.add_argument("--target_size", type=int, default=32, help="目标图像尺寸")
    parser.add_argument("--batch_size", type=int, default=512, help="批次大小")
    parser.add_argument("--epochs", type=int, default=500, help="训练轮数")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--T", type=int, default=2000, help="DDPM总扩散步数")
    parser.add_argument("--beta_start", type=float, default=1e-4, help="DDPM初始beta")
    parser.add_argument("--beta_end", type=float, default=0.01, help="DDPM最终beta")
    parser.add_argument("--num_samples", type=int, default=16, help="每次生成的样本数量")
    parser.add_argument("--save_interval", type=int, default=5, help="模型保存间隔")
    parser.add_argument("--log_interval", type=int, default=10, help="日志打印间隔")
    parser.add_argument("--device", type=str, default='npu', help="设备（npu/cpu）")
    parser.add_argument("--resume", type=str, default="", help="恢复训练的模型路径")
    
    args = parser.parse_args()
    
    # 设备配置
    device = args.device if torch.npu.is_available() else 'cpu'
    print(f"使用设备: {device}")
    print(f"DDPM分数匹配配置：目标尺寸{args.target_size}x{args.target_size}，T={args.T}")
    print(f"训练参数：batch_size={args.batch_size}，学习率={args.lr}，总轮数={args.epochs}")
    
    # 创建保存目录
    checkpoint_dir = "./checkpoints_ddpm_score"
    sample_dir = "./samples_ddpm_score"
    log_dir = "./logs_ddpm_score"
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # 初始化TensorBoard
    writer = SummaryWriter(log_dir=log_dir)
    
    # 数据加载
    dataset = UnconditionalImageDataset(args.data_root, args.target_size)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4 if torch.npu.is_available() else 0,
        pin_memory=True,
        drop_last=True
    )
    
    # 初始化模型、优化器、调度器
    model = DDPMScoreUNet(args.target_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    # 恢复训练
    start_epoch = 0
    best_loss = float('inf')
    if args.resume and os.path.exists(args.resume):
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['best_loss']
        print(f"恢复训练：从epoch {start_epoch} 开始，最佳损失={best_loss:.6f}")
    
    # 初始化DDPM分数匹配模块
    ddpm_score_matching = DDPMScoreMatching(
        T=args.T,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        device=device
    )
    
    # 模型参数量统计
    param_count = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {param_count:,}")
    
    # 训练循环
    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        
        for step, x_clean in enumerate(pbar):
            # 数据移到设备
            x_clean = x_clean.to(device, dtype=torch.float32)
            
            # 计算DDPM分数匹配损失
            loss = ddpm_score_matching.score_matching_loss(model, x_clean)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
            optimizer.step()
            
            # 累计损失
            total_loss += loss.item()
            avg_loss = total_loss / (step + 1)
            
            # 更新进度条
            pbar.set_postfix({"loss": f"{loss.item():.6f}", "avg_loss": f"{avg_loss:.6f}"})
            
            # 打印日志
            if step % args.log_interval == 0:
                global_step = epoch * len(dataloader) + step
                writer.add_scalar("train/loss", loss.item(), global_step)
                writer.add_scalar("train/avg_loss", avg_loss, global_step)
                writer.add_scalar("train/lr", optimizer.param_groups[0]['lr'], global_step)
        
        # 学习率调度
        scheduler.step()
        
        # 计算epoch平均损失
        epoch_avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch} 完成 | 平均损失: {epoch_avg_loss:.6f} | 学习率: {scheduler.get_last_lr()[0]:.8f}")
        
        # 保存最佳模型
        if epoch_avg_loss < best_loss:
            best_loss = epoch_avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_loss': best_loss,
                'args': args
            }, os.path.join(checkpoint_dir, "best_model.pt"))
            print(f"保存最佳模型（损失: {best_loss:.6f}）")
        
        # 定期保存模型
        if epoch % args.save_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_loss': best_loss,
                'args': args
            }, os.path.join(checkpoint_dir, f"model_epoch_{epoch}.pt"))
        
        # 生成DDPM采样样本
        if epoch % args.save_interval == 0:
            model.eval()
            with torch.no_grad():
                print("开始DDPM采样...")
                samples = ddpm_score_matching.ddpm_sampling(
                    model,
                    batch_size=args.num_samples
                )
                
                # 保存样本网格图
                grid_path = os.path.join(sample_dir, f"epoch_{epoch}_ddpm_samples.png")
                create_sample_grid(samples, epoch, grid_path, n_cols=4)
                
                # 保存单个样本
                for i in range(min(5, args.num_samples)):
                    single_path = os.path.join(sample_dir, f"epoch_{epoch}_sample_{i}.png")
                    save_image(samples[i], single_path)
                
                # 记录样本到TensorBoard
                grid_img = torchvision.utils.make_grid(samples, nrow=4, normalize=True, scale_each=True)
                writer.add_image("ddpm_generated_samples", grid_img, epoch)
                
                print(f"DDPM样本已保存到 {sample_dir}")
    
    # 训练结束
    writer.close()
    print("DDPM风格分数匹配生成模型训练完成！")
    print(f"最佳模型保存路径: {os.path.join(checkpoint_dir, 'best_model.pt')}")
    print(f"生成样本保存路径: {sample_dir}")

# 单独采样脚本（训练后使用）
def sample_from_pretrained():
    parser = argparse.ArgumentParser(description="Sample from DDPM Score Matching Pretrained Model")
    parser.add_argument("--model_path", type=str, required=True, help="预训练模型路径")
    parser.add_argument("--num_samples", type=int, default=16, help="生成样本数量")
    parser.add_argument("--device", type=str, default='npu', help="设备（npu/cpu）")
    parser.add_argument("--output_dir", type=str, default="./ddpm_generated_samples_final", help="输出目录")
    
    args = parser.parse_args()
    
    # 设备配置
    device = args.device if torch.npu.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载模型和训练参数
    checkpoint = torch.load(args.model_path, map_location=device)
    model_args = checkpoint['args']
    model = DDPMScoreUNet(model_args.target_size).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"模型加载完成（训练epoch: {checkpoint['epoch']}）")
    
    # 初始化DDPM分数匹配模块（使用训练时的参数）
    ddpm_score_matching = DDPMScoreMatching(
        T=model_args.T,
        beta_start=model_args.beta_start,
        beta_end=model_args.beta_end,
        device=device
    )
    
    # DDPM采样
    print(f"开始生成 {args.num_samples} 个样本...")
    samples = ddpm_score_matching.dps_sampling(
        model,
        batch_size=args.num_samples
    )
    
    # 保存结果
    grid_path = os.path.join(args.output_dir, "ddpm_sample_grid.png")
    create_sample_grid(samples, checkpoint['epoch'], grid_path, n_cols=4)
    
    for i in range(args.num_samples):
        single_path = os.path.join(args.output_dir, f"ddpm_sample_{i}.png")
        save_image(samples[i], single_path)
    
    print(f"所有DDPM生成样本已保存到 {args.output_dir}")

if __name__ == "__main__":
    import sys
    if "--sample" in sys.argv:
        sys.argv.remove("--sample")
        sample_from_pretrained()
    else:
        main()