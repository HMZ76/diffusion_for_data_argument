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
import argparse
import torch_npu
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.npu.is_available():
        torch.npu.manual_seed_all(seed)
    torch.set_default_dtype(torch.float32)

set_seed()

# ===================== 工具函数 =====================
def denormalize_img(x):
    if isinstance(x, np.ndarray):
        return np.clip(((x + 1) * 127.5), 0, 255).astype(np.uint8)
    return ((x + 1) * 127.5).clamp(0, 255).to(torch.uint8)

def save_image(tensor, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = denormalize_img(tensor.squeeze(0).cpu())
    Image.fromarray(img.numpy() if isinstance(img, torch.Tensor) else img).save(path)

# ===================== 数据集（仅用于DPS条件采样） =====================
class MultiScaleDataset(Dataset):
    def __init__(self, data_root, target_size=32, template_size=128):
        self.target_size = target_size
        self.template_size = template_size
        
        input_files = sorted(glob.glob(os.path.join(data_root, "example_input_*.npy")))
        target_files = sorted(glob.glob(os.path.join(data_root, "example_target_*.npy")))
        
        assert len(input_files) == len(target_files), f"条件图像对数量不匹配"
        assert len(input_files) > 0, f"未找到条件图像文件"
        
        self.data_pairs = list(zip(input_files, target_files))
        print(f"加载了 {len(self.data_pairs)} 个条件图像对（input-target）")

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        input_path, target_path = self.data_pairs[idx]
        
        try:
            # 加载input（条件图）和target（真实标签图）
            template = np.load(input_path).squeeze().astype(np.float32)  # input: 128x128
            target = np.load(target_path).squeeze().astype(np.float32)   # target: 32x32
            
            # 归一化处理
            if template.max() > 1.5:
                template = template / 255.0
            if target.max() > 1.5:
                target = target / 255.0
            
            # 归一化到[-1,1]
            template = template * 2.0 - 1.0
            target = target * 2.0 - 1.0
            
            # 转为tensor并添加通道维度
            input_tensor = torch.from_numpy(template).float().unsqueeze(0)    # (1,128,128)
            target_tensor = torch.from_numpy(target).float().unsqueeze(0)     # (1,32,32)
            
            # 返回：input、target、原始文件路径（用于日志）
            return input_tensor, target_tensor, input_path, target_path
            
        except Exception as e:
            print(f"加载图像出错（input: {input_path}, target: {target_path}）: {e}")
            return (torch.zeros(1, self.template_size, self.template_size), 
                   torch.zeros(1, self.target_size, self.target_size), 
                   input_path, target_path)

# ===================== 核心模型（与训练文件一致） =====================
class DDPMScoreUNet(nn.Module):
    def __init__(self, target_size=32, base_ch=64, time_embed_dim=128):
        super().__init__()
        self.target_size = target_size
        self.time_embed_dim = time_embed_dim
        self.base_ch = base_ch
        
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )
        
        # 编码器
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, base_ch, 3, padding=1),
            nn.GroupNorm(8, base_ch),
            nn.SiLU()
        )
        self.enc1_res = nn.Sequential(
            nn.Conv2d(base_ch, base_ch, 3, padding=1),
            nn.GroupNorm(8, base_ch),
            nn.SiLU(),
            nn.Conv2d(base_ch, base_ch, 3, padding=1),
            nn.GroupNorm(8, base_ch),
            nn.SiLU()
        )
        
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
        
        # 中间层
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
        
        # 解码器
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(base_ch*4, base_ch*2, 4, stride=2, padding=1),
            nn.GroupNorm(8, base_ch*2),
            nn.SiLU()
        )
        self.dec3_res = nn.Sequential(
            nn.Conv2d(base_ch*2, base_ch*2, 3, padding=1),
            nn.GroupNorm(8, base_ch*2),
            nn.SiLU(),
            nn.Conv2d(base_ch*2, base_ch*2, 3, padding=1),
            nn.GroupNorm(8, base_ch*2),
            nn.SiLU()
        )
        
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
        
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_ch, base_ch, 3, padding=1),
            nn.GroupNorm(8, base_ch),
            nn.SiLU(),
            nn.Conv2d(base_ch, 1, 3, padding=1)
        )
        
        # 时间嵌入融合层
        self.time_fuse1 = nn.Conv2d(time_embed_dim//4, base_ch, 1, padding=0)
        self.time_fuse2 = nn.Conv2d(time_embed_dim//4, base_ch*2, 1, padding=0)
        self.time_fuse3 = nn.Conv2d(time_embed_dim//4, base_ch*4, 1, padding=0)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def _expand_time_embed(self, time_embed, x):
        B, D = time_embed.shape
        H, W = x.shape[2], x.shape[3]
        embed1 = time_embed[:, :D//4].unsqueeze(-1).unsqueeze(-1).repeat(1, 1, H, W)
        embed2 = time_embed[:, D//4:D//2].unsqueeze(-1).unsqueeze(-1).repeat(1, 1, H//2, W//2)
        embed3 = time_embed[:, D//2:3*D//4].unsqueeze(-1).unsqueeze(-1).repeat(1, 1, H//4, W//4)
        return embed1, embed2, embed3
    
    def forward(self, x, t):
        time_embed = self.time_embed(t)
        embed1, embed2, embed3 = self._expand_time_embed(time_embed, x)
        
        # 编码器
        h1 = self.enc1(x)
        time_fused1 = self.time_fuse1(embed1)
        h1 = h1 + time_fused1
        h1 = self.enc1_res(h1) + h1
        
        h2 = self.enc2(h1)
        time_fused2 = self.time_fuse2(embed2)
        h2 = h2 + time_fused2
        h2 = self.enc2_res(h2) + h2
        
        h3 = self.enc3(h2)
        time_fused3 = self.time_fuse3(embed3)
        h3 = h3 + time_fused3
        h3 = self.enc3_res(h3) + h3
        
        # 中间层
        h_mid = self.mid(h3)
        h_mid = h_mid + time_fused3
        
        # 解码器
        h_dec3 = self.dec3(h_mid)
        h_dec3 = h_dec3 + h2
        h_dec3 = h_dec3 + time_fused2
        h_dec3 = self.dec3_res(h_dec3) + h_dec3
        
        h_dec2 = self.dec2(h_dec3)
        h_dec2 = h_dec2 + h1
        h_dec2 = h_dec2 + time_fused1
        h_dec2 = self.dec2_res(h_dec2) + h_dec2
        
        score = self.dec1(h_dec2)
        
        return score

# ===================== DDPM分数匹配核心模块（含DPS采样） =====================
class DDPMScoreMatching:
    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02, device='npu'):
        self.T = T
        self.device = device
        
        self.beta = torch.linspace(beta_start, beta_end, T, device=device, dtype=torch.float32)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - self.alpha_bar)
        
        self.sqrt_alpha = torch.sqrt(self.alpha)
        self.one_over_sqrt_alpha = 1.0 / self.sqrt_alpha
        self.beta_over_sqrt_one_minus_alpha_bar = self.beta / self.sqrt_one_minus_alpha_bar

    @torch.no_grad()
    def ddpm_sampling(self, model, batch_size):
        """标准DDPM无条件采样"""
        x_t = torch.randn((batch_size, 1, 32, 32), device=self.device, dtype=torch.float32)
        
        pbar = tqdm(range(self.T-1, -1, -1), desc="DDPM Unconditional Sampling", leave=False)
        for t in pbar:
            t_tensor = torch.tensor([t], device=self.device, dtype=torch.long).repeat(batch_size, 1).float()
            score_pred = model(x_t, t_tensor)
            
            sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bar[t].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            eps_pred = -score_pred * sqrt_one_minus_alpha_bar_t
            
            x_t = self.one_over_sqrt_alpha[t] * (x_t - self.beta_over_sqrt_one_minus_alpha_bar[t] * eps_pred)
            
            if t > 0:
                sigma_t = torch.sqrt(self.beta[t])
                z = torch.randn_like(x_t)
                x_t = x_t + sigma_t * z
            
            x_t = torch.clamp(x_t, -1.5, 1.5)
            pbar.set_postfix({"step": t})
        
        return x_t

    @torch.no_grad()
    def dps_sampling(self, model, batch_size, cond_data_root):
        """DPS条件采样（返回：生成结果 + input条件图 + target真实图 + 原始文件路径）"""
        # 加载条件数据集（获取input、target和文件路径）
        cond_dataset = MultiScaleDataset(data_root=cond_data_root, target_size=32, template_size=128)
        # 确保采样的条件图数量 >= batch_size
        if len(cond_dataset) < batch_size:
            print(f"警告：条件图像对数量（{len(cond_dataset)}）不足 batch_size（{batch_size}），使用全部条件图")
            batch_size = len(cond_dataset)
        
        cond_dataloader = DataLoader(cond_dataset, batch_size=batch_size, shuffle=True)
        batch_data = next(iter(cond_dataloader))
        input_tensor = batch_data[0].to(self.device)    # input: 128x128（原始条件图）
        target_tensor = batch_data[1].to(self.device)   # target: 32x32（真实标签图）
        input_paths = batch_data[2]                     # input文件路径
        target_paths = batch_data[3]                    # target文件路径
        
        # 下采样input到32x32（与生成目标尺寸一致，用于条件约束）
        def stride_sampling_downsample_4x(y):
            return y[:,:,::4, ::4]
        input_downsampled = stride_sampling_downsample_4x(input_tensor)
        
        # 初始化噪声
        x_t = torch.randn((batch_size, 1, 32, 32), device=self.device, dtype=torch.float32)
        
        pbar = tqdm(range(self.T-1, -1, -1), desc="DPS Conditional Sampling", leave=False)
        for t in pbar:
            t_tensor = torch.tensor([t], device=self.device, dtype=torch.long).repeat(batch_size, 1).float()
            score_pred = model(x_t, t_tensor)
            
            sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bar[t].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            eps_pred = -score_pred * sqrt_one_minus_alpha_bar_t
            
            # DDPM核心更新
            x_t = self.one_over_sqrt_alpha[t] * (x_t - self.beta_over_sqrt_one_minus_alpha_bar[t] * eps_pred)
            
            # 条件约束（引导生成结果向input下采样图靠近）
            x0_hat = (x_t - self.sqrt_one_minus_alpha_bar[t] * eps_pred) / self.sqrt_alpha_bar[t]
            x_t = x_t - 0.1/torch.norm(input_downsampled-x0_hat,2)*(input_downsampled-x0_hat)*(1/self.sqrt_alpha_bar[t])
            
            # 添加反向噪声
            if t > 0:
                sigma_t = torch.sqrt(self.beta[t])
                z = torch.randn_like(x_t)
                x_t = x_t + sigma_t * z
            
            x_t = torch.clamp(x_t, -1.5, 1.5)
            pbar.set_postfix({"step": t})
        
        # 返回：生成结果（32x32）、input（128x128）、target（32x32）、文件路径
        return x_t, input_tensor, target_tensor, input_paths, target_paths

# ===================== 主采样函数 =====================
def main():
    parser = argparse.ArgumentParser(description="DDPM Score Matching - DPS Sampling (Input/Target/Generated)")
    parser.add_argument("--model_path", type=str, required=True, help="预训练模型路径（必须）")
    parser.add_argument("--num_samples", type=int, default=16, help="生成样本数量（DPS模式下不能超过条件图对数量）")
    parser.add_argument("--output_dir", type=str, default="./dps_sample_results", help="输出根目录")
    parser.add_argument("--device", type=str, default='npu', help="设备（npu/cpu）")
    parser.add_argument("--cond_data_root", type=str, default="../generated_data/", help="DPS采样的条件图像根目录")
    
    args = parser.parse_args()
    
    # 设备配置
    device = args.device if torch.npu.is_available() else 'cpu'
    print(f"使用设备: {device}")
    print(f"采样配置：样本数={args.num_samples}，输出目录={args.output_dir}")
    
    # 创建三级输出目录（明确区分三类图像）
    output_root = args.output_dir
    input_dir = os.path.join(output_root, "input_images")    # 原始输入条件图（128x128）
    target_dir = os.path.join(output_root, "target_images")  # 真实标签图（32x32）
    gen_dir = os.path.join(output_root, "generated_images")  # 模型生成结果图（32x32）
    
    os.makedirs(output_root, exist_ok=True)
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(gen_dir, exist_ok=True)
    
    # 加载模型和训练参数（增加容错）
    print(f"加载预训练模型: {args.model_path}")
    try:
        checkpoint = torch.load(args.model_path, map_location=device)
    except Exception as e:
        print(f"模型加载失败！错误：{e}")
        print("请检查模型路径是否正确，或模型文件是否损坏")
        exit(1)
    
    model_args = checkpoint.get('args', None)
    trained_epoch = checkpoint.get('epoch', -1)
    if model_args is None:
        print("警告：模型文件中未找到训练参数，使用默认配置")
        class DefaultArgs:
            def __init__(self):
                self.target_size = 32
                self.T = 1000
                self.beta_start = 1e-4
                self.beta_end = 0.02
        model_args = DefaultArgs()
    
    # 初始化模型
    model = DDPMScoreUNet(model_args.target_size).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"模型加载完成（训练epoch: {trained_epoch}，目标尺寸: {model_args.target_size}x{model_args.target_size}）")
    
    # 初始化DDPM模块（使用训练时的参数）
    ddpm_score_matching = DDPMScoreMatching(
        T=model_args.T,
        beta_start=model_args.beta_start,
        beta_end=model_args.beta_end,
        device=device
    )
    
    # 执行DPS采样（只保留DPS模式，专注input/target/generated分离）
    print(f"开始DPS条件采样...")
    samples_gen, samples_input, samples_target, input_paths, target_paths = ddpm_score_matching.dps_sampling(
        model,
        batch_size=args.num_samples,
        cond_data_root=args.cond_data_root
    )
    actual_num = len(samples_gen)  # 实际采样数量（可能小于指定的num_samples）
    print(f"DPS采样完成：{actual_num} 个样本对（input-target-generated）")
    
    # 单张保存（按1.png、2.png...序号排列，索引一一对应）
    print("开始保存图像文件...")
    for i in range(actual_num):
        # 序号从1开始（1.png、2.png...）
        seq_num = i + 1
        
        # 1. 保存input图像（原始128x128）
        input_filename = f"{seq_num}.png"
        input_path = os.path.join(input_dir, input_filename)
        save_image(samples_input[i], input_path)
        
        # 2. 保存target图像（真实32x32）
        target_filename = f"{seq_num}.png"
        target_path = os.path.join(target_dir, target_filename)
        save_image(samples_target[i], target_path)
        
        # 3. 保存generated图像（生成32x32）
        gen_filename = f"{seq_num}.png"
        gen_path = os.path.join(gen_dir, gen_filename)
        save_image(samples_gen[i], gen_path)
        
        # 打印对应关系日志（便于追溯）
        original_input = os.path.basename(input_paths[i])
        original_target = os.path.basename(target_paths[i])
        print(f"样本 {seq_num}：")
        print(f"  - Input: {original_input} -> input_images/{input_filename}（128x128）")
        print(f"  - Target: {original_target} -> target_images/{target_filename}（32x32）")
        print(f"  - Generated: generated_images/{gen_filename}（32x32）")
        print("-" * 60)
    
    # 输出总结（明确目录结构）
    print(f"\n采样完成！文件存储结构：")
    print(f"输出根目录：{output_root}")
    print(f"├─ input_images/  # 原始输入条件图（128x128）")
    print(f"│  ├─ 1.png")
    print(f"│  ├─ 2.png")
    print(f"│  └─ ...（共{actual_num}张，按序号1~{actual_num}排列）")
    print(f"├─ target_images/  # 真实标签图（32x32）")
    print(f"│  ├─ 1.png")
    print(f"│  ├─ 2.png")
    print(f"│  └─ ...（共{actual_num}张，按序号1~{actual_num}排列）")
    print(f"└─ generated_images/  # 模型生成结果图（32x32）")
    print(f"   ├─ 1.png")
    print(f"   ├─ 2.png")
    print(f"   └─ ...（共{actual_num}张，按序号1~{actual_num}排列）")
    print(f"\n关键说明：")
    print(f"1. 三类图像的序号严格对应：input_images/1.png ↔ target_images/1.png ↔ generated_images/1.png")
    print(f"3. Input图像保留原始128x128尺寸，Target和Generated为32x32尺寸")


if __name__ == "__main__":
    main()