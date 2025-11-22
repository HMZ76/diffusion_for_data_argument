import os
import glob
import math
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import argparse
import torch_npu
from tqdm import tqdm

# ===================== 与训练代码完全一致的基础组件 =====================
# 设置随机种子（确保采样可复现）
def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.npu.is_available():
        torch.npu.manual_seed_all(seed)
    torch.set_default_dtype(torch.float32)

set_seed()

# 数据预处理（与训练代码完全一致）
def normalize_img(x):
    x = np.array(x, dtype=np.float32)
    if x.max() > 1.5:  # 0~255范围
        x = x / 127.5 - 1.0
    else:  # 0~1范围
        x = x * 2.0 - 1.0
    return x

def denormalize_img(x):
    if isinstance(x, np.ndarray):
        return np.clip(((x + 1) * 127.5), 0, 255).astype(np.uint8)
    return ((x + 1) * 127.5).clamp(0, 255).to(torch.uint8)

def save_image(tensor, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = denormalize_img(tensor.squeeze(0).cpu())
    Image.fromarray(img.numpy()).save(path)

# 数据集（仅用于加载input和target，不做训练相关增强）
class MultiScaleDataset(Dataset):
    def __init__(self, data_root, target_size=32, template_size=128):
        self.target_size = target_size
        self.template_size = template_size
        
        input_files = sorted(glob.glob(os.path.join(data_root, "example_input_*.npy")))
        target_files = sorted(glob.glob(os.path.join(data_root, "example_target_*.npy")))
        
        assert len(input_files) == len(target_files), f"input和target文件数量不匹配（input: {len(input_files)}, target: {len(target_files)}）"
        assert len(input_files) > 0, f"在 {data_root} 未找到example_input_*.npy或example_target_*.npy文件"
        
        self.data_pairs = list(zip(input_files, target_files))
        print(f"加载了 {len(self.data_pairs)} 个数据对（input-target）")
        print(f"Input尺寸: {template_size}x{template_size}, Target尺寸: {target_size}x{target_size}")

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        input_path, target_path = self.data_pairs[idx]
        
        try:
            # 加载原始数据
            template = np.load(input_path).squeeze().astype(np.float32)  # input: 128x128
            target = np.load(target_path).squeeze().astype(np.float32)   # target: 32x32
            
            # 归一化（与训练代码一致）
            if template.max() > 1.5:
                template = template / 255.0
            if target.max() > 1.5:
                target = target / 255.0
            
            input_norm = normalize_img(template)
            target_norm = normalize_img(target)
            
            # 转为tensor并添加通道维度
            input_tensor = torch.from_numpy(input_norm).float().unsqueeze(0)  # (1,128,128)
            target_tensor = torch.from_numpy(target_norm).float().unsqueeze(0) # (1,32,32)
            
            # 返回：归一化tensor + 原始文件路径（用于日志追溯）
            return input_tensor, target_tensor, input_path, target_path
            
        except Exception as e:
            print(f"加载数据出错（input: {input_path}, target: {target_path}）: {e}")
            return (torch.zeros(1, self.template_size, self.template_size), 
                   torch.zeros(1, self.target_size, self.target_size), 
                   input_path, target_path)

# 时间嵌入（与训练代码一致）
class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=time.device, dtype=torch.float32) * -emb)
        emb = time.float()[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb

# 条件编码器（与训练代码一致）
class ConditionEncoder(nn.Module):
    def __init__(self, in_ch=1, base_ch=64):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 3, padding=1),
            nn.GroupNorm(8, base_ch),
            nn.SiLU()
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(base_ch, base_ch*2, 3, stride=2, padding=1),
            nn.GroupNorm(8, base_ch*2),
            nn.SiLU()
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(base_ch*2, base_ch*4, 3, stride=2, padding=1),
            nn.GroupNorm(8, base_ch*4),
            nn.SiLU()
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(base_ch*4, base_ch*4, 3, stride=2, padding=1),
            nn.GroupNorm(8, base_ch*4),
            nn.SiLU()
        )
        
        self.conv5 = nn.Sequential(
            nn.Conv2d(base_ch*4, base_ch*8, 3, stride=2, padding=1),
            nn.GroupNorm(8, base_ch*8),
            nn.SiLU()
        )

    def forward(self, cond):
        h1 = self.conv1(cond)
        h2 = self.conv2(h1)
        h3 = self.conv3(h2)
        h4 = self.conv4(h3)
        h5 = self.conv5(h4)
        
        return {
            '128': h1,
            '64': h2, 
            '32': h3,
            '16': h4,
            '8': h5
        }

# 多尺度UNet（与训练代码一致）
class MultiScaleUNet(nn.Module):
    def __init__(self, target_size=32):
        super().__init__()
        
        self.condition_encoder = ConditionEncoder()
        
        self.time_embed = nn.Sequential(
            TimeEmbedding(64),
            nn.Linear(64, 256),
            nn.SiLU(),
            nn.Linear(256, 256)
        )
        
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU()
        )
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(64 + 256, 128, 3, padding=1, stride=2),
            nn.GroupNorm(8, 128),
            nn.SiLU()
        )
        
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1, stride=2),
            nn.GroupNorm(8, 256),
            nn.SiLU()
        )
        
        self.mid = nn.Sequential(
            nn.Conv2d(256 + 512, 512, 3, padding=1),
            nn.GroupNorm(8, 512),
            nn.SiLU(),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.GroupNorm(8, 256),
            nn.SiLU()
        )
        
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU()
        )
        
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, 4, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU()
        )
        
        self.dec1 = nn.Sequential(
            nn.Conv2d(128 + 256, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 1, 3, padding=1)
        )
        
        self.time_proj1 = nn.Linear(256, 64)
        self.time_proj2 = nn.Linear(256, 128)
        self.time_proj3 = nn.Linear(256, 256)

    def forward(self, x, t, cond):
        x = x.float()
        cond = cond.float()
        t = t.long()
        
        cond_features = self.condition_encoder(cond)
        t_emb = self.time_embed(t)
        
        h1 = self.enc1(x)
        h1 = h1 + self.time_proj1(t_emb)[:, :, None, None]
        
        h1_with_cond = torch.cat([h1, cond_features['32']], dim=1)
        
        h2 = self.enc2(h1_with_cond)
        h2 = h2 + self.time_proj2(t_emb)[:, :, None, None]
        
        h3 = self.enc3(h2)
        h3 = h3 + self.time_proj3(t_emb)[:, :, None, None]
        
        h3_with_cond = torch.cat([h3, cond_features['8']], dim=1)
        h = self.mid(h3_with_cond)
        
        h = self.dec3(h)
        h = torch.cat([h, h2], dim=1)
        
        h = self.dec2(h)
        h_with_skip_and_cond = torch.cat([h, h1, cond_features['32']], dim=1)
        
        h = self.dec1(h_with_skip_and_cond)
        
        return h

# 扩散过程（与训练代码一致，仅保留采样逻辑）
class SimpleDiffusion:
    def __init__(self, timesteps=1000, device='npu'):
        self.timesteps = timesteps
        self.device = device
        
        betas = torch.linspace(1e-4, 2e-2, timesteps, dtype=torch.float32, device=device)
        
        self.betas = betas
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    @torch.no_grad()
    def sample(self, model, cond, shape, steps=50, cfg_scale=3.0):
        """普通条件采样（与训练代码生成逻辑一致）"""
        b = shape[0]
        x = torch.randn(shape, device=self.device, dtype=torch.float32)
        
        cond_null = torch.zeros_like(cond)  # CFG空条件
        
        step_size = self.timesteps // steps
        
        # 逆扩散过程（带进度条）
        pbar = tqdm(range(0, self.timesteps, step_size), desc="Sampling", total=steps)
        for i in reversed(range(0, self.timesteps, step_size)):
            t = torch.full((b,), i, device=self.device, dtype=torch.long)
            
            # CFG引导（与训练一致的条件生成逻辑）
            eps_cond = model(x, t, cond)
            eps_uncond = model(x, t, cond_null)
            eps = eps_uncond + cfg_scale * (eps_cond - eps_uncond)
            
            alpha = self.alphas[i]
            alpha_cumprod = self.alphas_cumprod[i]
            beta = self.betas[i]
            
            # 预测x0并更新
            pred_x0 = (x - torch.sqrt(1 - alpha_cumprod) * eps) / torch.sqrt(alpha_cumprod)
            pred_x0 = pred_x0.clamp(-1, 1)
            
            if i > 0:
                noise = torch.randn_like(x)
                x = torch.sqrt(alpha) * pred_x0 + torch.sqrt(beta) * noise
            else:
                x = pred_x0
            
            pbar.update(1)
        pbar.close()
        
        return x

# ===================== 采样主逻辑 =====================
def main():
    parser = argparse.ArgumentParser(description="普通条件采样脚本 - 生成input/target/generated三类图像")
    parser.add_argument("--model_path", type=str, required=True, help="预训练模型路径（例如：./checkpoints/multiscale_model_epoch_400.pt）")
    parser.add_argument("--data_root", type=str, default="../generated_data/", help="数据根目录（包含example_input_*.npy和example_target_*.npy）")
    parser.add_argument("--output_root", type=str, default="./sampling_results", help="输出根目录")
    parser.add_argument("--num_samples", type=int, default=1000, help="采样数量（-1表示使用全部数据）")
    parser.add_argument("--batch_size", type=int, default=32, help="采样batch_size（根据NPU内存调整）")
    parser.add_argument("--cfg_scale", type=float, default=3.0, help="CFG引导强度（越大越贴合条件，建议1.0-5.0）")
    parser.add_argument("--sample_steps", type=int, default=50, help="采样步数（越大越精细，建议30-100）")
    parser.add_argument("--target_size", type=int, default=32, help="生成目标尺寸（与训练一致）")
    parser.add_argument("--template_size", type=int, default=128, help="条件图像尺寸（与训练一致）")
    parser.add_argument("--device", type=str, default='npu', help="设备（npu/cpu）")
    
    args = parser.parse_args()
    
    # 设备配置
    device = args.device if torch.npu.is_available() else 'cpu'
    print(f"=== 采样配置 ===")
    print(f"设备: {device}")
    print(f"模型路径: {args.model_path}")
    print(f"数据根目录: {args.data_root}")
    print(f"输出根目录: {args.output_root}")
    print(f"采样数量: {'全部数据' if args.num_samples == -1 else args.num_samples}")
    print(f"batch_size: {args.batch_size}, CFG强度: {args.cfg_scale}, 采样步数: {args.sample_steps}")
    
    # 创建输出目录（三类图像分开存储，序号一一对应）
    input_dir = os.path.join(args.output_root, "input_images")    # 原始条件图（128x128）
    target_dir = os.path.join(args.output_root, "target_images")  # 真实标签图（32x32）
    gen_dir = os.path.join(args.output_root, "generated_images")  # 模型生成图（32x32）
    
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(gen_dir, exist_ok=True)
    
    # 加载数据集
    dataset = MultiScaleDataset(
        data_root=args.data_root,
        target_size=args.target_size,
        template_size=args.template_size
    )
    
    # 限制采样数量
    if args.num_samples != -1 and args.num_samples < len(dataset):
        dataset.data_pairs = dataset.data_pairs[:args.num_samples]
        print(f"限制采样数量为: {args.num_samples}")
    
    # 数据加载器（不shuffle，保持序号稳定）
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # 初始化模型
    model = MultiScaleUNet(target_size=args.target_size).to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 加载预训练权重
    try:
        checkpoint = torch.load(args.model_path, map_location=device)
        # 兼容两种保存格式（仅保存state_dict或包含其他信息）
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"模型加载成功！")
    except Exception as e:
        print(f"模型加载失败: {e}")
        exit(1)
    
    # 模型设为eval模式
    model.eval()
    
    # 初始化扩散器
    diffusion = SimpleDiffusion(timesteps=1000, device=device)
    
    # 开始采样（按batch处理，序号连续）
    total_samples = 0
    print(f"\n=== 开始采样 ===")
    for batch_idx, (input_tensor, target_tensor, input_paths, target_paths) in enumerate(dataloader):
        # 数据移到设备
        input_tensor = input_tensor.to(device)  # (B,1,128,128)
        target_tensor = target_tensor.to(device)  # (B,1,32,32)
        
        # 生成样本
        generated_tensor = diffusion.sample(
            model=model,
            cond=input_tensor,
            shape=(input_tensor.shape[0], 1, args.target_size, args.target_size),
            steps=args.sample_steps,
            cfg_scale=args.cfg_scale
        )
        
        # 保存当前batch的图像（序号从1开始，连续递增）
        for idx_in_batch in range(len(generated_tensor)):
            total_samples += 1
            seq_num = total_samples  # 序号：1,2,3...
            
            # 1. 保存input图像（128x128）
            input_path = os.path.join(input_dir, f"{seq_num}.png")
            save_image(input_tensor[idx_in_batch], input_path)
            
            # 2. 保存target图像（32x32）
            target_path = os.path.join(target_dir, f"{seq_num}.png")
            save_image(target_tensor[idx_in_batch], target_path)
            
            # 3. 保存generated图像（32x32）
            gen_path = os.path.join(gen_dir, f"{seq_num}.png")
            save_image(generated_tensor[idx_in_batch], gen_path)
            
            # 打印日志（追溯原始文件）
            original_input = os.path.basename(input_paths[idx_in_batch])
            original_target = os.path.basename(target_paths[idx_in_batch])
            print(f"样本 {seq_num}:")
            print(f"  - Input: {original_input} -> {input_path}")
            print(f"  - Target: {original_target} -> {target_path}")
            print(f"  - Generated: {gen_path}")
            print("-" * 50)
    
    # 采样完成总结
    print(f"\n=== 采样完成 ===")
    print(f"总采样数量: {total_samples}")
    print(f"文件存储结构:")
    print(f"{args.output_root}/")
    print(f"├─ input_images/  # 原始条件图（128x128）: 1.png, 2.png, ...")
    print(f"├─ target_images/  # 真实标签图（32x32）: 1.png, 2.png, ...")
    print(f"└─ generated_images/  # 模型生成图（32x32）: 1.png, 2.png, ...")
    print(f"关键说明: 三类图像的序号严格对应，便于对比评估")

if __name__ == "__main__":
    main()