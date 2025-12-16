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
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子和默认数据类型
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_default_dtype(torch.float32)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

# ===================== 数据预处理工具函数 =====================
def normalize_img(x):
    x = np.array(x, dtype=np.float32) if not isinstance(x, np.ndarray) else x.astype(np.float32)
    if x.max() > 1.5:
        x = x / 127.5 - 1.0
    else:
        x = x * 2.0 - 1.0
    return x

def denormalize_img(x):
    if isinstance(x, np.ndarray):
        return np.clip(((x + 1) * 127.5), 0, 255).astype(np.uint8)
    return ((x + 1) * 127.5).clamp(0, 255).to(torch.uint8)

def save_image(tensor, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = denormalize_img(tensor.squeeze(0).cpu())
    Image.fromarray(img.numpy() if isinstance(img, torch.Tensor) else img).save(path)

def create_sample_grid(samples, epoch, save_path, n_cols=4):
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
        self.target_size = target_size
        self.image_files = sorted(glob.glob(os.path.join(data_root, "*.npy")))
        self.image_files = [f for f in self.image_files if "target" in f.lower() or "image" in f.lower()]
        
        assert len(self.image_files) > 0, f"未找到数据文件！在 {data_root} 中寻找 .npy 文件"
        
        print(f"加载了 {len(self.image_files)} 个图像文件")
        print(f"目标图像尺寸: {target_size}x{target_size}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        
        try:
            img = np.load(image_path).squeeze().astype(np.float32)
            img = normalize_img(img)
            
            if len(img.shape) == 2:
                img = img[np.newaxis, ...]
            
            if img.shape[1] != self.target_size or img.shape[2] != self.target_size:
                img = torch.from_numpy(img).unsqueeze(0)
                img = F.interpolate(img, size=(self.target_size, self.target_size), mode='bilinear', align_corners=False)
                img = img.squeeze(0).numpy()
            
            return torch.from_numpy(img).float()
            
        except Exception as e:
            print(f"加载数据出错 {image_path}: {e}")
            return torch.zeros(1, self.target_size, self.target_size).float()

# ===================== DDPM分数匹配UNet =====================
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

# ===================== DDPM分数匹配核心模块 =====================
class DDPMScoreMatching:
    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02, device='cuda'):
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
        
        print(f"DDPM配置：总扩散步数T={T}，beta范围[{beta_start:.4f}, {beta_end:.4f}]")

    def forward_diffusion(self, x_clean, t):
        eps = torch.randn_like(x_clean, device=self.device)
        t_long = t.long().squeeze(1)
        sqrt_alpha_bar_t = self.sqrt_alpha_bar[t_long].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bar[t_long].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        
        x_t = sqrt_alpha_bar_t * x_clean + sqrt_one_minus_alpha_bar_t * eps
        
        return x_t, eps

    def score_matching_loss(self, model, x_clean):
        B = x_clean.shape[0]
        t = torch.randint(0, self.T, (B, 1), device=self.device, dtype=torch.long)
        x_t, eps_true = self.forward_diffusion(x_clean, t)
        
        score_pred = model(x_t, t.float())
        
        t_long = t.long().squeeze(1)
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bar[t_long].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        eps_pred = -score_pred * sqrt_one_minus_alpha_bar_t
        
        loss = F.mse_loss(eps_pred, eps_true)
        reg_loss = 1e-6 * torch.norm(score_pred, p=2)
        total_loss = loss + reg_loss
        
        return total_loss

    @torch.no_grad()
    def ddpm_sampling(self, model, batch_size):
        x_t = torch.randn((batch_size, 1, 32, 32), device=self.device, dtype=torch.float32)
        
        pbar = tqdm(range(self.T-1, -1, -1), desc="DDPM Sampling", leave=False)
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

# ===================== 主训练函数 =====================
def main():
    parser = argparse.ArgumentParser(description="DDPM Style Unconditional Score Matching - Training")
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
    parser.add_argument("--device", type=str, default='cuda', help="设备（cuda/cpu）")
    parser.add_argument("--resume", type=str, default="", help="恢复训练的模型路径")
    
    args = parser.parse_args()
    
    # 设备配置
    device = args.device if torch.cuda.is_available() else 'cpu'
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
        num_workers=4 if torch.cuda.is_available() else 0,
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
            x_clean = x_clean.to(device, dtype=torch.float32)
            loss = ddpm_score_matching.score_matching_loss(model, x_clean)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            avg_loss = total_loss / (step + 1)
            pbar.set_postfix({"loss": f"{loss.item():.6f}", "avg_loss": f"{avg_loss:.6f}"})
            
            if step % args.log_interval == 0:
                global_step = epoch * len(dataloader) + step
                writer.add_scalar("train/loss", loss.item(), global_step)
                writer.add_scalar("train/avg_loss", avg_loss, global_step)
                writer.add_scalar("train/lr", optimizer.param_groups[0]['lr'], global_step)
        
        scheduler.step()
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
        
        # 生成训练过程中的样本（用于监控）
        if epoch % args.save_interval == 0:
            model.eval()
            with torch.no_grad():
                print("开始DDPM采样...")
                samples = ddpm_score_matching.ddpm_sampling(
                    model,
                    batch_size=args.num_samples
                )
                
                grid_path = os.path.join(sample_dir, f"epoch_{epoch}_ddpm_samples.png")
                create_sample_grid(samples, epoch, grid_path, n_cols=4)
                
                for i in range(min(5, args.num_samples)):
                    single_path = os.path.join(sample_dir, f"epoch_{epoch}_sample_{i}.png")
                    save_image(samples[i], single_path)
                
                grid_img = torchvision.utils.make_grid(samples, nrow=4, normalize=True, scale_each=True)
                writer.add_image("ddpm_generated_samples", grid_img, epoch)
                print(f"DDPM样本已保存到 {sample_dir}")
    
    writer.close()
    print("训练完成！")
    print(f"最佳模型保存路径: {os.path.join(checkpoint_dir, 'best_model.pt')}")
    print(f"训练过程样本保存路径: {sample_dir}")

if __name__ == "__main__":
    main()