import os
import math
import glob
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import argparse
import torch_npu
from torch.utils.tensorboard import SummaryWriter  # 导入TensorBoard
from datetime import datetime  # 用于生成时间戳

# 设置随机种子和默认数据类型
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.npu.is_available():
        torch.npu.manual_seed_all(seed)
    torch.set_default_dtype(torch.float32)

set_seed()

# 数据预处理函数（保持不变）
def normalize_img(x):
    x = np.array(x, dtype=np.float32)
    if x.max() > 1.5:  # 0~255范围
        x = x / 127.5 - 1.0
    else:  # 0~1范围
        x = x * 2.0 - 1.0
    return x

def denormalize_img(x):
    return ((x + 1) * 127.5).clamp(0, 255).to(torch.uint8)

def save_image(tensor, path):
    img = denormalize_img(tensor.squeeze(0).cpu())
    Image.fromarray(img.numpy()).save(path)

def create_single_comparison(condition, target, generated, save_path, epoch, idx):
    cond_img = denormalize_img(condition.squeeze(0).cpu()).numpy()
    target_img = denormalize_img(target.squeeze(0).cpu()).numpy()
    gen_img = denormalize_img(generated.squeeze(0).cpu()).numpy()
    
    size = 128
    cond_pil = Image.fromarray(cond_img).resize((size, size), Image.NEAREST)
    target_pil = Image.fromarray(target_img).resize((size, size), Image.NEAREST)
    gen_pil = Image.fromarray(gen_img).resize((size, size), Image.NEAREST)
    
    label_height = 30
    total_width = size * 3
    total_height = size + label_height
    comparison = Image.new('RGB', (total_width, total_height), color='white')
    
    comparison.paste(cond_pil, (0, label_height))
    comparison.paste(target_pil, (size, label_height))  
    comparison.paste(gen_pil, (size*2, label_height))
    
    draw = ImageDraw.Draw(comparison)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except:
        font = ImageFont.load_default()
    
    labels = ['Input (128x128)', 'Target (32x32)', 'Generated (32x32)']
    for i, label in enumerate(labels):
        x_pos = i * size + 10
        if font:
            draw.text((x_pos, 8), label, fill='black', font=font)
    
    comparison.save(save_path)
    return comparison

def create_comparison_grid(conditions, targets, generated, epoch, save_path):
    try:
        n_samples = len(conditions)
        fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4*n_samples))
        
        if n_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(n_samples):
            cond_np = denormalize_img(conditions[i].squeeze(0).cpu()).numpy()
            target_np = denormalize_img(targets[i].squeeze(0).cpu()).numpy()  
            gen_np = denormalize_img(generated[i].squeeze(0).cpu()).numpy()
            
            axes[i, 0].imshow(cond_np, cmap='gray')
            axes[i, 0].set_title(f'Input (128×128)', fontsize=12)
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(target_np, cmap='gray')
            axes[i, 1].set_title(f'Target (32×32)', fontsize=12)
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(gen_np, cmap='gray')
            axes[i, 2].set_title(f'Generated (32×32)', fontsize=12) 
            axes[i, 2].axis('off')
        
        plt.suptitle(f'Epoch {epoch} - Comparison Results', fontsize=16, y=0.98)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"创建网格失败: {e}")
        for i, (cond, target, gen) in enumerate(zip(conditions, targets, generated)):
            simple_path = save_path.replace('.png', f'_sample_{i}.png')
            create_single_comparison(cond, target, gen, simple_path, epoch, i)

def save_comparison(condition, target, generated, path, epoch, idx):
    try:
        comparison_path = path.replace('.png', f'_comparison.png')
        create_single_comparison(condition, target, generated, comparison_path, epoch, idx)
        save_image(condition, path.replace('.png', f'_input.png'))
        save_image(target, path.replace('.png', f'_target.png'))
        save_image(generated, path.replace('.png', f'_generated.png'))
        
    except Exception as e:
        print(f"保存对比图像出错: {e}")
        save_image(condition, path.replace('.png', f'_input.png'))
        save_image(target, path.replace('.png', f'_target.png'))
        save_image(generated, path.replace('.png', f'_generated.png'))

# 数据集（保持不变）
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

# 时间嵌入（保持不变）
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

# 条件编码器（保持不变）
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

# 多尺度UNet（保持不变）
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

# 扩散过程（修改设备相关配置）
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

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_losses(self, model, x_start, cond, t):
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)
        predicted_noise = model(x_noisy, t, cond)
        return F.mse_loss(predicted_noise, noise)

    @torch.no_grad()
    def sample(self, model, cond, shape, steps=50, cfg_scale=3.0):
        b = shape[0]
        x = torch.randn(shape, device=self.device, dtype=torch.float32)
        
        cond_null = torch.zeros_like(cond)
        
        step_size = self.timesteps // steps
        
        for i in reversed(range(0, self.timesteps, step_size)):
            t = torch.full((b,), i, device=self.device, dtype=torch.long)
            
            eps_cond = model(x, t, cond)
            eps_uncond = model(x, t, cond_null)
            eps = eps_uncond + cfg_scale * (eps_cond - eps_uncond)
            
            alpha = self.alphas[i]
            alpha_cumprod = self.alphas_cumprod[i]
            beta = self.betas[i]
            
            pred_x0 = (x - torch.sqrt(1 - alpha_cumprod) * eps) / torch.sqrt(alpha_cumprod)
            pred_x0 = pred_x0.clamp(-1, 1)
            
            if i > 0:
                noise = torch.randn_like(x)
                x = torch.sqrt(alpha) * pred_x0 + torch.sqrt(beta) * noise
            else:
                x = pred_x0
        
        return x

def main():
    # 配置（修改为单卡配置）
    data_root = "../generated_data/"
    target_size = 32
    template_size = 128
    batch_size = 1024  # 单卡batch_size（可根据GPU内存调整）
    epochs = 500
    lr = 2e-4  # 单卡学习率（无需缩放）
    device = 'npu' if torch.npu.is_available() else 'cpu'
    
    print(f"使用设备: {device}")
    print(f"多尺度训练：条件{template_size}x{template_size} -> 目标{target_size}x{target_size}")
    print(f"batch_size: {batch_size}")
    
    # ========== TensorBoard 配置 ==========
    # 创建时间戳，避免日志冲突
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"./tensorboard_logs/run_{timestamp}"
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard日志保存路径: {log_dir}")
    # ======================================
    
    # 数据加载（移除DistributedSampler，使用普通DataLoader）
    dataset = MultiScaleDataset(data_root, target_size, template_size)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,  # 单卡直接shuffle
        num_workers=4,
        pin_memory=True
    )
    
    # 模型初始化（无需DDP包装）
    model = MultiScaleUNet(target_size).to(device)
    
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # 创建保存目录
    os.makedirs("./checkpoints", exist_ok=True)
    os.makedirs("./samples", exist_ok=True)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 扩散器
    diffusion = SimpleDiffusion(device=device)
    
    # ========== 记录模型图（可选） ==========
    try:
        # 创建虚拟输入用于绘制模型图
        dummy_x = torch.randn(1, 1, target_size, target_size).to(device)
        dummy_t = torch.tensor([0]).to(device)
        dummy_cond = torch.randn(1, 1, template_size, template_size).to(device)
        writer.add_graph(model, (dummy_x, dummy_t, dummy_cond))
        print("模型图已保存到TensorBoard")
    except Exception as e:
        print(f"保存模型图失败: {e}")
    # ======================================
    
    # 训练循环
    global_step = 0  # 全局步数，用于TensorBoard记录
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for step, (cond, target) in enumerate(dataloader):
            try:
                cond = cond.to(device, dtype=torch.float32)
                target = target.to(device, dtype=torch.float32)
                
                # 随机时间步
                t = torch.randint(0, diffusion.timesteps, (target.size(0),), device=device)
                
                # 计算损失
                loss = diffusion.p_losses(model, target, cond, t)
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                # ========== 记录训练损失（每步） ==========
                writer.add_scalar('Train/Loss_Step', loss.item(), global_step)
                writer.add_scalar('Train/Learning_Rate', optimizer.param_groups[0]['lr'], global_step)
                global_step += 1
                # ==========================================
                
                # 打印日志
                if step % 50 == 0:
                    print(f"Epoch {epoch}, Step {step}, Loss: {loss.item():.4f}")
                    
            except Exception as e:
                print(f"训练步骤出错: {e}")
                continue
        
        # 计算平均损失
        avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
        print(f"Epoch {epoch} 完成, 平均损失: {avg_loss:.4f}")
        
        # ========== 记录epoch平均损失 ==========
        writer.add_scalar('Train/Avg_Loss_Epoch', avg_loss, epoch)
        # ======================================
        
        # 保存模型
        if epoch % 2 == 0:
            torch.save(model.state_dict(), f"./checkpoints/multiscale_model_epoch_{epoch}.pt")
        
        # 生成样本并记录到TensorBoard
        if epoch % 2 == 0:
            try:
                model.eval()
                with torch.no_grad():
                    # 取一批数据
                    batch_data = next(iter(dataloader))
                    cond_batch, target_batch = batch_data
                    cond_batch = cond_batch[:4].to(device)
                    target_batch = target_batch[:4].to(device)
                    
                    # 生成样本
                    samples = diffusion.sample(
                        model, 
                        cond_batch, 
                        shape=(4, 1, target_size, target_size),
                        steps=50
                    )
                    
                    # 保存对比图像
                    for i in range(min(4, samples.size(0))):
                        save_comparison(
                            condition=cond_batch[i], 
                            target=target_batch[i],
                            generated=samples[i],
                            path=f"./samples/epoch_{epoch}_sample_{i}.png",
                            epoch=epoch,
                            idx=i
                        )
                    
                    # 创建网格对比图
                    try:
                        grid_path = f"./samples/epoch_{epoch}_grid.png"
                        create_comparison_grid(
                            conditions=cond_batch,
                            targets=target_batch,
                            generated=samples,
                            epoch=epoch,
                            save_path=grid_path
                        )
                    except Exception as grid_error:
                        print(f"跳过网格创建: {grid_error}")
                
                # ========== 记录图像到TensorBoard ==========
                # 归一化图像到[0,1]范围用于TensorBoard显示
                cond_imgs = denormalize_img(cond_batch) / 255.0  # (4,1,128,128)
                target_imgs = denormalize_img(target_batch) / 255.0  # (4,1,32,32)
                gen_imgs = denormalize_img(samples) / 255.0  # (4,1,32,32)
                
                # 记录输入图像（128x128）
                writer.add_images('Images/Input_128x128', cond_imgs, epoch, dataformats='NCHW')
                
                # 记录目标图像（32x32）
                writer.add_images('Images/Target_32x32', target_imgs, epoch, dataformats='NCHW')
                
                # 记录生成图像（32x32）
                writer.add_images('Images/Generated_32x32', gen_imgs, epoch, dataformats='NCHW')
                
                # 创建对比网格并记录
                # 将32x32的目标和生成图上采样到128x128以便对比
                target_imgs_upscaled = F.interpolate(target_imgs, size=(128, 128), mode='nearest')
                gen_imgs_upscaled = F.interpolate(gen_imgs, size=(128, 128), mode='nearest')
                
                # 拼接成 [Input, Target, Generated] 的格式
                comparison_grid = torch.cat([cond_imgs, target_imgs_upscaled, gen_imgs_upscaled], dim=3)  # (4,1,128, 384)
                writer.add_images('Images/Comparison_Grid', comparison_grid, epoch, dataformats='NCHW')
                # ==========================================
                
                print(f"对比样本已保存到 ./samples/ 和 TensorBoard")
            except Exception as e:
                print(f"采样出错: {e}")
    
    # ========== 关闭TensorBoard写入器 ==========
    writer.close()
    print(f"TensorBoard日志已保存完成，可通过命令查看: tensorboard --logdir={log_dir}")
    # ==========================================
    
    print("训练完成!")

if __name__ == "__main__":
    main()