import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from PIL import Image
from pytorch_fid import fid_score
from lpips import LPIPS
from cfg_mult_npu import MultiScaleDataset, MultiScaleUNet, SimpleDiffusion
import torch_npu
import tqdm
from datetime import datetime

# ---------------------- 配置参数 ----------------------
TEST_DATA_ROOT = "./generated_data"
CHECKPOINT_PATH = "./checkpoints/multiscale_model_epoch_4.pt"
SAVE_DIR = "./conditional_fid_results"
GENERATED_ROOT = os.path.join(SAVE_DIR, "generated")  # 全局保存生成图像
TARGET_ROOT = os.path.join(SAVE_DIR, "target")        # 全局保存真实图像
TARGET_SIZE = 32
TEMPLATE_SIZE = 128
BATCH_SIZE = 1000  # 建议根据设备内存调整
DIFFUSION_STEPS = 50
DEVICE = 'npu' if torch.npu.is_available() else 'cpu'
MAX_SAMPLES = 20000  # 仅使用前2000个样本


def save_images(images, save_root, batch_idx):
    """直接保存图像到根目录（不分组）"""
    os.makedirs(save_root, exist_ok=True)
    for i, img in enumerate(images):
        img_np = img.squeeze().cpu().detach().numpy()
        # 归一化到[0,255]并转为uint8
        img_np = ((img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8) * 255).astype(np.uint8)
        Image.fromarray(img_np).save(os.path.join(save_root, f"batch_{batch_idx}_img_{i}.png"))


if __name__ == "__main__":
    # 创建保存目录
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(GENERATED_ROOT, exist_ok=True)
    os.makedirs(TARGET_ROOT, exist_ok=True)

    # 加载测试数据集（仅前MAX_SAMPLES个样本）
    full_dataset = MultiScaleDataset(TEST_DATA_ROOT, TEMPLATE_SIZE, TARGET_SIZE)
    sample_count = min(MAX_SAMPLES, len(full_dataset))
    test_dataset = Subset(full_dataset, range(sample_count))
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True if DEVICE == 'npu' else False
    )
    print(f"测试集规模: {len(test_dataset)} 样本（仅使用前{sample_count}个）")
    print(f"使用设备: {DEVICE}")

    # 初始化模型和扩散器
    model = MultiScaleUNet(TARGET_SIZE).to(DEVICE)
    diffusion = SimpleDiffusion(device=DEVICE)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint)
    model.eval()
    print(f"已加载模型权重: {CHECKPOINT_PATH}")

    # 初始化LPIPS
    lpips_calculator = LPIPS(net='alex').to(DEVICE)
    lpips_calculator.eval()
    total_lpips = 0.0
    total_batches = 0

    # 生成并保存图像（不分组，直接保存到根目录）
    with torch.no_grad():
        for batch_idx, (cond, target) in enumerate(tqdm.tqdm(test_loader, desc="生成与保存")):
            cond = cond.to(DEVICE, dtype=torch.float32)  # [B, 1, 128, 128]
            target = target.to(DEVICE, dtype=torch.float32)  # [B, 1, 32, 32]

            # 生成图像
            sampled = diffusion.sample(
                model,
                cond,
                shape=(cond.size(0), 1, TARGET_SIZE, TARGET_SIZE),
                steps=DIFFUSION_STEPS
            )

            # 保存生成图像和真实图像（不分组）
            save_images(sampled, GENERATED_ROOT, batch_idx)
            save_images(target, TARGET_ROOT, batch_idx)

            # 计算LPIPS（需转为3通道输入）
            sampled_rgb = sampled.repeat(1, 3, 1, 1)  # [B, 3, H, W]
            target_rgb = target.repeat(1, 3, 1, 1)
            lpips_score = lpips_calculator(sampled_rgb, target_rgb).mean().item()
            total_lpips += lpips_score
            total_batches += 1

        # 计算全局FID
        print("\n开始计算全局FID...")
        try:
            # 统计有效图像数量
            gen_files = [f for f in os.listdir(GENERATED_ROOT) if f.endswith(('.png', '.jpg', '.jpeg'))]
            tgt_files = [f for f in os.listdir(TARGET_ROOT) if f.endswith(('.png', '.jpg', '.jpeg'))]
            if len(gen_files) == 0 or len(tgt_files) == 0:
                raise ValueError("生成图像或真实图像目录为空")

            # 计算FID
            global_batch_size = min(BATCH_SIZE, len(gen_files), len(tgt_files), 50)  # 控制批次大小
            global_fid = fid_score.calculate_fid_given_paths(
                paths=[GENERATED_ROOT, TARGET_ROOT],
                batch_size=global_batch_size,
                device=DEVICE,
                dims=2048,
                num_workers=4
            )
        except Exception as e:
            print(f"全局FID计算失败: {e}")
            global_fid = -1

    # 保存结果
    result_path = os.path.join(SAVE_DIR, "evaluation_results.txt")
    with open(result_path, "w") as f:
        f.write(f"评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"模型路径: {CHECKPOINT_PATH}\n")
        f.write(f"测试样本数: {len(test_dataset)}\n")
        f.write(f"全局FID: {global_fid:.4f}\n")
        f.write(f"平均LPIPS: {total_lpips/total_batches:.4f}\n")

    # 打印结果
    print("\n===== 评估结果 =====")
    print(f"全局FID: {global_fid:.4f}")
    print(f"平均LPIPS: {total_lpips/total_batches:.4f}")
    print(f"结果已保存至: {result_path}")