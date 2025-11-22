import os
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
import argparse
from lpips import LPIPS
from pytorch_fid.fid_score import calculate_fid_given_paths
import torch_npu

# ===================== 配置与工具函数 =====================
def set_seed(seed=42):
    """设置随机种子确保结果可复现"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.npu.is_available():
        torch.npu.manual_seed_all(seed)

set_seed()

def get_image_paths(folder, sort=True):
    """获取文件夹内所有PNG图像路径，按序号排序"""
    image_paths = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.png')]
    if sort:
        # 按文件名中的数字序号排序（支持1.png、0001.png等格式）
        def extract_num(filename):
            name = os.path.splitext(os.path.basename(filename))[0]
            return int(name) if name.isdigit() else 999999
        image_paths.sort(key=extract_num)
    return image_paths

def load_image_tensor(path, size=None, device='npu'):
    """加载图像并转为模型输入格式的tensor"""
    transform = transforms.Compose([
        transforms.Resize(size) if size else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # [-1,1] 归一化（LPIPS要求）
    ])
    image = Image.open(path).convert('RGB')  # 转为RGB（即使是灰度图也转为3通道）
    tensor = transform(image).unsqueeze(0).to(device)  # (1,3,H,W)
    return tensor

def load_image_for_fid(path, size=(299, 299), device='npu'):
    """加载图像用于FID计算（Inception v3要求输入299x299）"""
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet归一化
    ])
    image = Image.open(path).convert('RGB')
    tensor = transform(image).unsqueeze(0).to(device)  # (1,3,299,299)
    return tensor

# ===================== 核心评估函数 =====================
def calculate_lpips(gen_paths, target_paths, device='npu'):
    """
    计算LPIPS（感知相似度）
    LPIPS值越小，感知相似度越高（理想值≈0，最大值≈1）
    """
    # 初始化LPIPS模型（使用预训练的vgg网络）
    lpips_model = LPIPS(net='vgg', verbose=False).to(device)
    lpips_model.eval()
    
    lpips_scores = []
    print(f"\n=== 计算LPIPS（感知相似度）===")
    with torch.no_grad():
        for gen_path, target_path in tqdm(zip(gen_paths, target_paths), total=len(gen_paths)):
            # 加载图像（保持原始尺寸一致）
            gen_tensor = load_image_tensor(gen_path, device=device)
            target_tensor = load_image_tensor(target_path, device=device)
            
            # 计算单张图像的LPIPS
            score = lpips_model(gen_tensor, target_tensor).item()
            lpips_scores.append(score)
    
    # 统计结果
    mean_lpips = np.mean(lpips_scores)
    std_lpips = np.std(lpips_scores)
    print(f"LPIPS - 平均值: {mean_lpips:.4f}, 标准差: {std_lpips:.4f}")
    return mean_lpips, std_lpips, lpips_scores

def calculate_fid(gen_folder, target_folder, device='npu', batch_size=32):
    """
    计算FID（Fréchet Inception Distance）
    FID值越小，生成分布与真实分布越接近（理想值≈0，一般<100为优秀）
    """
    print(f"\n=== 计算FID（分布相似度）===")
    # 确保文件夹路径正确
    gen_folder = os.path.abspath(gen_folder)
    target_folder = os.path.abspath(target_folder)
    
    # 使用pytorch-fid库计算FID（自动处理图像加载和特征提取）
    fid_value = calculate_fid_given_paths(
        paths=[gen_folder, target_folder],
        batch_size=batch_size,
        device=device,
        dims=2048,  # Inception v3的2048维特征
        num_workers=4
    )
    
    print(f"FID值: {fid_value:.4f}")
    return fid_value

# ===================== 主函数 =====================
def main():
    parser = argparse.ArgumentParser(description="计算生成结果与目标的LPIPS和FID指标")
    parser.add_argument("--gen_folder", type=str, required=True, help="生成结果文件夹（含1.png、2.png...）")
    parser.add_argument("--target_folder", type=str, required=True, help="目标图像文件夹（含1.png、2.png...）")
    parser.add_argument("--device", type=str, default='npu', help="设备（npu/cpu，建议使用GPU）")
    parser.add_argument("--batch_size", type=int, default=32, help="FID计算的batch_size（根据GPU内存调整）")
    parser.add_argument("--save_scores", action='store_true', help="是否保存每张图像的LPIPS分数到txt文件")
    
    args = parser.parse_args()
    
    # 检查设备
    if args.device == 'npu' and not torch.npu.is_available():
        print("警告：npu不可用，自动切换到CPU（计算速度会很慢！）")
        args.device = 'cpu'
    
    # 1. 获取图像路径并验证
    gen_paths = get_image_paths(args.gen_folder)
    target_paths = get_image_paths(args.target_folder)
    
    print(f"=== 数据统计 ===")
    print(f"生成结果图像数量: {len(gen_paths)}")
    print(f"目标图像数量: {len(target_paths)}")
    
    # 确保两张图像数量一致
    min_num = min(len(gen_paths), len(target_paths))
    if len(gen_paths) != len(target_paths):
        print(f"警告：生成结果和目标图像数量不一致，将使用前{min_num}张图像计算指标")
        gen_paths = gen_paths[:min_num]
        target_paths = target_paths[:min_num]
    
    # 2. 计算指标
    mean_lpips, std_lpips, lpips_scores = calculate_lpips(gen_paths, target_paths, device=args.device)
    fid_value = calculate_fid(args.gen_folder, args.target_folder, device=args.device, batch_size=args.batch_size)
    
    # 3. 保存结果（可选）
    if args.save_scores:
        save_dir = "./metric_results"
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存LPIPS单张分数
        with open(os.path.join(save_dir, "lpips_scores.txt"), 'w') as f:
            f.write("图像序号\t生成路径\t目标路径\tLPIPS分数\n")
            for i, (gen_path, target_path, score) in enumerate(zip(gen_paths, target_paths, lpips_scores)):
                f.write(f"{i+1}\t{gen_path}\t{target_path}\t{score:.4f}\n")
        
        # 保存汇总结果
        with open(os.path.join(save_dir, "metric_summary.txt"), 'w') as f:
            f.write("=== 评估指标汇总 ===\n")
            f.write(f"生成文件夹: {args.gen_folder}\n")
            f.write(f"目标文件夹: {args.target_folder}\n")
            f.write(f"评估图像数量: {len(gen_paths)}\n")
            f.write(f"\nLPIPS - 平均值: {mean_lpips:.4f}, 标准差: {std_lpips:.4f}\n")
            f.write(f"FID - 数值: {fid_value:.4f}\n")
            f.write(f"\n指标说明:\n")
            f.write(f"1. LPIPS: 感知相似度，值越小越优（理想≈0，最大≈1）\n")
            f.write(f"2. FID: 分布相似度，值越小越优（理想≈0，一般<100为优秀）\n")
        
        print(f"\n指标结果已保存到: {save_dir}")
    
    # 4. 打印最终汇总
    print(f"\n=== 最终评估结果 ===")
    print(f"评估图像数量: {len(gen_paths)}")
    print(f"LPIPS: {mean_lpips:.4f} ± {std_lpips:.4f}")
    print(f"FID: {fid_value:.4f}")
    print(f"\n指标说明:")
    print(f"- LPIPS（感知相似度）: 衡量图像的感知质量，值越小表示生成图像与目标越相似（基于人类视觉感知）")
    print(f"- FID（分布相似度）: 衡量生成分布与真实分布的差异，值越小表示生成结果的分布越接近真实数据")

if __name__ == "__main__":
    main()