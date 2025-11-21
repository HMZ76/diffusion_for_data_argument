import pickle
import pandas as pd
from tqdm import tqdm
tqdm.pandas()
import simulate
import numpy as np
import matplotlib.pyplot as plt
def stride_sampling_downsample_4x(x):
    """NumPy 间隔采样下采样（128×128 → 32×32）：直接取每隔3个像素的点"""
    is_channel_first = len(x.shape) == 3 and x.shape[0] < x.shape[1]
    if is_channel_first:
        # [C, H, W] → 取 H和W维度的步幅4
        return x[::4, ::4]  # 索引：从0开始，每4个取1个
    else:
        # [H, W, C] → 取 H和W维度的步幅4
        return x[::4, ::4]

# 读取数据
with open('data/36h11_training.pkl', 'rb') as file:
    data1 = pickle.load(file)
with open('data/16h5_training.pkl', 'rb') as file:
    data2 = pickle.load(file)
data = pd.concat([data1, data2], ignore_index=True)[0:100]

# 后续处理（保持原逻辑）
data["Marker"] = data.progress_apply(
    lambda e: simulate.load_tag(e.Size, e.MarkerType, e.Id, e.Angle, MARKER_PATH="./markers_official/"), 
    axis=1
)
print(data)
img1 = data.loc[10, 'Marker']
img2 = data.loc[10, 'Blob']
print(img1.shape, img2.shape)
img1 = stride_sampling_downsample_4x(img1)
print(img1.shape, img2.shape)
plt.imsave("test_marker.png", img1, cmap='gray')
plt.imsave("test_blob.png", img2, cmap='gray')

