import pickle
import pandas as pd
from tqdm import tqdm
tqdm.pandas()
import simulate
import numpy as np


# 读取数据
with open('data/36h11_training.pkl', 'rb') as file:
    data1 = pickle.load(file)
with open('data/16h5_training.pkl', 'rb') as file:
    data2 = pickle.load(file)
data = pd.concat([data1, data2], ignore_index=True)


# 后续处理（保持原逻辑）
data["Marker"] = data.progress_apply(
    lambda e: simulate.load_tag(e.Size, e.MarkerType, e.Id, e.Angle, MARKER_PATH="./markers_official/"), 
    axis=1
)
data = data.drop(columns=["Size", "MarkerType", "Id", "Angle", "Timeseries"])

# 保存数据（保持原逻辑）
for idx, row in data.iterrows():
    input_filename = f"./generated_data/example_input_{idx:05d}.npy"
    target_filename = f"./generated_data/example_target_{idx:05d}.npy"
    np.save(input_filename, row['Marker'])
    np.save(target_filename, row['Blob'])


# 读取数据
with open('data/36h11_validation.pkl', 'rb') as file:
    data1 = pickle.load(file)
with open('data/16h5_validation.pkl', 'rb') as file:
    data2 = pickle.load(file)
data = pd.concat([data1, data2], ignore_index=True) #只取了前10000条数据


# 后续处理（保持原逻辑）
data["Marker"] = data.progress_apply(
    lambda e: simulate.load_tag(e.Size, e.MarkerType, e.Id, e.Angle, MARKER_PATH="./markers_official/"), 
    axis=1
)
data = data.drop(columns=["Size", "MarkerType", "Id", "Angle", "Timeseries"])

# 保存数据（保持原逻辑）
for idx, row in data.iterrows():
    input_filename = f"./val_data/example_input_{idx:05d}.npy"
    target_filename = f"./val_data/example_target_{idx:05d}.npy"
    np.save(input_filename, row['Marker'])
    np.save(target_filename, row['Blob'])
