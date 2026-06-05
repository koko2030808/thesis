import numpy as np
import sys
import os
sys.path.append(os.getcwd())
from common.h36m_dataset import Human36mDataset

print("==== 3D GT Sanity Check ====")
dataset_3d = Human36mDataset(r'data\data_3d_h36m.npz')
subjects = dataset_3d.subjects()
print(f"GT subjects: {subjects}")

# 取出 S9 Walking 的 GT 資料
gt_pos = dataset_3d['S9']['Walking']['positions']
print(f"positions shape: {gt_pos.shape}")
print(f"positions 值域: {gt_pos.min():.4f} ~ {gt_pos.max():.4f}")

print("\n==== 2D Pred Sanity Check ====")
d2d = np.load(r'data\data_2d_h36m_cpn_ft_h36m_dbb.npz', allow_pickle=True)['positions_2d'].item()
kps = d2d['S9']['Walking'][0] # 取 Cam 0
print(f"2D shape: {kps.shape}")
print(f"2D 值域: {kps.min():.4f} ~ {kps.max():.4f}")PYTHON