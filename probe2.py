import numpy as np
import sys
import os
sys.path.append(os.getcwd())

d2d = np.load(r'data\data_2d_h36m_cpn_ft_h36m_dbb.npz', allow_pickle=True)['positions_2d'].item()
from common.h36m_dataset import Human36mDataset
dataset_3d = Human36mDataset(r'data\data_3d_h36m.npz')
gt_data = dataset_3d._data

sub = 'S9'
action = 'Walking'

print("==== 時序長度探測 ====")
print("2D cameras:", len(d2d[sub][action]))
for i, cam in enumerate(d2d[sub][action]):
    print(f"  Cam{i} 2D shape: {cam.shape}")

print("GT shape:", gt_data[sub][action]['positions'].shape)