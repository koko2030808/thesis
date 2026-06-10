import numpy as np

d = np.load('data/data_3d_h36m.npz', allow_pickle=True)
dataset = d['positions_3d'].item()
s9 = dataset['S9']

for action in ['Sitting', 'SittingDown', 'Walking']:
    if action not in s9:
        continue
    pos = s9[action]  # (T, 32, 3)
    
    print(f"\n=== {action} (shape: {pos.shape}) ===")
    print(f"第0幀 joint0(骨盆) XYZ: {pos[0, 0, :]}")
    print(f"第0幀 joint0 → joint1 XYZ差: {pos[0, 1, :] - pos[0, 0, :]}")
    
    # 找哪個軸是高度軸（變化最大的）
    for ax, label in enumerate(['X', 'Y', 'Z']):
        vals = pos[:, 0, ax]
        print(f"  骨盆 {label}: min={vals.min():.1f}  max={vals.max():.1f}  "
              f"range={vals.max()-vals.min():.1f}  change={vals[-1]-vals[0]:.1f}")