import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# ==========================================
# 1. 加載量產後的成品 [cite: 2026-01-19]
# ==========================================
data_path = "final_v11_standardized_dataset.npz"
data = np.load(data_path, allow_pickle=True)['positions_3d'].item()
sequence_name = list(data.keys())[0] # 取第一個序列進行驗證
poses = data[sequence_name] # 形狀: (Frames, 17, 3)

# H3.6M 17點連線拓撲 [cite: 2026-01-19]
SKELETON_EDGES = [
    (0, 7), (7, 8), (8, 9), (9, 10),      # 軀幹
    (0, 1), (1, 2), (2, 3),               # 右腿
    (0, 4), (4, 5), (5, 6),               # 左腿
    (8, 11), (11, 12), (12, 13),          # 左臂
    (8, 14), (14, 15), (15, 16)           # 右臂
]

# ==========================================
# 2. 建立動態畫布 (Temporal Audit) [cite: 2026-01-20]
# ==========================================
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

def update(frame):
    ax.cla()
    pose = poses[frame]
    
    # 設置固定觀察範圍，確保動畫不晃動 [cite: 2026-01-19]
    ax.set_xlim(-0.8, 0.8)
    ax.set_ylim(-0.8, 0.8)
    ax.set_zlim(-0.8, 0.8)
    ax.set_title(f"Processed Audit: {sequence_name} | Frame: {frame}")
    
    # 繪製點
    ax.scatter(pose[:, 0], pose[:, 1], pose[:, 2], c='red', s=40)
    
    # 繪製骨架連線
    for s, e in SKELETON_EDGES:
        ax.plot([pose[s,0], pose[e,0]], 
                [pose[s,1], pose[e,1]], 
                [pose[s,2], pose[e,2]], c='blue', lw=2)

# ==========================================
# 3. 執行播放 (First Principles Logic) [cite: 2026-01-20]
# ==========================================
print(f">>> 正在生成時域動畫驗證...")
ani = FuncAnimation(fig, update, frames=range(0, len(poses), 2), interval=50)
plt.show()