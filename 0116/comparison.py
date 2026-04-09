import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# ==========================================
# 1. 系統配置與路徑
# ==========================================
RAW_NPY_PATH = r"D:\videopose2\VideoPose3D\0116\ue_raw_sequences\sitting_01.npy"
PROCESSED_NPZ_PATH = "final_rigid_dataset.npz"

# H3.6M 17點語義標籤 [cite: 2026-01-16]
JOINT_NAMES = [
    'Pelvis', 'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle',
    'Spine', 'Thorax', 'Neck', 'Head', 'LShoulder', 'LElbow', 'LWrist',
    'RShoulder', 'RElbow', 'RWrist'
]

# 骨架連線拓撲 [cite: 2026-01-19]
SKELETON_EDGES = [(0,7),(7,8),(8,9),(9,10),(0,1),(1,2),(2,3),(0,4),(4,5),(5,6),(8,11),(11,12),(12,13),(8,14),(14,15),(15,16)]

# ==========================================
# 2. 數據加載與對齊 (Normalization) [cite: 2026-01-20]
# ==========================================
def load_and_normalize():
    # A. 讀取原始數據 (來自 Blender, 單位 cm)
    raw = np.load(RAW_NPY_PATH) / 100.0  # 轉公尺
    # 執行與處理系統相同的鏡像對齊 [cite: 2026-01-19]
    raw_aligned = raw.copy()
    raw_aligned[:, :, 0] = -raw_aligned[:, :, 0] # X 軸鏡像
    raw_aligned[:, :, 1] = -raw_aligned[:, :, 1] # Y 軸鏡像
    # 以 Pelvis 為原點進行中心化
    for f in range(len(raw_aligned)):
        raw_aligned[f] = raw_aligned[f] - raw_aligned[f, 0]

    # B. 讀取處理後的 NPZ (已是公尺, 已中心化) [cite: 2026-01-19]
    proc_dict = np.load(PROCESSED_NPZ_PATH, allow_pickle=True)['positions_3d'].item()
    proc = proc_dict['sitting_01']
    
    return raw_aligned, proc

raw_seq, proc_seq = load_and_normalize()

# ==========================================
# 3. 軌跡分析與視覺化 (Temporal & Spatial Audit) [cite: 2026-01-20]
# ==========================================
fig = plt.figure(figsize=(15, 7))

# --- 左半部：3D 空間重疊比對 (選定第100幀) ---
ax1 = fig.add_subplot(121, projection='3d')
frame = 100
raw_p = raw_seq[frame]
proc_p = proc_seq[frame]

# 繪製原始數據 (半透明灰色)
ax1.scatter(raw_p[:,0], raw_p[:,1], raw_p[:,2], c='gray', alpha=0.3, label='Raw Signal')
for s, e in SKELETON_EDGES:
    ax1.plot([raw_p[s,0], raw_p[e,0]], [raw_p[s,1], raw_p[e,1]], [raw_p[s,2], raw_p[e,2]], c='gray', alpha=0.2)

# 繪製處理後數據 (紅色實線)
ax1.scatter(proc_p[:,0], proc_p[:,1], proc_p[:,2], c='red', s=40, edgecolors='black', label='Processed Alpha')
for s, e in SKELETON_EDGES:
    ax1.plot([proc_p[s,0], proc_p[e,0]], [proc_p[s,1], proc_p[e,1]], [proc_p[s,2], proc_p[e,2]], c='red', lw=2)

# 標註手腕誤差向量 (13: LWrist, 16: RWrist) [cite: 2026-01-16]
for wrist_idx in [13, 16]:
    ax1.plot([raw_p[wrist_idx,0], proc_p[wrist_idx,0]], 
             [raw_p[wrist_idx,1], proc_p[wrist_idx,1]], 
             [raw_p[wrist_idx,2], proc_p[wrist_idx,2]], c='green', lw=3, label=f'Error Vector {JOINT_NAMES[wrist_idx]}')

ax1.set_title(f"Spatial Alignment Audit (Frame {frame})\nGreen line = Refinement Shift")
ax1.set_xlim(-0.8, 0.8); ax1.set_ylim(-0.8, 0.8); ax1.set_zlim(-0.8, 0.8)
ax1.legend()

# --- 右半部：時域軌跡守恆檢查 (Trajectory Invariance) ---
ax2 = fig.add_subplot(122)
# 檢查左手腕 (LWrist) 的 Z 軸運動趨勢是否平行
target_idx = 13 
ax2.plot(raw_seq[:, target_idx, 2], 'g--', alpha=0.5, label='Original Path (Z)')
ax2.plot(proc_seq[:, target_idx, 2], 'r-', linewidth=2, label='Processed Path (Z)')

ax2.set_title(f"Motion Invariance Check: {JOINT_NAMES[target_idx]}")
ax2.set_xlabel("Time (Frames)")
ax2.set_ylabel("Vertical Position (Meters)")
ax2.grid(True, linestyle=':')
ax2.legend()

plt.tight_layout()
plt.show()

# ==========================================
# 4. 數值化審計 (Error Magnitude) [cite: 2026-01-20]
# ==========================================
dist_diff = np.linalg.norm(raw_p - proc_p, axis=1) * 1000 # 轉為 mm
print(f"\n{'='*30} 關節位移偏差報告 (mm) {'='*30}")
for i, name in enumerate(JOINT_NAMES):
    print(f"{name:<15}: {dist_diff[i]:>8.2f} mm")
print(f"{'='*78}")