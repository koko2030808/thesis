import os
# 解決 OpenMP 重複初始化導致的崩潰問題
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 確保路徑讀取 VideoPose3D 的 common 模組
sys.path.append(os.getcwd()) 
from common.model import TemporalModel
from common.camera import normalize_screen_coordinates, normalize_sequence_fixed_scale

# ==========================================
# 1. 索引與結構定義 (標準 H3.6M 17點映射)
# ==========================================
# 從 32 點 GT 提取模型所需的 17 點索引
H36M_17J_INDICES = [0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27]

# 繪圖連線定義
SKELETON_EDGES = [
    (0, 1), (1, 2), (2, 3),        # 右腿
    (0, 4), (4, 5), (5, 6),        # 左腿
    (0, 7), (7, 8), (8, 9), (9, 10), # 軀幹與頭部
    (8, 11), (11, 12), (12, 13),   # 左臂
    (8, 14), (14, 15), (15, 16)    # 右臂
]

# CV 診斷鏈條
BONE_CHAINS = {
    'Right_Leg': (0, 1, 2, 3), 
    'Left_Leg': (0, 4, 5, 6),
    'Torso': (0, 7, 8, 9), 
    'Left_Arm': (8, 11, 12, 13), 
    'Right_Arm': (8, 14, 15, 16)
}

# ==========================================
# 2. 物理指標函數：變異係數 (CV)
# ==========================================
def get_lengths(pos_3d, joints):
    """計算特定骨骼鏈在每一幀的物理長度"""
    lengths = []
    for i in range(len(joints) - 1):
        p1 = pos_3d[:, joints[i], :]
        p2 = pos_3d[:, joints[i+1], :]
        lengths.append(np.linalg.norm(p1 - p2, axis=1))
    return np.sum(lengths, axis=0)

# ==========================================
# 3. 主執行流程
# ==========================================
def run_diagnostic():
    chk_path = r'D:\videopose2\VideoPose3D\checkpoint\pretrained_h36m_cpn.bin'
    data_2d_path = r'data\data_2d_h36m_cpn_ft_h36m_dbb.npz'
    data_3d_path = r'data\data_3d_h36m.npz'
    target = ('S11', 'SittingDown 1')

    # A. 數據載入與 17 點重映射
    d2d = np.load(data_2d_path, allow_pickle=True)['positions_2d'].item()
    d3d = np.load(data_3d_path, allow_pickle=True)['positions_3d'].item()
    kps_raw = d2d[target[0]][target[1]][0]
    gt_3d_17 = d3d[target[0]][target[1]][:, H36M_17J_INDICES, :]

    # B. 推理流程
    kps_screen = normalize_screen_coordinates(kps_raw, w=1000, h=1002)
    # kps_screen = normalize_sequence_fixed_scale(kps_screen)
    model = TemporalModel(17, 2, 17, filter_widths=[3,3,3,3,3], causal=False, channels=1024)
    checkpoint = torch.load(chk_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_pos'])
    model.eval()

    with torch.no_grad():
        pred_3d = model(torch.from_numpy(kps_screen).float().unsqueeze(0)).numpy()[0]

    # C. 時序對齊 (補償 Valid Convolution 造成的影格縮減)
    pad = (kps_screen.shape[0] - pred_3d.shape[0]) // 2
    gt_3d_aligned = gt_3d_17[pad : pad + pred_3d.shape[0]]

    # D. 產出物理品質報告
    print(f"\n" + "="*60)
    print(f" 物理一致性診斷報告: {target[0]} - {target[1]}")
    print("-" * 60)
    print(f"{'部位':<15} | {'GT CV':<12} | {'Pred CV':<12} | {'狀態'}")
    print("-" * 60)
    
    for name, joints in BONE_CHAINS.items():
        p_len = get_lengths(pred_3d, joints)
        g_len = get_lengths(gt_3d_aligned, joints)
        
        # CV = 標準差 / 平均值 (衡量骨骼伸縮程度)
        g_cv = np.std(g_len) / np.mean(g_len)
        p_cv = np.std(p_len) / np.mean(p_len)
        
        status = "✅ 穩定" if g_cv < 0.001 and p_cv < 0.05 else "⚠️ 畸變"
        print(f"{name:<15} | {g_cv:<12.4f} | {p_cv:<12.4f} | {status}")
    print("="*60 + "\n")

    # E. 視覺化人形展示 (影格 300)
    show_human_plot(pred_3d[300], gt_3d_aligned[300], 300)

def show_human_plot(pred, gt, frame_idx):
    """3D 骨架對比圖"""
    fig = plt.figure(figsize=(12, 6))
    all_pts = np.vstack([pred, gt])
    max_range = np.ptp(all_pts, axis=0).max() / 2.0
    mid_x, mid_y, mid_z = np.mean(all_pts, axis=0)
    
    for i, (pos, title) in enumerate([(gt, "Ground Truth"), (pred, "Prediction")]):
        ax = fig.add_subplot(1, 2, i+1, projection='3d')
        ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=20, c='red')
        for start, end in SKELETON_EDGES:
            ax.plot([pos[start, 0], pos[end, 0]], [pos[start, 1], pos[end, 1]], 
                    [pos[start, 2], pos[end, 2]], c='blue', linewidth=2)
        ax.set_title(f"{title} (Frame {frame_idx})")
        # 強制等比例縮放
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_diagnostic()