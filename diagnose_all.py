import os
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
# 1. 結構定義與重映射
# ==========================================
H36M_17J_INDICES = [0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27]

BONE_CHAINS = {
    'Right_Leg': (0, 1, 2, 3), 'Left_Leg': (0, 4, 5, 6),
    'Torso': (0, 7, 8, 9), 'Left_Arm': (8, 11, 12, 13), 'Right_Arm': (8, 14, 15, 16)
}

SKELETON_EDGES = [
    (0, 1), (1, 2), (2, 3),        # 右腿
    (0, 4), (4, 5), (5, 6),        # 左腿
    (0, 7), (7, 8), (8, 9), (9, 10), # 軀幹與頭部
    (8, 11), (11, 12), (12, 13),   # 左臂
    (8, 14), (14, 15), (15, 16)    # 右臂
]

def get_lengths(pos_3d, joints):
    """計算每一影格的骨骼鏈總長度"""
    lengths = []
    for i in range(len(joints) - 1):
        p1 = pos_3d[:, joints[i], :]
        p2 = pos_3d[:, joints[i+1], :]
        lengths.append(np.linalg.norm(p1 - p2, axis=1))
    return np.sum(lengths, axis=0)

# ==========================================
# 2. 自動獵人掃描核心
# ==========================================
def run_full_scan():
    # 路徑設定
    chk_path = r'D:\videopose2\VideoPose3D\checkpoint\pretrained_h36m_cpn.bin'
    data_2d_path = r'data\data_2d_h36m_cpn_ft_h36m_dbb.npz'
    data_3d_path = r'data\data_3d_h36m.npz'
    
    # 載入模型
    model = TemporalModel(17, 2, 17, filter_widths=[3,3,3,3,3], causal=False, channels=1024)
    checkpoint = torch.load(chk_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_pos'])
    model.eval()

    # 載入全數據集
    d2d = np.load(data_2d_path, allow_pickle=True)['positions_2d'].item()
    d3d = np.load(data_3d_path, allow_pickle=True)['positions_3d'].item()
    
    all_results = []
    subjects = ['S9', 'S11'] # 專注於測試集對象
    
    print("--- 正在啟動全數據集獵人掃描，定位物理崩潰點... ---")

    for sub in subjects:
        for action in d2d[sub].keys():
            # 處理多個 sub-action 子序列
            for idx in range(len(d2d[sub][action])):
                kps_raw = d2d[sub][action][idx]
                gt_3d_all = d3d[sub][action][:, H36M_17J_INDICES, :] # 重映射

                # 推理
                kps_screen = normalize_screen_coordinates(kps_raw, w=1000, h=1002)
                # kps_screen = normalize_sequence_fixed_scale(kps_screen)
                with torch.no_grad():
                    pred_3d = model(torch.from_numpy(kps_screen).float().unsqueeze(0)).numpy()[0]
                
                # 時序對齊
                pad = (kps_screen.shape[0] - pred_3d.shape[0]) // 2
                gt_3d = gt_3d_all[pad : pad + pred_3d.shape[0]]

                # 計算此動作的平均 Pred CV (物理不穩定度)
                total_cv = 0
                for name, joints in BONE_CHAINS.items():
                    p_len = get_lengths(pred_3d, joints)
                    total_cv += (np.std(p_len) / np.mean(p_len))
                
                avg_cv = total_cv / len(BONE_CHAINS)
                all_results.append({
                    'subject': sub,
                    'action': action,
                    'idx': idx,
                    'avg_cv': avg_cv,
                    'pred': pred_3d,
                    'gt': gt_3d
                })

    # 根據 CV 排序，找出物理畸變最嚴重的動作 (Pred CV 最大的)
    all_results.sort(key=lambda x: x['avg_cv'], reverse=True)

    print("\n" + "="*60)
    print(" 🏆 物理崩潰動作排行榜 (Top 3 Worst Actions)")
    print("-" * 60)
    for i in range(3):
        res = all_results[i]
        print(f"Rank {i+1}: {res['subject']} - {res['action']} (Sub {res['idx']}) | Avg Pred CV: {res['avg_cv']:.4f}")
    print("="*60 + "\n")

    # 展示排名第一（最爛）的動作
    worst = all_results[0]
    print(f"正在展示物理畸變最嚴重的動作: {worst['subject']} - {worst['action']}")
    show_human_plot(worst['pred'][300], worst['gt'][300], f"{worst['subject']}-{worst['action']}")

def show_human_plot(pred, gt, title_suffix):
    """3D 對比繪圖"""
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
        ax.set_title(f"{title}\n{title_suffix}")
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
    plt.show()

if __name__ == "__main__":
    run_full_scan()