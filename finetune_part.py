import os
# [修正] 解決 OpenMP 重複初始化導致的崩潰問題
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter

# 確保路徑能讀取到 VideoPose3D 的 common 模組
sys.path.append(os.getcwd()) 
from common.model import TemporalModel
from common.camera import normalize_screen_coordinates

# ==========================================
# 1. 結構定義與重映射 (對齊你的數據排毒成果)
# ==========================================
H36M_17J_INDICES = [0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27]

# 專注於手臂的物理診斷
TARGET_CHAIN = ('Left_Arm', (8, 11, 12, 13)) 

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
# 2. 動畫生成核心 (修正解包錯誤)
# ==========================================
def generate_disease_gif():
    # 路徑與目標設定
    chk_path = r'D:\videopose2\VideoPose3D\checkpoint\pretrained_h36m_cpn.bin'
    data_2d_path = r'data\data_2d_h36m_cpn_ft_h36m_dbb.npz'
    data_3d_path = r'data\data_3d_h36m.npz'
    target = ('S9', 'Sitting 1') 
    sub_action_idx = 3 # 你排行榜中的病態子動作

    print(f"--- 正在啟動病灶動畫產生器: {target[0]}-{target[1]} ---")

    # A. 數據載入
    d2d = np.load(data_2d_path, allow_pickle=True)['positions_2d'].item()
    d3d = np.load(data_3d_path, allow_pickle=True)['positions_3d'].item()
    kps_raw = d2d[target[0]][target[1]][sub_action_idx]
    gt_3d_all = d3d[target[0]][target[1]][:, H36M_17J_INDICES, :]

    # B. 推理流程
    kps_screen = normalize_screen_coordinates(kps_raw, w=1000, h=1002)
    model = TemporalModel(17, 2, 17, filter_widths=[3,3,3,3,3], causal=False, channels=1024)
    checkpoint = torch.load(chk_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_pos'])
    model.eval()

    with torch.no_grad():
        pred_3d = model(torch.from_numpy(kps_screen).float().unsqueeze(0)).numpy()[0]
    
    pad = (kps_screen.shape[0] - pred_3d.shape[0]) // 2
    gt_3d = gt_3d_all[pad : pad + pred_3d.shape[0]]

    # C. 定位畸變最嚴重的片段
    p_len = get_lengths(pred_3d, TARGET_CHAIN[1])
    variation_cm = (np.max(p_len) - np.min(p_len)) / 10.0 # 假設單位mm轉cm
    
    window_size = 60 
    cv_scores = []
    for i in range(len(p_len) - window_size):
        window = p_len[i : i + window_size]
        cv_scores.append(np.std(window) / np.mean(window))
    
    worst_start_idx = np.argmax(cv_scores)
    worst_end_idx = worst_start_idx + window_size

    # D. 準備繪圖中心與縮放 (關鍵修正點)
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    
    # 確保 mean 作用於二維陣列
    combined_samples = np.vstack([pred_3d[worst_start_idx:worst_end_idx].reshape(-1, 3), 
                                 gt_3d[worst_start_idx:worst_end_idx].reshape(-1, 3)])
    center = np.mean(combined_samples, axis=0)
    mid_x, mid_y, mid_z = center[0], center[1], center[2] # 修正解包
    max_range = np.ptp(combined_samples, axis=0).max() / 2.0

    def update(frame):
        idx = worst_start_idx + frame
        ax1.clear(); ax2.clear()
        
        # 繪製 GT (左)
        plot_frame(ax1, gt_3d[idx], f"Ground Truth (Stable)\nFrame {idx}", mid_x, mid_y, mid_z, max_range)
        # 繪製 Pred (右)
        plot_frame(ax2, pred_3d[idx], f"Prediction (Elastic Error)\n{TARGET_CHAIN[0]} Var: {variation_cm:.2f} cm", mid_x, mid_y, mid_z, max_range)

    ani = FuncAnimation(fig, update, frames=window_size, interval=50)
    gif_path = f"worst_case_{target[1].replace(' ', '_')}.gif"
    print(f"正在生成病灶動畫: {gif_path} ...")
    ani.save(gif_path, writer=PillowWriter(fps=20))
    print("✅ 成功！請檢查目錄下的 GIF 檔案。")

def plot_frame(ax, pos, title, mid_x, mid_y, mid_z, max_range):
    ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=20, c='red')
    for start, end in SKELETON_EDGES:
        ax.plot([pos[start, 0], pos[end, 0]], [pos[start, 1], pos[end, 1]], 
                [pos[start, 2], pos[end, 2]], c='blue', linewidth=2)
    ax.set_title(title)
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    ax.view_init(elev=20, azim=-60)

if __name__ == "__main__":
    generate_disease_gif()