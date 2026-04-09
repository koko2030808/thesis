import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import pearsonr

# ==========================================
# 1. 數據加載
# ==========================================
DATA_PATH = r'D:\videopose3d1\VideoPose3D\data\data_2d_h36m_cpn_ft_h36m_dbb.npz'
TEST_SUBJECT = 'S11'
TEST_ACTION = 'SittingDown 1' 

def load_data(path):
    if not os.path.exists(path):
        print(f"錯誤：找不到檔案 {path}")
        exit()
    return np.load(path, allow_pickle=True)['positions_2d'].item()

# ==========================================
# 2. 歸一化邏輯 (修正維度報錯)
# ==========================================

def normalize_old_per_frame(X):
    """舊方法：逐幀拉伸 (抹除深度特徵)"""
    X = np.array(X) # 確保為 (17, 2)
    # 明確指定 axis=0 得到 (2,) 向量
    x_min, y_min = np.min(X, axis=0)
    x_max, y_max = np.max(X, axis=0)
    
    y_scale = 2.0 / (y_max - y_min + 1e-6)
    
    # 修正重點：將 offset 強制設為 (1, 2) 以利廣播
    center_x = x_min + (x_max - x_min) / 2.0
    center_offset = np.array([[center_x, y_min]]) 
    
    # 執行運算: (17,2) - (1,2) -> (17,2)
    X_norm = (X - center_offset) * y_scale - np.array([[0, 1]])
    return X_norm

def normalize_new_fixed_scale(X_seq, ref_idx=0):
    """新方法：固定比例縮放 (保留深度特徵)"""
    X_seq = np.array(X_seq)
    ref_frame = X_seq[ref_idx]
    ref_height = np.max(ref_frame[:, 1]) - np.min(ref_frame[:, 1])
    fixed_scale = 2.0 / (ref_height + 1e-6)
    
    X_norm_seq = np.zeros_like(X_seq)
    for f in range(X_seq.shape[0]):
        # 以 Hip (Index 0) 為中心平移
        root = X_seq[f, 0:1, :].copy() # 保持維度為 (1, 2)
        X_norm_seq[f] = (X_seq[f] - root) * fixed_scale
    return X_norm_seq

# ==========================================
# 3. 系統性驗證分析 (SOP: 相關性驗證)
# ==========================================

print(f"正在執行系統性驗證: {TEST_ACTION}...")
data = load_data(DATA_PATH)
raw_seq = np.array(data[TEST_SUBJECT][TEST_ACTION][0])

# 1. 執行歸一化
kps_old = np.array([normalize_old_per_frame(f) for f in raw_seq])
kps_new = normalize_new_fixed_scale(raw_seq)

# 2. 提取關鍵高度軌跡 (頭部 10, 腳踝 6)
# 原始像素高度 (Ground Truth)
raw_h = np.abs(raw_seq[:, 10, 1] - raw_seq[:, 6, 1])
# 舊方法高度 (應該會趨於平坦)
old_h = np.abs(kps_old[:, 10, 1] - kps_old[:, 6, 1])
# 新方法高度 (應該與原始信號同步)
new_h = np.abs(kps_new[:, 10, 1] - kps_new[:, 6, 1])

# 3. 計算皮爾森相關係數
corr_old, _ = pearsonr(raw_h, old_h)
corr_new, _ = pearsonr(raw_h, new_h)

# 4. 繪製 SOP 驗證圖
fig, ax1 = plt.subplots(figsize=(12, 6))

ax1.set_xlabel('Frame Index')
ax1.set_ylabel('Raw Pixel Height (Original)', color='gray')
ax1.plot(raw_h, color='gray', alpha=0.3, label='Raw Signal (Physical Reality)')
ax1.tick_params(axis='y', labelcolor='gray')

ax2 = ax1.twinx()
ax2.plot(old_h, color='red', linestyle='--', label=f'Old: Per-frame (Corr: {corr_old:.4f})')
ax2.plot(new_h, color='blue', linewidth=2, label=f'New: Fixed Scale (Corr: {corr_new:.4f})')
ax2.set_ylabel('Normalized Height (Feature Space)')
ax2.set_ylim(-0.1, 2.5)

plt.title(f"Systematic SOP: Feature Fidelity Verification\n{TEST_ACTION}")
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

plt.grid(True, alpha=0.2)
plt.show()

print(f"\n[驗證報告]")
print(f"舊方法相關性: {corr_old:.4f}")
print(f"新方法相關性: {corr_new:.4f}")

if corr_new > 0.99:
    print(">> 結論：藍色波型與原始動作完美同步，證明深度特徵已被正確保留。")