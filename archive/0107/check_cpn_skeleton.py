import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================
# 1. 核心映射函式 (Mapping Function)
# ==========================================
def coco_to_h36m_mapping(coco_kps):
    """
    將 COCO 17 點格式轉換為 H36M 預期的 17 點骨架
    """
    if len(coco_kps.shape) == 2:
        coco_kps = coco_kps[np.newaxis, ...]
        
    N = coco_kps.shape[0]
    h36m_kps = np.zeros((N, 17, 2), dtype='float32')
    
    # 計算 H36M 缺失的虛擬中心點
    pelvis = (coco_kps[:, 11] + coco_kps[:, 12]) / 2.0
    thorax = (coco_kps[:, 5] + coco_kps[:, 6]) / 2.0
    spine = (pelvis + thorax) / 2.0
    neck = thorax * 0.7 + coco_kps[:, 0] * 0.3 

    # 重新填入 H36M 索引位置
    h36m_kps[:, 0] = pelvis
    h36m_kps[:, 1] = coco_kps[:, 12]; h36m_kps[:, 2] = coco_kps[:, 14]; h36m_kps[:, 3] = coco_kps[:, 16]
    h36m_kps[:, 4] = coco_kps[:, 11]; h36m_kps[:, 5] = coco_kps[:, 13]; h36m_kps[:, 6] = coco_kps[:, 15]
    h36m_kps[:, 7] = spine; h36m_kps[:, 8] = thorax; h36m_kps[:, 9] = neck
    h36m_kps[:, 10] = coco_kps[:, 0] # Head (Nose)
    h36m_kps[:, 11] = coco_kps[:, 5]; h36m_kps[:, 12] = coco_kps[:, 7]; h36m_kps[:, 13] = coco_kps[:, 9]
    h36m_kps[:, 14] = coco_kps[:, 6]; h36m_kps[:, 15] = coco_kps[:, 8]; h36m_kps[:, 16] = coco_kps[:, 10]
    
    return h36m_kps

# ==========================================
# 2. 定義連線邏輯 (Connectivity)
# ==========================================
coco_lines = [[5, 6, 12, 11, 5], [5, 7, 9], [6, 8, 10], [11, 13, 15], [12, 14, 16], [0, 1, 3], [0, 2, 4]]
h36m_lines = [[0, 7, 8, 9, 10], [0, 1, 2, 3], [0, 4, 5, 6], [8, 11, 12, 13], [8, 14, 15, 16]]

# ==========================================
# 3. 讀取資料 (動態路徑與動態索引)
# ==========================================
# 請確保此路徑指向你目前的檔案
file_path = 'VideoPose3D/data/data_2d_custom_1.npz'
if not os.path.exists(file_path):
    file_path = 'data/data_2d_custom_T-pose_ch01.npz'

print(f"正在讀取檔案: {file_path}")
data = np.load(file_path, allow_pickle=True)
kps_dict = data['positions_2d'].item()

# --- 動態抓取第一個可用的數據路徑 ---
subj_list = list(kps_dict.keys())
first_subj = subj_list[0]
action_list = list(kps_dict[first_subj].keys())
first_action = action_list[0]

print(f"成功找到數據: Subject '{first_subj}', Action '{first_action}'")

# 抓取第一幀
sample_coco = kps_dict[first_subj][first_action][0][0] 
sample_h36m = coco_to_h36m_mapping(sample_coco)[0]

# ==========================================
# 4. 繪製對照圖
# ==========================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))

# --- 左圖: COCO (Before) ---
ax1.scatter(sample_coco[:, 0], sample_coco[:, 1], c='red', s=40)
for line in coco_lines:
    ax1.plot(sample_coco[line, 0], sample_coco[line, 1], 'r-', alpha=0.5)
for i in range(17):
    ax1.text(sample_coco[i, 0], sample_coco[i, 1], str(i), fontsize=10, color='darkred')
ax1.set_title(f"BEFORE: COCO Format\n(Subject: {first_subj})")
ax1.invert_yaxis()
ax1.axis('equal')

# --- 右圖: H36M (After) ---
ax2.scatter(sample_h36m[:, 0], sample_h36m[:, 1], c='blue', s=40)
for line in h36m_lines:
    ax2.plot(sample_h36m[line, 0], sample_h36m[line, 1], 'b-', alpha=0.5)
for i in range(17):
    ax2.text(sample_h36m[i, 0], sample_h36m[i, 1], str(i), fontsize=10, color='darkblue')
ax2.set_title("AFTER: Mapped H36M Format\n(Ready for VideoPose3D)")
ax2.invert_yaxis()
ax2.axis('equal')

plt.suptitle(f"Skeleton Mapping Verification: {os.path.basename(file_path)}", fontsize=16)
plt.tight_layout()
plt.show()

print("驗證結束。如果右圖骨架結構正確，即可將 Mapping 函式移回 run.py 進行訓練。")