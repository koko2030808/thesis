import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from tqdm import tqdm

# ==========================================
# 1. 配置路徑
# ==========================================
DATA_PATH = r'D:\videopose3d1\VideoPose3D\data\data_2d_h36m_cpn_ft_h36m_dbb.npz'
# 請放入對應的 H36M 影片路徑 (例如 S11 的 SittingDown 1)
VIDEO_PATH = r'D:\videopose3d1\VideoPose3D\S11\Videos\Sitting 1.54138969.mp4' 
OUTPUT_VIDEO = 'RGB_Raw_Norm_Comparison.mp4'
TEST_ACTION = 'SittingDown 1'

def load_data(path):
    return np.load(path, allow_pickle=True)['positions_2d'].item()

# ==========================================
# 2. 歸一化邏輯 (鎖定第一幀比例)
# ==========================================
def normalize_fixed_scale(X_seq, ref_idx=0):
    ref_frame = X_seq[ref_idx]
    y_min, y_max = ref_frame[:, 1].min(), ref_frame[:, 1].max()
    ref_height = y_max - y_min
    fixed_scale = 2.0 / (ref_height + 1e-6)
    
    X_norm = np.zeros_like(X_seq)
    for f in range(X_seq.shape[0]):
        root = X_seq[f, 0].copy()
        X_norm[f] = (X_seq[f] - root) * fixed_scale
    return X_norm

# ==========================================
# 3. 三聯動渲染邏輯
# ==========================================
def get_h36m_connections():
    return [(0,1), (1,2), (2,3), (0,4), (4,5), (5,6), (0,7), (7,8), 
            (8,9), (9,10), (8,11), (11,12), (12,13), (8,14), (14,15), (15,16)]

def render_triple_frame(rgb_img, raw_f, norm_f, frame_idx, standing_head_y, raw_limits):
    connections = get_h36m_connections()
    # 創建三個子圖
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6), dpi=100)
    
    # --- 左圖：Original RGB Video ---
    if rgb_img is not None:
        ax1.imshow(cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB))
    ax1.set_title("Original RGB Video")
    ax1.axis('off')

    # --- 中圖：Raw 2D Pixels (Static View) ---
    raw_p = raw_f.copy()
    raw_p[:, 1] = -raw_p[:, 1] # 轉正
    for s, e in connections:
        ax2.plot([raw_p[s, 0], raw_p[e, 0]], [raw_p[s, 1], raw_p[e, 1]], '-o', markersize=3, alpha=0.6)
    ax2.set_title("Raw 2D Pixels (Pixel Space)")
    ax2.axis('equal')
    ax2.set_xlim(raw_limits['x_min'], raw_limits['x_max'])
    ax2.set_ylim(raw_limits['y_min'], raw_limits['y_max'])

    # --- 右圖：Normalized (Fixed Scale) ---
    norm_p = norm_f.copy()
    norm_p[:, 1] = -norm_p[:, 1] # 轉正
    ax3.axhline(y=-standing_head_y, color='red', linestyle=':', label='Standing Head Level')
    for s, e in connections:
        ax3.plot([norm_p[s, 0], norm_p[e, 0]], [norm_p[s, 1], norm_p[e, 1]], '-o', markersize=4, linewidth=2)
    ax3.set_title("Normalized (Proposed Method)")
    ax3.axis('equal')
    ax3.set_xlim(-1.5, 1.5)
    ax3.set_ylim(-1.5, 1.5)
    ax3.legend(loc='lower right')

    plt.suptitle(f"Frame: {frame_idx} | Action: {TEST_ACTION}", fontsize=16)
    
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

# ==========================================
# 4. 主流程
# ==========================================
print("準備數據中...")
data = load_data(DATA_PATH)
raw_seq = data['S11'][TEST_ACTION][0]
norm_seq = normalize_fixed_scale(raw_seq)

# 計算邊界
raw_flipped_y = -raw_seq[:, :, 1]
raw_limits = {'x_min': raw_seq[:, :, 0].min()-50, 'x_max': raw_seq[:, :, 0].max()+50,
              'y_min': raw_flipped_y.min()-50, 'y_max': raw_flipped_y.max()+50}
standing_head_y = norm_seq[0][10, 1]

# 影片讀取器
cap = cv2.VideoCapture(VIDEO_PATH)
total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"影片總影格數: {total_video_frames}, 2D 數據影格數: {len(raw_seq)}")

# 影片寫入器設定
sample_img = render_triple_frame(None, raw_seq[0], norm_seq[0], 0, standing_head_y, raw_limits)
h, w, _ = sample_img.shape
out = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))

# 執行渲染
for i in tqdm(range(min(len(raw_seq), total_video_frames))):
    ret, frame_rgb = cap.read()
    if not ret: break
    
    triple_frame = render_triple_frame(frame_rgb, raw_seq[i], norm_seq[i], i, standing_head_y, raw_limits)
    out.write(triple_frame)

cap.release()
out.release()
print(f"強大版對比影片生成完成：{os.path.abspath(OUTPUT_VIDEO)}")