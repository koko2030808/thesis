"""
UE5 疫苗數據 JSON → NPZ 轉換器 V2
===================================
根據 H36M NPZ 格式分析正確撰寫

輸入：
  - vaccine_data.json      (3D, UE5 世界座標, cm)
  - vaccine_camera.json    (相機外參)
  - vaccine_data_2d.json   (2D, 像素座標)

輸出：
  - data_3d_vaccine.npz    (相機空間, root-relative, 公尺)
  - data_2d_vaccine_ue5.npz (歸一化螢幕座標)

執行：python json_to_npz_v2.py
"""

import json
import numpy as np
import os
from scipy.spatial.transform import Rotation as R

# ============================================================
# 路徑設定
# ============================================================
BASE_DIR         = r"D:\04.09\VideoPose3D"
JSON_3D_PATH     = os.path.join(BASE_DIR, "vaccine_data.json")
JSON_CAM_PATH    = os.path.join(BASE_DIR, "vaccine_camera.json")
JSON_2D_PATH     = os.path.join(BASE_DIR, "vaccine_data_2d.json")
OUTPUT_DIR       = os.path.join(BASE_DIR, "data")

OUTPUT_3D        = os.path.join(OUTPUT_DIR, "data_3d_vaccine.npz")
OUTPUT_2D        = os.path.join(OUTPUT_DIR, "data_2d_vaccine_ue5.npz")

SUBJECT          = "S_vaccine"
ACTION           = "Sitting 1"
IMAGE_W          = 1000
IMAGE_H          = 1000

# ============================================================
# Step 1：載入 JSON
# ============================================================
print("[1] 載入 JSON...")

with open(JSON_3D_PATH, 'r') as f:
    d3 = json.load(f)
with open(JSON_CAM_PATH, 'r') as f:
    dc = json.load(f)
with open(JSON_2D_PATH, 'r') as f:
    d2 = json.load(f)

# 3D：shape (T, 17, 3)，cm，UE5 世界座標
frames_3d = np.array(d3["data"][0], dtype=np.float64)

# 2D：shape (T, 17, 2)，像素座標
frames_2d_raw = d2["data"] if "data" in d2 else d2
if isinstance(frames_2d_raw[0][0], list):
    frames_2d = np.array(frames_2d_raw, dtype=np.float64)
else:
    frames_2d = np.array(frames_2d_raw, dtype=np.float64)

# 相機（取第一幀的參數）
cam_frame = dc["data"][0] if "data" in dc else dc[0]
cam_loc   = np.array(cam_frame["location"], dtype=np.float64)   # cm
cam_rot   = cam_frame["rotation"]                                # [Roll, Pitch, Yaw]

T = frames_3d.shape[0]
print(f"    幀數：{T}，3D shape：{frames_3d.shape}，2D shape：{frames_2d.shape}")

# ============================================================
# Step 2：UE5 世界座標 → OpenCV 相機座標（公尺）
# ============================================================
print("[2] 座標系轉換...")

r_obj = R.from_euler('ZYX', [cam_rot[2], -cam_rot[1], -cam_rot[0]], degrees=True)

poses_cam_m = np.zeros((T, 17, 3), dtype=np.float64)
for t in range(T):
    for j in range(17):
        p = frames_3d[t, j] - cam_loc              # 平移到相機原點
        p_local = r_obj.apply(p, inverse=True)     # 旋轉到相機局部空間

        # UE5 局部(X前,Y右,Z上) → OpenCV(X右,Y下,Z前)
        cv_x = p_local[1]
        cv_y = -p_local[2]
        cv_z = p_local[0]

        # cm → 公尺
        poses_cam_m[t, j] = [cv_x / 100.0, cv_y / 100.0, cv_z / 100.0]

print(f"    深度(Z)範圍：{poses_cam_m[:, :, 2].min():.3f} ~ {poses_cam_m[:, :, 2].max():.3f} m")

# ============================================================
# Step 3：Root-relative（減去 Pelvis，Joint 0）
# ============================================================
print("[3] Root-relative 處理...")

poses_rel = poses_cam_m.copy()
poses_rel[:, 1:, :] -= poses_rel[:, :1, :]  # 保留 joint 0，其餘減去 root

# Sanity check：骨骼長度
spine_len = np.linalg.norm(poses_rel[:, 7, :], axis=1).mean()
print(f"    平均 Spine 骨長：{spine_len * 1000:.1f} mm（H36M 參考：~240mm）")
if spine_len > 0.5 or spine_len < 0.05:
    print("    ⚠️  骨長異常，請檢查座標轉換！")
else:
    print("    ✅ 骨長正常")

# ============================================================
# Step 4：2D 歸一化（與 run.py 的 normalize_screen_coordinates 一致）
# ============================================================
print("[4] 2D 歸一化...")

# normalize_screen_coordinates 公式：
# x_norm = x / (w/2) - 1
# y_norm = y / (w/2) - 1   （注意分母是 w，不是 h）
poses_2d_norm = frames_2d.copy()
poses_2d_norm[..., 0] = frames_2d[..., 0] / (IMAGE_W / 2.0) - 1.0
poses_2d_norm[..., 1] = frames_2d[..., 1] / (IMAGE_W / 2.0) - 1.0

print(f"    2D 範圍：{poses_2d_norm.min():.3f} ~ {poses_2d_norm.max():.3f}（應在 -1 ~ 1 附近）")

# ============================================================
# Step 5：儲存 NPZ
# ============================================================
print("[5] 儲存 NPZ...")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 3D NPZ ---
# 格式：positions_3d[subject][action] = (T, 17, 3)
# 這裡直接存相機空間 + root-relative，不需要 world_to_camera
positions_3d = {
    SUBJECT: {
        ACTION: poses_rel.astype(np.float32)
    }
}
np.savez_compressed(OUTPUT_3D, positions_3d=positions_3d)
print(f"    ✅ 3D NPZ：{OUTPUT_3D}")

# --- 2D NPZ ---
# 格式：positions_2d[subject][action][cam_idx] = (T, 17, 2)
positions_2d = {
    SUBJECT: {
        ACTION: [poses_2d_norm.astype(np.float32)]  # 單相機，index 0
    }
}
metadata = {
    'num_joints': 17,
    'keypoints_symmetry': [
        [4, 5, 6, 11, 12, 13],   # left
        [1, 2, 3, 14, 15, 16]    # right
    ]
}
np.savez_compressed(OUTPUT_2D, positions_2d=positions_2d, metadata=metadata)
print(f"    ✅ 2D NPZ：{OUTPUT_2D}")

print("\n" + "="*60)
print("轉換完成！")
print("="*60)
