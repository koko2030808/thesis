import json
import cv2
import numpy as np
import os
import glob
from scipy.spatial.transform import Rotation as R

# ============================================================
# 配置區：V167 量產級特徵榨取器
# ============================================================
JSON_DATA_PATH   = r"D:\04.09\VideoPose3D\vaccine_data.json"
JSON_CAMERA_PATH = r"D:\04.09\VideoPose3D\vaccine_camera.json"
RENDER_DIR       = r"D:\04.09\VideoPose3D\Renders"
OUTPUT_DIR       = r"D:\04.09\VideoPose3D\Verified_2D"
OUTPUT_2D_JSON   = r"D:\04.09\VideoPose3D\vaccine_data_2d.json"

IMAGE_WIDTH      = 1000
IMAGE_HEIGHT     = 1000

# 【產線控制閥】：量產時務必將 RENDER_IMAGES 設為 False 關閉磁碟 I/O
RENDER_IMAGES    = False
EXPORT_2D_JSON   = True

def find_image_by_frame(directory, frame_idx):
    patterns = [f"*{frame_idx:04d}.png", f"*{frame_idx:04d}.jpeg", f"*{frame_idx:04d}.jpg"]
    for p in patterns:
        matches = glob.glob(os.path.join(directory, p))
        if matches: return matches[0]
    return None

def build_projection_pipeline():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"[*] 啟動 V167 量產管線 | 繪圖模式: {RENDER_IMAGES} | 輸出 JSON: {EXPORT_2D_JSON}")

    with open(JSON_DATA_PATH, 'r', encoding='utf-8') as f:
        raw = json.load(f)["data"]
    pose_frames = raw[0] if isinstance(raw[0], list) and isinstance(raw[0][0], list) else raw

    with open(JSON_CAMERA_PATH, 'r', encoding='utf-8') as f:
        cam_frames = json.load(f)["data"]

    # 建立 2D 關鍵點資料集容器
    dataset_2d = []

    for frame_idx, (frame_pose, frame_cam) in enumerate(zip(pose_frames, cam_frames)):
        
        # 繪圖前置準備
        img = None
        if RENDER_IMAGES:
            img_path = find_image_by_frame(RENDER_DIR, frame_idx)
            if img_path: 
                img = cv2.imread(img_path)

        cam_loc = frame_cam["location"]
        cam_rot = frame_cam["rotation"]
        fov_h   = frame_cam["fov"]

        fx = (IMAGE_WIDTH / 2.0) / np.tan(np.radians(fov_h) / 2.0)
        cx, cy = IMAGE_WIDTH / 2.0, IMAGE_HEIGHT / 2.0

        r_obj = R.from_euler('ZYX', [cam_rot[2], -cam_rot[1], -cam_rot[0]], degrees=True)
        
        # 當前幀的 17 個 2D 點 (預設為 0, 0 表示無效/遮擋)
        frame_2d_pts = np.zeros((17, 2), dtype=np.float64)

        for i, joint in enumerate(frame_pose):
            p_world = np.array(joint, dtype=np.float64)
            is_zero = (p_world == 0).all()

            p_trans = p_world - np.array(cam_loc)
            p_local = r_obj.apply(p_trans, inverse=True)

            cv_x, cv_y, cv_z = p_local[1], -p_local[2], p_local[0]

            if cv_z <= 0: continue

            u = (fx * cv_x / cv_z) + cx
            v = (fy * cv_y / cv_z) + cy

            # 座標存入記憶體陣列
            frame_2d_pts[i] = [round(u, 3), round(v, 3)]

            # 繪圖邏輯 (僅在 RENDER_IMAGES = True 時觸發)
            if RENDER_IMAGES and img is not None:
                if 0 <= u < IMAGE_WIDTH and 0 <= v < IMAGE_HEIGHT:
                    color = (0, 0, 255) if is_zero else (0, 255, 0)
                    cv2.circle(img, (int(u), int(v)), 6, color, -1)
                    cv2.putText(img, str(i), (int(u)+8, int(v)-8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        dataset_2d.append(frame_2d_pts.tolist())

        if RENDER_IMAGES and img is not None:
            out_path = os.path.join(OUTPUT_DIR, f"Fixed_{os.path.basename(img_path)}")
            cv2.imwrite(out_path, img)

        # 量產模式下，每 500 幀印出一次進度以監控產線
        if frame_idx % 500 == 0:
            print(f"[*] 已處理 {frame_idx} 幀...")

    # 最終序列化導出
    if EXPORT_2D_JSON:
        output_payload = {
            "metadata": {
                "format": "2D_Keypoints_VideoPose3D_Input",
                "resolution": [IMAGE_WIDTH, IMAGE_HEIGHT]
            },
            "data": [dataset_2d]
        }
        with open(OUTPUT_2D_JSON, 'w', encoding='utf-8') as f:
            json.dump(output_payload, f, separators=(',', ':')) # 關閉縮排壓縮體積
        print(f"[*] ✅ 2D 特徵序列化完成，已匯出至：{OUTPUT_2D_JSON}")

if __name__ == "__main__":
    build_projection_pipeline()