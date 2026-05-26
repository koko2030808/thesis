import json
import cv2
import numpy as np
import os
import glob
from scipy.spatial.transform import Rotation as R

JSON_DATA_PATH   = r"D:\04.09\VideoPose3D\vaccine_data.json"
JSON_CAMERA_PATH = r"D:\04.09\VideoPose3D\vaccine_camera.json"
RENDER_DIR       = r"D:\04.09\VideoPose3D\Renders"
OUTPUT_DIR       = r"D:\04.09\VideoPose3D\Verified_2D"
IMAGE_WIDTH      = 1000
IMAGE_HEIGHT     = 1000

def find_image_by_frame(directory, frame_idx):
    patterns = [f"*{frame_idx:04d}.png", f"*{frame_idx:04d}.jpeg", f"*{frame_idx:04d}.jpg"]
    for p in patterns:
        matches = glob.glob(os.path.join(directory, p))
        if matches: return matches[0]
    return None

def build_projection_pipeline():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("[*] 啟動 V167 投影管線 (歐拉角修正版)...")

    with open(JSON_DATA_PATH, 'r', encoding='utf-8') as f:
        raw = json.load(f)["data"]
    pose_frames = raw[0] if isinstance(raw[0], list) and isinstance(raw[0][0], list) else raw

    with open(JSON_CAMERA_PATH, 'r', encoding='utf-8') as f:
        cam_frames = json.load(f)["data"]

    for frame_idx, (frame_pose, frame_cam) in enumerate(zip(pose_frames, cam_frames)):
        img_path = find_image_by_frame(RENDER_DIR, frame_idx)
        if not img_path: continue
        img = cv2.imread(img_path)
        if img is None: continue

        cam_loc = frame_cam["location"]
        cam_rot = frame_cam["rotation"]
        fov_h   = frame_cam["fov"]

        # 焦距與光心
        fx = (IMAGE_WIDTH / 2.0) / np.tan(np.radians(fov_h) / 2.0)
        cx, cy = IMAGE_WIDTH / 2.0, IMAGE_HEIGHT / 2.0

        # 【關鍵數學修正】：UE5 (左) 轉 OpenCV (右)，Pitch 與 Roll 取負號
        r_obj = R.from_euler('ZYX', [cam_rot[2], -cam_rot[1], -cam_rot[0]], degrees=True)
        
        for i, joint in enumerate(frame_pose):
            p_world = np.array(joint, dtype=np.float64)
            is_zero = (p_world == 0).all()

            p_trans = p_world - np.array(cam_loc)
            p_local = r_obj.apply(p_trans, inverse=True)

            cv_x, cv_y, cv_z = p_local[1], -p_local[2], p_local[0]

            if cv_z <= 0: continue

            u = int((fx * cv_x / cv_z) + cx)
            v = int((fx * cv_y / cv_z) + cy)

            if 0 <= u < IMAGE_WIDTH and 0 <= v < IMAGE_HEIGHT:
                # 正常綠點，異常紅點
                color = (0, 0, 255) if is_zero else (0, 255, 0)
                cv2.circle(img, (u, v), 6, color, -1)
                cv2.putText(img, str(i), (u+8, v-8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        out_path = os.path.join(OUTPUT_DIR, f"Fixed_{os.path.basename(img_path)}")
        cv2.imwrite(out_path, img)
        print(f"[*] Frame {frame_idx:02d} 投影完畢 → {out_path}")

if __name__ == "__main__":
    build_projection_pipeline()