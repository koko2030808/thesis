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
OUTPUT_2D_JSON   = r"D:\04.09\VideoPose3D\vaccine_data_2d.json"
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
    print("[*] 啟動 V167 投影管線...")

    with open(JSON_DATA_PATH, 'r', encoding='utf-8') as f:
        raw = json.load(f)["data"]
    pose_frames = raw[0] if isinstance(raw[0], list) and isinstance(raw[0][0], list) else raw

    with open(JSON_CAMERA_PATH, 'r', encoding='utf-8') as f:
        cam_frames = json.load(f)["data"]

    all_2d_frames = []  # 收集所有幀的 2D 結果

    for frame_idx, (frame_pose, frame_cam) in enumerate(zip(pose_frames, cam_frames)):
        cam_loc = frame_cam["location"]
        cam_rot = frame_cam["rotation"]
        fov_h   = frame_cam["fov"]

        fx = (IMAGE_WIDTH / 2.0) / np.tan(np.radians(fov_h) / 2.0)
        cx, cy = IMAGE_WIDTH / 2.0, IMAGE_HEIGHT / 2.0

        r_obj = R.from_euler('ZYX', [cam_rot[2], -cam_rot[1], -cam_rot[0]], degrees=True)

        frame_2d = []
        for i, joint in enumerate(frame_pose):
            p_world = np.array(joint, dtype=np.float64)
            p_trans = p_world - np.array(cam_loc)
            p_local = r_obj.apply(p_trans, inverse=True)

            cv_x, cv_y, cv_z = p_local[1], -p_local[2], p_local[0]

            if cv_z > 0:
                u = (fx * cv_x / cv_z) + cx
                v = (fx * cv_y / cv_z) + cy
            else:
                u, v = cx, cy  # 預設畫面中心

            frame_2d.append([round(u, 3), round(v, 3)])

        all_2d_frames.append(frame_2d)

        # 視覺化（如果有渲染圖）
        img_path = find_image_by_frame(RENDER_DIR, frame_idx)
        if img_path:
            img = cv2.imread(img_path)
            if img is not None:
                for i, (u, v) in enumerate(frame_2d):
                    u_int, v_int = int(u), int(v)
                    if 0 <= u_int < IMAGE_WIDTH and 0 <= v_int < IMAGE_HEIGHT:
                        cv2.circle(img, (u_int, v_int), 6, (0, 255, 0), -1)
                        cv2.putText(img, str(i), (u_int+8, v_int-8),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                out_path = os.path.join(OUTPUT_DIR, f"Fixed_{os.path.basename(img_path)}")
                cv2.imwrite(out_path, img)

        print(f"[*] Frame {frame_idx:03d} 完成")

    # 儲存 2D JSON
    output = {
        "metadata": {
            "frame_count": len(all_2d_frames),
            "image_width": IMAGE_WIDTH,
            "image_height": IMAGE_HEIGHT,
            "unit": "pixels"
        },
        "data": all_2d_frames
    }
    with open(OUTPUT_2D_JSON, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)

    print(f"\n[✅] vaccine_data_2d.json 已儲存：{OUTPUT_2D_JSON}")
    print(f"[✅] 總幀數：{len(all_2d_frames)}")

if __name__ == "__main__":
    build_projection_pipeline()