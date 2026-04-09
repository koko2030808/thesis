import numpy as np
import os

def v99_final_graduation_qualifier():
    """
    [2026-01-29 19:45] 莊家畢業論文：量化重投影終極定影 (V99)
    環境：Standalone Python (CMD / Conda)
    邏輯：公尺修正 + Y-Mirror + Yaw -90 + 相機矩陣 + 螢幕 Y 翻轉
    """
    print("--- 正在啟動 V99 終極量化校準：物理真相解鎖 ---")

    # 1. 內參定影 (H3.6M Cam4 官方數據)
    f_x, f_y = 1145.5113, 1144.7739
    c_x, c_y = 514.9682, 501.8820
    
    # 2. 外參定影 (UE5 座標與相機 Yaw) [cite: 2026-01-29]
    # 位置：鏡像修正後的 cm 座標 (Cam4)
    cam_pos = np.array([382.7099, 169.6043, 159.1412])
    
    # 相機 Yaw 203.9 度 (Look-at)。為將點轉入相機空間，需取逆旋轉 (-203.9) [cite: 2026-01-29]
    cam_yaw_rad = np.radians(-203.9) 
    c_c, s_c = np.cos(cam_yaw_rad), np.sin(cam_yaw_rad)
    # 相機旋轉矩陣 (繞 Z 軸旋轉)
    R_cam = np.array([
        [c_c, -s_c, 0],
        [s_c,  c_c, 0],
        [0,    0,   1]
    ])

    # 3. 數據路徑掃描 [cite: 2026-01-29]
    data_path = r'D:\videopose2\VideoPose3D\data\data_3d_h36m.npz'
    if os.path.exists(data_path):
        d3d = np.load(data_path, allow_pickle=True)['positions_3d'].item()
        # 獲取 S9 Sitting 第 0 幀 (32 關節原始定義)
        raw_pose = d3d['S9']['Sitting'][0] 
        print(f"✅ 成功載入 S9 數據集。")
    else:
        print("⚠️ 未偵測到路徑，使用 V98 Output 樣本進行驗證...")
        raw_pose = np.zeros((32, 3))
        raw_pose[14] = np.array([-0.0697, -0.1026, 1.6027]) # Right Shoulder
        raw_pose[0]  = np.array([-0.0737, -0.0753, 0.9906]) # Hip

    def project_to_absolute_pixel(p_h36m):
        # A. 坐標換算 (Mirror + CM 轉換) [cite: 2026-01-29]
        # UE_X = -H_Y, UE_Y = -H_X, UE_Z = H_Z
        p_ue = np.array([
            -p_h36m[1] * 100.0, 
            -p_h36m[0] * 100.0, 
            p_h36m[2] * 100.0
        ])
        
        # B. 人物 Yaw -90 補償 [cite: 2026-01-29]
        # 模擬面向 +Y 的資產轉向世界正前方 (+X)
        theta_p = np.radians(-90.0)
        cp, sp = np.cos(theta_p), np.sin(theta_p)
        p_rot = np.array([
            p_ue[0] * cp - p_ue[1] * sp,
            p_ue[0] * sp + p_ue[1] * cp,
            p_ue[2]
        ])
        
        # C. 視圖空間轉換 (View Space) [cite: 2026-01-27, 2026-01-29]
        # 1. Translation: 移動到相機原點
        p_local = p_rot - cam_pos
        # 2. Rotation: 旋轉對齊相機視軸
        p_view = R_cam.dot(p_local)
        
        # D. 最終投影 (修正 V 軸翻轉) [cite: 2026-01-29]
        # 在相機空間，X 是深度，Y 是水平位移，Z 是垂直位移
        # u = (fx * Y / X) + cx
        u = (f_x * (p_view[1] / p_view[0])) + c_x 
        # v = cy - (fy * Z / X) -> 實現數字越大越靠下 (0,0 在左上)
        v = c_y - (f_y * (p_view[2] / p_view[0])) 
        return u, v

    # 4. 驗證產出 [cite: 2026-01-29]
    targets = {14: "Right Shoulder", 11: "Left Shoulder", 0: "Hip"}
    print("-" * 75)
    print(f"{'Joint Name':<15} | {'H36M 3D (meters)':<30} | {'UE5 Pixel (u, v)':<20}")
    print("-" * 75)
    
    for idx, name in targets.items():
        u, v = project_to_absolute_pixel(raw_pose[idx])
        print(f"{name:<15} | {str(raw_pose[idx]):<30} | ({u:.2f}, {v:.2f})")
    
    print("-" * 75)
    print("🚀 最終驗證：標註這些點在 image_29570c.png 上，若重合則畢業！")

if __name__ == "__main__":
    v99_final_graduation_qualifier()