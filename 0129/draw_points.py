import numpy as np
import cv2
import os

def v138_rigorous_symmetry_diagnostic():
    """
    [2026-01-30] 莊家畢業論文：1:1 物理主權定影 (V138)
    修正：精確旋轉矩陣計算 + 181cm 縮放對標 + Ch01 旋轉補償
    """
    print("--- 正在啟動 V138 終極定影：排除 Sitting 數據姿勢雜訊 ---")

    # 1. 物理內參 (鎖死 41.22mm 焦距與畫布中心) [cite: 2026-01-30]
    W, H = 1000, 1000
    f_pixel = 1145.043 # 對應 41.2215mm
    c_x, c_y = 500.0, 500.0 # 對齊 UE5 渲染中心

    # 2. 空間外參 (Cam4 基準位姿)
    cam_pos = np.array([382.7099, 169.6043, 159.1412])
    
    # 【核心修正】精確計算相機旋轉矩陣 (對應 Pitch -8.71, Yaw -156.1)
    yaw_rad = np.radians(-156.100006)
    pitch_rad = np.radians(-8.71)
    
    cy, sy = np.cos(yaw_rad), np.sin(yaw_rad)
    cp, sp = np.cos(pitch_rad), np.sin(pitch_rad)
    
    # UE5 坐標系旋轉矩陣 R (Pitch -> Yaw)
    R = np.array([
        [cp*cy,  cp*sy,  sp],
        [-sy,    cy,     0],
        [-sp*cy, -sp*sy, cp]
    ])

    # 3. 測試數據定義：與 Ch01 181cm 匹配的標準 T-Pose [cite: 2026-01-27, 2026-01-30]
    # 莊家注意：我們手動定義這些點，是為了排除 Sitting 數據姿勢不對的變因
    tpose_joints = {
        "Head_Top":       np.array([0.0,  0.0,  1.81]), # 181cm 頭頂
        "Right_Shoulder": np.array([0.0, -0.21, 1.55]), # 肩膀高度 155cm, 寬度 21cm
        "Left_Shoulder":  np.array([0.0,  0.21, 1.55]), 
        "Hip_Center":     np.array([0.0,  0.0,  0.95])  # 髖部高度 95cm
    }

    # 4. 讀取真理截圖
    img_path = r'c:\GPS_Gaussian\Saved\Screenshots\WindowsEditor\HighresScreenshot00004.png'
    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ 錯誤：找不到路徑 {img_path}")
        return

    def project_v138(p_m):
        """
        [第一性原理] 坐標轉換協議 [cite: 2020-01-20]
        """
        # A. 數據到世界坐標映射 (單位: cm) [cite: 2026-01-30]
        p_ue = p_m * 100.0
        
        # B. 【資產補償】Ch01 Yaw -90 旋轉修正
        # 這能解決「左右肩膀」縮成一團的問題，將數據旋轉到資產面向的方向
        theta_asset = np.radians(-90.0)
        ca, sa = np.cos(theta_asset), np.sin(theta_asset)
        p_rot = np.array([
            p_ue[0]*ca - p_ue[1]*sa,
            p_ue[0]*sa + p_ue[1]*ca,
            p_ue[2]
        ])
        
        # C. 轉換至相機空間 (Translation -> Rotation) [cite: 2026-01-30]
        p_cam = R.dot(p_rot - cam_pos)
        
        # D. 最終投影公式
        u = c_x - (f_pixel * (p_cam[1] / p_cam[0]))
        v = c_y - (f_pixel * (p_cam[2] / p_cam[0]))
        return int(u), int(v)

    # 5. 繪製輸出
    print("-" * 65)
    for name, p_3d in tpose_joints.items():
        u, v = project_v138(p_3d)
        cv2.circle(img, (u, v), 8, (0, 0, 255), -1) 
        cv2.circle(img, (u, v), 10, (255, 255, 255), 1)
        cv2.putText(img, f"{name}({u},{v})", (u+15, v+5), 0, 0.5, (0, 255, 0), 2)
        print(f"🎯 {name:15} | 181cm 定影座標: ({u}, {v})")
    print("-" * 65)

    cv2.imwrite('S9_V138_RIGOROUS_SYMMETRY_TEST.png', img)
    print("🚀 邏輯診斷完成！如果紅點精確釘入 T-Pose 小人，代表我們已完全鎖定物理主權。")

if __name__ == "__main__":
    v138_rigorous_symmetry_diagnostic()