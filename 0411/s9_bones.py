import numpy as np
import os
import sys

# 將父目錄加入搜尋路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.h36m_dataset import Human36mDataset

def extract_s9_and_calculate_torso(npz_path=r'D:\04.09\VideoPose3D\data\data_3d_h36m.npz'):
    try:
        dataset = Human36mDataset(npz_path)
    except Exception as e:
        print(f"【錯誤】載入失敗: {e}")
        return

    subject = 'S9'
    available_actions = list(dataset[subject].keys())
    target_action = 'Walking' if 'Walking' in available_actions else available_actions[0]
    
    # 抓取 S9 第一幀的 17 個點 (單位預設通常是公尺)
    frame_0 = dataset[subject][target_action]['positions'][0] 
    
    def calc_dist(j1, j2):
        return np.linalg.norm(frame_0[j1] - frame_0[j2]) * 1000 # m 轉 mm
    
    print(f"=== S9 基準骨長精算 (Action: {target_action}) ===")
    
    # --- 原本的四肢運算 ---
    bone_lengths = {
        "Right Thigh (右大腿)": calc_dist(1, 2),
        "Right Calf  (右小腿)": calc_dist(2, 3),
        "Left Thigh  (左大腿)": calc_dist(4, 5),
        "Left Calf   (左小腿)": calc_dist(5, 6),
        "Left Up-Arm (左上臂)": calc_dist(11, 12),
        "Left Forearm(左下臂)": calc_dist(12, 13),
        "Right Up-Arm(右上臂)": calc_dist(14, 15),
        "Right Forearm(右下臂)": calc_dist(15, 16)
    }
    for bone, length in bone_lengths.items():
        print(f"{bone}: {length:.2f} mm")

    # ========================================================
    # === V167 新增：軀幹均分藥劑提取器 ======================
    # ========================================================
    s9_torso_length = calc_dist(0, 8)
    mixamo_torso_length = 513.0
    torso_ratio = s9_torso_length / mixamo_torso_length
    
    print("\n=========================================")
    print(f"【S9 軀幹總長】: {s9_torso_length:.2f} mm")
    print(f"【Mixamo 軀幹總長】: {mixamo_torso_length:.2f} mm")
    print(f"👉 【軀幹均分倍率】: {torso_ratio:.4f}")
    print("=========================================")

    # ========================================================
    # === 終極版：中軸線絕對真理萃取 (Ground Truth) ===
    # ========================================================
    pelvis = frame_0[0]
    
    # H36M 數據通常是公尺 (m) 為單位，乘以 1000 轉為 mm
    # 由於 VideoPose3D 的座標系可能 Y 軸朝下或 Z 軸朝上，兩者皆抽出供判讀
    pelvis_height_z = abs(pelvis[2]) * 1000
    pelvis_height_y = abs(pelvis[1]) * 1000

    d_0_7 = calc_dist(0, 7)
    d_7_8 = calc_dist(7, 8)
    d_8_9 = calc_dist(8, 9)

    print("\n=== S9 中軸線絕對真理 (mm) ===")
    print(f"1. Pelvis (00) 到地面高度 (若 Z 為高): {pelvis_height_z:.2f} mm")
    print(f"   Pelvis (00) 到地面高度 (若 Y 為高): {pelvis_height_y:.2f} mm")
    print(f"2. Pelvis (00) 到 Spine (07) 絕對長度: {d_0_7:.2f} mm")
    print(f"3. Spine (07) 到 Thorax (08) 絕對長度: {d_7_8:.2f} mm")
    print(f"4. Thorax (08) 到 Nose (09) 絕對長度:  {d_8_9:.2f} mm")
    print("================================")

    # 5. 肩膀寬度 (Left Shoulder 11 到 Right Shoulder 14)
    shoulder_width = calc_dist(11, 14)

    # 6. 骨盆寬度 (Right Hip 1 到 Left Hip 4)
    hip_width = calc_dist(1, 4)

    print(f"5. S9 絕對肩寬 (11 to 14): {shoulder_width:.2f} mm")
    print(f"6. S9 絕對骨盆寬 (1 to 4): {hip_width:.2f} mm")

    # 7. 骨盆到大腿的 3D 相對向量 (Node 1 減去 Node 0)
    vec_0_to_1 = (frame_0[1] - frame_0[0]) * 1000
    
    print("\n=== 骨盆三角絕對向量 (mm) ===")
    print(f"7. Pelvis(0) 到 R_Hip(1) 的 XYZ 向量: {vec_0_to_1}")
    print("================================")
    # 8. Nose (09) 到 Head (10) 的 3D 相對向量與絕對長度
    vec_9_to_10 = (frame_0[10] - frame_0[9]) * 1000
    length_9_to_10 = np.linalg.norm(vec_9_to_10) # 計算純量長度
    
    print(f"8. Nose (09) 到 Head (10) 絕對長度: {length_9_to_10:.2f} mm")
    print(f"   (XYZ 向量: {vec_9_to_10})")

if __name__ == '__main__':
    extract_s9_and_calculate_torso()