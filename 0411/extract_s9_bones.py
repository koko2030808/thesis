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
    
    # 抓取 S9 第一幀的 17 個點
    frame_0 = dataset[subject][target_action]['positions'][0] 
    
    def calc_dist(j1, j2):
        return np.linalg.norm(frame_0[j1] - frame_0[j2]) * 1000 # m 轉 mm
    
    print(f"=== S9 基準骨長精算 (Action: {target_action}) ===")
    
    # --- 原本的四肢運算 (保留給你參考) ---
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
    
    # 1. 計算 S9 (H36M) 的軀幹總長：骨盆(Node 0) 到 脖子底端(Node 8)
    s9_torso_length = calc_dist(0, 8)
    
    # 2. 定義 Mixamo 的軀幹總長 (UE5 實測 Hips 到 Neck 大約為 513 mm)
    mixamo_torso_length = 513.0
    
    # 3. 算出均分倍率
    torso_ratio = s9_torso_length / mixamo_torso_length
    
    print("\n=========================================")
    print(f"【S9 軀幹總長】: {s9_torso_length:.2f} mm")
    print(f"【Mixamo 軀幹總長】: {mixamo_torso_length:.2f} mm")
    print(f"👉 【軀幹均分藥劑】: {torso_ratio:.4f}")
    print("=========================================")
    print("請將此數字填入 UE5 的 Spine, Spine1, Spine2 的 Scale Y 欄位！")
    # === V167 最終版：插槽絕對偏移量計算器 ===
    # 假設 frame_0 已經抓到了 17 個點
    
    print("\n=========================================")
    print("【虛擬插槽 (Socket) 絕對偏移量數據】")
    print("=========================================")
    
    # 1. 中段 Spine (Node 7) 距離骨盆 (Node 0) 有多高？
    dist_spine = calc_dist(0, 7)
    print(f"👉 H36M_07_Spine 距離骨盆高度 (Y軸): {dist_spine:.2f} mm (約為 UE5 的 {dist_spine/10:.2f} 單位)")

    # 2. 左肩 (Node 11) 距離胸腔中心 (Node 8) 有多遠？
    dist_l_shoulder = calc_dist(8, 11)
    print(f"👉 H36M_11_L_Shoulder 距離胸腔中心: {dist_l_shoulder:.2f} mm (約為 UE5 的 {dist_l_shoulder/10:.2f} 單位)")

    # 3. 右肩 (Node 14) 距離胸腔中心 (Node 8) 有多遠？
    dist_r_shoulder = calc_dist(8, 14)
    print(f"👉 H36M_14_R_Shoulder 距離胸腔中心: {dist_r_shoulder:.2f} mm (約為 UE5 的 {dist_r_shoulder/10:.2f} 單位)")
    
    # 4. 脖子/鼻子 (Node 9) 距離胸腔中心 (Node 8) 有多高？
    dist_neck = calc_dist(8, 9)
    print(f"👉 H36M_09_Neck 距離胸腔中心: {dist_neck:.2f} mm (約為 UE5 的 {dist_neck/10:.2f} 單位)")

    

if __name__ == '__main__':
    extract_s9_and_calculate_torso()