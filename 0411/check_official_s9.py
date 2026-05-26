import numpy as np
import math

def verify_official_s9_aligned():
    dataset_path = r'D:\04.09\VideoPose3D\data\data_3d_h36m.npz' 
    
    try:
        print("=== 載入 H36M 官方資料集 (實施源碼級降維) ===")
        dataset = np.load(dataset_path, allow_pickle=True)['positions_3d'].item()
        action_name = list(dataset['S9'].keys())[0]
        
        # 1. 抓取原始 32 節點 (單位轉為公分)
        s9_32_joints = dataset['S9'][action_name][0] * 100.0
        
        # 2. 核心對齊：複製 h36m_dataset.py 的剔除邏輯
        removed_indices = [4, 5, 9, 10, 11, 16, 20, 21, 22, 23, 24, 28, 29, 30, 31]
        kept_indices = [i for i in range(32) if i not in removed_indices]
        
        # 3. 降維壓縮 (產生與 UE5 完全同構的 17 節點張量)
        s9_17_joints = s9_32_joints[kept_indices]
        
        print(f"\n=== 官方 S9 核心 17 節點數據 (動作: {action_name}) ===")
        
        # 🚨 終極對齊：這裡的 Index 已經與您 UE5 的 0~16 完美同步！
        bones = {
            "骨盆至脊椎 (Pelvis -> Spine)": (0, 7),
            "右大腿長度 (R_Hip -> R_Knee)": (1, 2),
            "右小腿長度 (R_Knee -> R_Ankle)": (2, 3),
            "左大腿長度 (L_Hip -> L_Knee)": (4, 5),
            "左小腿長度 (L_Knee -> L_Ankle)": (5, 6),
        }
        
        for name, (idx1, idx2) in bones.items():
            p1 = s9_17_joints[idx1]
            p2 = s9_17_joints[idx2]
            dist = math.sqrt(sum((a - b)**2 for a, b in zip(p1, p2)))
            print(f"- {name}: {dist:.2f} cm")
            
    except FileNotFoundError:
        print(f"\n[錯誤] 找不到檔案: {dataset_path}")

verify_official_s9_aligned()