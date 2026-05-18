import json, math

def verify_s9_tensors():
    file_path = "C:/Users/Public/H36M_Sequence_10Frames.json"
    
    try:
        # 1. 讀取 JSON 檔案
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        print("=== S9 張量資料集驗證報告 ===")
        print(f"[維度檢查] 總幀數: {len(data)} 幀")
        print(f"[維度檢查] 單幀節點數: {len(data[0])} 點")
        
        # 取第一幀來驗證幾何比例
        frame_0 = data[0] 
        
        # 2. 定義要驗證的關鍵骨骼 (對應 17 點陣列的 Index)
        bones_to_check = {
            "骨盆至脊椎 (Pelvis -> Spine)": (0, 7),
            "右大腿長度 (R_Hip -> R_Knee)": (1, 2),
            "右小腿長度 (R_Knee -> R_Ankle)": (2, 3),
            "左大腿長度 (L_Hip -> L_Knee)": (4, 5),
            "左小腿長度 (L_Knee -> L_Ankle)": (5, 6),
            "右上手臂 (R_Shoulder -> R_Elbow)": (14, 15)
        }
        
        print("\n=== 幾何特徵驗證 (Frame 0) ===")
        for name, (idx1, idx2) in bones_to_check.items():
            # 取出兩個點的 XYZ
            x1, y1, z1 = frame_0[idx1]
            x2, y2, z2 = frame_0[idx2]
            
            # 計算歐幾里得距離
            dist = math.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)
            print(f"- {name}: {dist:.2f} cm")
            
        # 3. 時序動態檢查 (比較第 0 幀與第 9 幀的骨盆位置)
        p0 = data[0][0] # 第 0 幀的 Pelvis
        p9 = data[9][0] # 第 9 幀的 Pelvis
        movement = math.sqrt((p0[0]-p9[0])**2 + (p0[1]-p9[1])**2 + (p0[2]-p9[2])**2)
        print("\n=== 時序動態驗證 ===")
        print(f"- 10幀內骨盆位移量: {movement:.2f} cm")
        if movement > 0.1:
            print("=> [PASS] 確認為動態序列，非靜止死圖。")
        else:
            print("=> [WARNING] 位移極小，請確認匯入的動畫是否有動作。")

    except FileNotFoundError:
        print(f"錯誤：找不到檔案 {file_path}")

verify_s9_tensors()