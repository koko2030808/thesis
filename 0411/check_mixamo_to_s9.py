import json, math

def verify_s9_tensors_single():
    file_path = "C:/Users/Public/H36M_Sequence_10Frames.json"
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        print("=== S9 幾何張量驗證報告 (單幀核准) ===")
        print(f"[維度檢查] 目前總幀數: {len(data)} 幀")
        
        # 直接抓取第 0 幀
        frame_0 = data[0] 
        
        bones_to_check = {
            "骨盆至脊椎 (Pelvis -> Spine)": (0, 7),
            "右大腿長度 (R_Hip -> R_Knee)": (1, 2),
            "右小腿長度 (R_Knee -> R_Ankle)": (2, 3),
            "左大腿長度 (L_Hip -> L_Knee)": (4, 5),
            "左小腿長度 (L_Knee -> L_Ankle)": (5, 6),
        }
        
        for name, (idx1, idx2) in bones_to_check.items():
            x1, y1, z1 = frame_0[idx1]
            x2, y2, z2 = frame_0[idx2]
            dist = math.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)
            print(f"- {name}: {dist:.2f} cm")

    except Exception as e:
        print(f"錯誤: {e}")

verify_s9_tensors_single()