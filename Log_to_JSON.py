import os
import re
import json

# 1. 定義路徑 (已對齊您的最新 C 槽環境)
log_path = r"C:\GPS_Gaussian\Saved\Logs\GPS_Gaussian.log"
output_json_path = r"D:\04.09\VideoPose3D\vaccine_data.json"

def clean_and_extract_log():
    if not os.path.exists(log_path):
        print(f"【錯誤】找不到 Log 檔案: {log_path}")
        return

    frames_data = []
    
    # 2. 讀取並正則清洗
    with open(log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    for line in lines:
        # 鎖定特徵：我們在藍圖中用逗號組合的字串
        if "X=" in line and "Y=" in line and "Z=" in line:
            # 剃除 Log 前方的時間戳記與引擎雜訊，只保留座標字串
            clean_str = re.sub(r'^.*?LogBlueprintUserMessages: \[.*?\]\s*', '', line).strip()
            
            # 切割 17 個點
            points = clean_str.split(',')
            frame_points = []
            
            for p in points:
                # 萃取浮點數
                coords = re.findall(r'-?\d+\.\d+', p)
                if len(coords) == 3:
                    # 轉為浮點數陣列 [X, Y, Z]
                    frame_points.append([float(coords[0]), float(coords[1]), float(coords[2])])
            
            if len(frame_points) > 0:
                frames_data.append(frame_points)

    # 3. 陣列防呆截斷與輸出標準 JSON
    if frames_data:
        # 【物理截斷】：強制只取陣列最後的 11 幀 (剔除歷史髒緩衝)
        target_frames = 11
        clean_frames_data = frames_data[-target_frames:] 
        
        # VideoPose3D 預期形狀為 (Frames, Joints, 3)
        dataset = {
            "metadata": {
                "num_frames": len(clean_frames_data),
                "num_joints": len(clean_frames_data[0]) if clean_frames_data else 0
            },
            "data": clean_frames_data
        }
        
        with open(output_json_path, 'w', encoding='utf-8') as jf:
            json.dump(dataset, jf, indent=4)
            
        print("========================================")
        print(f"【收割成功】已強制截取最後 {len(clean_frames_data)} 幀純淨數據。")
        print(f"【檔案寫入】{output_json_path}")
        print("========================================")
    else:
        print("【警告】未能在 Log 中找到有效的座標特徵。")

if __name__ == "__main__":
    clean_and_extract_log()