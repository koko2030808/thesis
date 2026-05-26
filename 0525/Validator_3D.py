import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

def verify_3d_skeleton():
    # ============================================================
    # 1. 資產路徑配置 (Asset Configuration)
    # ============================================================
    json_path = r"D:\04.09\VideoPose3D\vaccine_data.json"
    
    if not os.path.exists(json_path):
        print(f"[!] 致命錯誤：找不到檔案 {json_path}")
        return

    print(f"[*] 啟動 3D 物理觀測器，載入目標：{json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
        
    # ============================================================
    # 2. 維度降解與防禦 (Dimensional Degradation & Defense)
    # ============================================================
    frames = raw_data.get("data", [])
    if not frames:
        print("[!] 致命錯誤：找不到 'data' 欄位。")
        return
        
    # 防禦機制：剝除多餘的俄羅斯娃娃嵌套 (List inside List)
    if isinstance(frames[0], list) and isinstance(frames[0][0], list):
         print("[*] 系統防禦：偵測到多重維度嵌套，啟動強制降維處理...")
         frames = frames[0]

    # 提取第 0 幀
    frame_0 = frames[0]
    
    # 防呆檢測：確認關節數量為 17
    if len(frame_0) != 17:
        print(f"[!] 警告：偵測到的關節數量為 {len(frame_0)}，預期為 17。")

    print("[*] 成功提取第一幀 3D 骨骼矩陣。")

    # ============================================================
    # 3. 陣列廣播與空間映射 (Array Broadcasting & Spatial Mapping)
    # ============================================================
    # 強制轉換為 float 浮點數陣列，確保後續運算精度
    xs = np.array([joint[0] for joint in frame_0], dtype=float)
    ys = np.array([joint[1] for joint in frame_0], dtype=float)
    zs = np.array([joint[2] for joint in frame_0], dtype=float)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 繪製關節點 (紅色圓圈)
    ax.scatter(xs, ys, zs, c='r', marker='o', s=50, label='Joints')

    # H36M 拓樸連線 (骨架)
    skeleton_connections = [
        (0,1), (1,2), (2,3),         # 右腿
        (0,4), (4,5), (5,6),         # 左腿
        (0,7), (7,8), (8,9), (9,10), # 脊椎到頭部 (Thorax 位於索引 8)
        (8,11), (11,12), (12,13),    # 左手
        (8,14), (14,15), (15,16)     # 右手
    ]
    
    print("[*] 開始重建幾何拓樸連線...")
    for connection in skeleton_connections:
        try:
            i, j = connection
            ax.plot([xs[i], xs[j]], [ys[i], ys[j]], [zs[i], zs[j]], 'b-', linewidth=2)
        except IndexError:
            pass

    # ============================================================
    # 4. 物理等比約束演算法 (Isometric Cubic Bounding Box)
    # ============================================================
    print("[*] 注入架構師等比例約束 (Isometric Constraints)...")
    # 找出三軸中最大的跨度，作為正方體的半徑
    max_range = np.array([xs.max()-xs.min(), ys.max()-ys.min(), zs.max()-zs.min()]).max() / 2.0
    
    # 算出三軸的中心點
    mid_x = (xs.max()+xs.min()) * 0.5
    mid_y = (ys.max()+ys.min()) * 0.5
    mid_z = (zs.max()+zs.min()) * 0.5
    
    # 劃定嚴格的正立方體邊界，拒絕哈哈鏡變形
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # 隱藏外框比例尺，強制視覺為 1:1:1
    ax.set_box_aspect([1, 1, 1])

    # ============================================================
    # 5. UI 與輸出
    # ============================================================
    ax.set_title("V167 Sovereignty OS - 3D Spatial Validation (Isometric)")
    ax.set_xlabel('X Axis (cm)')
    ax.set_ylabel('Y Axis (cm)')
    ax.set_zlabel('Z Axis (cm)')
    
    print("====================================================")
    print(" ✅ 渲染就緒，請於彈出視窗觀測 3D 骨架。")
    print("====================================================")
    plt.show()

if __name__ == "__main__":
    verify_3d_skeleton()