import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def run_v44_final_audit(h36m_path, test_mode='T-POSE'):
    print(f"\n>>> 啟動 V44.4 終極審計 | 模式: {test_mode} | 鎖定物理比例 1:1:1...")
    
    data = np.load(h36m_path, allow_pickle=True)['positions_3d'].item()
    h36m_idx = [0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27]
    
    if test_mode == 'T-POSE':
        s9_gt = data['S9']['Sitting 1'][0][h36m_idx]
        # 【相位修正】：解決「面對面」問題，執行 180 度 Z 軸旋轉
        # 原本 [x, y, z] 是面對面，現在用 [-x, -y, z] 轉回來
        mix_raw = np.array([
            [0.0, 2.6, 99.1], [-9.8, 2.3, 93.5], [-12.3, -0.5, 50.9], [-14.7, -5.0, 11.7], 
            [9.8, 2.3, 93.5], [12.3, -0.7, 50.9], [14.7, -3.8, 11.7], [0.0, 0.2, 120.9], 
            [0.0, -1.2, 134.3], [0.0, -2.8, 149.4], [0.0, -0.01, 156.8], [21.0, -2.9, 142.9], 
            [46.2, -4.3, 141.3], [69.4, -3.2, 142.2], [-21.0, -2.8, 142.9], [-46.2, -4.3, 141.3], [-69.4, -3.1, 142.2]
        ]) / 100.0
        mix_m = np.stack([-mix_raw[:, 0], -mix_raw[:, 1], mix_raw[:, 2]], axis=1)
    else:
        s9_gt = data['S9']['Sitting 1'][158][h36m_idx]
        mix_raw = np.array([
            [-46.73, -4.04, 60.38], [-44.84, 6.22, 56.04], [-3.70, 17.25, 51.59], [-11.37, 14.84, 12.91], 
            [-44.51, -13.33, 54.33], [-2.54, -21.07, 50.51], [-9.70, -19.06, 11.83], [-51.97, -4.88, 68.98], 
            [-53.32, -4.72, 80.70], [-50.92, -2.79, 109.13], [-46.10, -3.25, 115.38], [-49.65, -22.96, 101.12], 
            [-51.42, -32.63, 77.79], [-41.81, -35.43, 56.90], [-51.98, 16.56, 98.77], [-55.32, 22.89, 74.48], [-43.12, 25.87, 55.01]
        ]) / 100.0
        mix_m = np.stack([-mix_raw[:, 1], -mix_raw[:, 0], mix_raw[:, 2]], axis=1)
        
    s9_gt -= s9_gt[0]

    # --- Part C: 生理重構 (注入 S9 長度) ---
    mix_final = np.zeros_like(mix_m); mix_final[0] = [0, 0, 0]
    def bridge(s, e, target, source, current):
        v = source[e] - source[s]; unit_v = v / (np.linalg.norm(v) + 1e-8)
        length = np.linalg.norm(target[e] - target[s])
        return current[s] + unit_v * length

    skel = [(0,7), (7,8), (8,9), (9,10), (0,1), (1,2), (2,3), (0,4), (4,5), (5,6), (8,11), (11,12), (12,13), (8,14), (14,15), (15,16)]
    for s, e in skel: mix_final[e] = bridge(s, e, s9_gt, mix_m, mix_final)

    # --- Part D: 結構殘差報告 ---
    errors = np.linalg.norm(s9_gt - (mix_final - mix_final[0]), axis=1) * 1000
    print(f"\n📊 結構殘差審計 (mm) | 鎖定方向與比例後")
    print("-" * 50)
    print(f"📍 Thorax (J8) : {errors[8]:.2f} mm")
    print(f"📍 Neck   (J9) : {errors[9]:.2f} mm")
    print("-" * 50)

    # --- Part E: 終極視覺化 (固定比例解決方案) ---
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    for s, e in skel:
        ax.plot([s9_gt[s,0], s9_gt[e,0]], [s9_gt[s,1], s9_gt[e,1]], [s9_gt[s,2], s9_gt[e,2]], 'b-', alpha=0.3, label='S9 GT' if s==0 else "")
        ax.plot([mix_final[s,0], mix_final[e,0]], [mix_final[s,1], mix_final[e,1]], [mix_final[s,2], mix_final[e,2]], 'r-', linewidth=3, label='V44.4 Fixed' if s==0 else "")

    # 強制 1:1:1 比例關鍵代碼
    max_range = 0.7
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-0.2, 1.2) # Z 軸通常不需要對稱，但範圍需與 XY 相當
    ax.set_box_aspect([1, 1, 1]) # 確保 Matplotlib 渲染為正立方體
    
    plt.legend(); plt.title(f"V44.4 Final Audit: {test_mode}"); plt.show()

if __name__ == "__main__":
    h36m_path = r'D:\videopose2\VideoPose3D\data\data_3d_h36m.npz'
    run_v44_final_audit(h36m_path, test_mode='T-POSE')