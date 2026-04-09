import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def run_v44_chest_plane_alignment(dataset_path):
    # --- Part A: 提取 S9 黃金地基 (H3.6M 右手系) ---
    if not os.path.exists(dataset_path):
        print("【資料不足】路徑錯誤。")
        return
    data = np.load(dataset_path, allow_pickle=True)['positions_3d'].item()
    s9_f = data['S9']['Sitting' if 'Sitting' in data['S9'] else 'Sitting 1'][158]
    idx_17 = [0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27]
    s9_rel = s9_f[idx_17] - s9_f[idx_17][0]

    # --- Part B: 注入 Sovereign 數據 (包含手腕) ---
    mix_raw_sovereign = np.array([
        [-46.7374, -4.0454, 60.3800], [-44.8492, 6.2228, 56.0491], [-3.7043, 17.2525, 51.5998], 
        [-11.3757, 14.8404, 12.9130], [-44.5159, -13.3314, 54.3335], [-2.5485, -21.0703, 50.5121], 
        [-9.7090, -19.0624, 11.8365], [-51.9762, -4.8874, 68.9892], [-53.3201, -4.7290, 80.7097], 
        [-50.9290, -2.7915, 109.1328], [-46.1023, -3.2559, 115.3882], [-49.6589, -22.9677, 101.1248], 
        [-51.4232, -32.6360, 77.7958], [-41.8183, -35.4349, 56.9010], [-51.9814, 16.5695, 98.7787], 
        [-55.3254, 22.8910, 74.4860], [-43.1264, 25.8796, 55.0146]
    ]) / 100.0
    
    # 【主權回歸】：視覺正確映射 [-y, -x, z]
    # 保留 Mixamo 原始動作向量方向
    mix_m = np.stack([-mix_raw_sovereign[:, 1], -mix_raw_sovereign[:, 0], mix_raw_sovereign[:, 2]], axis=1)

    # --- Part C: 【V44 核心修正】胸腔深度主權補償 ---
    # 1. 語義 Thorax (J8) 定義：肩膀中點
    mix_m[8] = (mix_m[11] + mix_m[14]) / 2.0
    
    # 2. 深度補償：將凹陷的肩膀(11,14)與胸口(8)整體推向身體前方 (Y+)
    # 補償 S9 表面感測器平面與 Mixamo 內部骨架支點的 5.5cm 斷層
    chest_plane_indices = [8, 11, 14]
    mix_m[chest_plane_indices] += np.array([0, 0.055, 0]) 

    # --- Part D: 生理重構 (Renormalization) ---
    # 這裡不再執行 V42 的脊椎旋轉，保留原始 hunchback 動作意圖
    mix_v44 = np.zeros_like(mix_m); mix_v44[0] = [0, 0, 0]
    def bridge(s, e, target, source, current):
        v = source[e] - source[s]; u = v / (np.linalg.norm(v) + 1e-8)
        l = np.linalg.norm(target[e] - target[s])
        return current[s] + u * l

    skel = [(0,7), (7,8), (8,9), (9,10), (0,1), (1,2), (2,3), (0,4), (4,5), (5,6), (8,11), (11,12), (12,13), (8,14), (14,15), (15,16)]
    for s, e in skel: mix_v44[e] = bridge(s, e, s9_rel, mix_m, mix_v44)

    mix_rel = mix_v44 - mix_v44[0]
    
    # --- Part E: 最終分區審計 ---
    groups = {"Trunk (地基)": [0,7,8], "Head (頭頸)": [9,10], "Arms (動作差)": [11,12,13,14,15,16], "Legs (下肢)": [1,2,3,4,5,6]}
    total_mpjpe = np.mean(np.linalg.norm(s9_rel - mix_rel, axis=1)) * 1000
    print(f"\n🚀 V44.0 胸腔平面校準版 | 總 MPJPE: {total_mpjpe:.2f} mm")
    print("-" * 65)
    for name, nodes in groups.items():
        err = np.mean(np.linalg.norm(s9_rel[nodes] - mix_rel[nodes], axis=1)) * 1000
        print(f"📌 {name:<15} : {err:>8.2f} mm")
    print("-" * 65)

    # --- Part F: 視覺化審計 ---
    fig = plt.figure(figsize=(10, 10)); ax = fig.add_subplot(111, projection='3d')
    for s, e in skel:
        ax.plot([s9_rel[s,0], s9_rel[e,0]], [s9_rel[s,1], s9_rel[e,1]], [s9_rel[s,2], s9_rel[e,2]], 'b-', alpha=0.3, label='S9 GT' if s==0 else "")
        ax.plot([mix_rel[s,0], mix_rel[e,0]], [mix_rel[s,1], mix_rel[e,1]], [mix_rel[s,2], mix_rel[e,2]], 'r-', linewidth=2, label='V44 Sovereign' if s==0 else "")
    
    limit = 0.8; ax.set_xlim(-limit, limit); ax.set_ylim(-limit, limit); ax.set_zlim(-0.2, 1.2)
    ax.set_box_aspect([1, 1, 1]); plt.legend(); plt.show()
    
    # 保存這組最具「物理一致性」的標定結果
    np.save('v44_sitting_visual_optimum.npy', s9_rel - mix_rel)

if __name__ == "__main__":
    run_v44_chest_plane_alignment(r'D:\videopose2\VideoPose3D\data\data_3d_h36m.npz')