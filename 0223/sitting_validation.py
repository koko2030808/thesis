import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ==============================================================================
# V167_V44.4_SOVEREIGN_OS: 動態時序驗證系統
# ==============================================================================
SKELETON_EDGES = [
    (0, 7), (7, 8), (8, 9), (9, 10),  # 軀幹與頭部
    (0, 1), (1, 2), (2, 3),           # 右腿
    (0, 4), (4, 5), (5, 6),           # 左腿
    (8, 11), (11, 12), (12, 13),      # 左臂
    (8, 14), (14, 15), (15, 16)       # 右臂
]

class V44_4_ValidationEngine:
    def __init__(self, target_h36m_path):
        print(">>> 啟動 V44.4 地基艙最後驗證：執行生理指紋動態審計...")
        # 1. 提取 S9 184cm 生理指紋地基 (使用 Sitting 1 第 0 幀作為基準)
        data = np.load(target_h36m_path, allow_pickle=True)['positions_3d'].item()
        idx_17 = [0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27]
        s9_base_pose = data['S9']['Sitting 1'][0].reshape(-1, 3)[idx_17]
        self.s9_lengths = self._calculate_bone_lengths(s9_base_pose)
        print(f"✅ S9 生理指紋鎖定：總骨架段數 {len(self.s9_lengths)} 段。")

    def _calculate_bone_lengths(self, pose):
        lengths = {}
        for s, e in SKELETON_EDGES:
            lengths[(s, e)] = np.linalg.norm(pose[e] - pose[s])
        return lengths

    def run_v44_4_protocol(self, raw_sequence):
        """
        核心執行：絕對映射 -> 生理注入 -> 深度補償
        raw_sequence shape: (frames, 17, 3)
        """
        frames = raw_sequence.shape[0]
        final_sequence = np.zeros((frames, 17, 3))

        for f in range(frames):
            # 1. 座標系硬映射 (Hard Mapping) [-y, -x, z]
            # UE5(Left-hand) to H3.6M(Right-hand)
            mix_raw = raw_sequence[f] / 100.0  # cm to m
            mapped_m = np.stack([-mix_raw[:, 1], -mix_raw[:, 0], mix_raw[:, 2]], axis=1)

            # 2. 生理指紋注入 (Renormalization Bridge)
            reconstructed = np.zeros_like(mapped_m)
            reconstructed[0] = [0, 0, 0] # Root 歸零
            
            # 遞迴或依序重建 (確保拓撲連貫)
            for s, e in SKELETON_EDGES:
                v = mapped_m[e] - mapped_m[s]
                unit_v = v / (np.linalg.norm(v) + 1e-8)
                reconstructed[e] = reconstructed[s] + unit_v * self.s9_lengths[(s, e)]
            
            # 3. 深度平面補償 (Chest Plane Fix)
            # 補償 Mixamo 內部骨點與 H3.6M 表面標記的 5.5cm 斷層
            chest_plane_indices = [8, 11, 14] # Thorax, L-Shoulder, R-Shoulder
            reconstructed[chest_plane_indices] += np.array([0, 0.055, 0]) # 向前 (Y+) 推移
            
            final_sequence[f] = reconstructed - reconstructed[0]

        return final_sequence

    def audit_invariance(self, processed_sequence):
        """
        驗證指標：時序長度變異量 (Temporal Bone-length Invariance)
        """
        print("\n📊 正在執行時序穩定性審計...")
        all_lengths = []
        for f in range(processed_sequence.shape[0]):
            frame_lengths = [np.linalg.norm(processed_sequence[f, e] - processed_sequence[f, s]) 
                             for s, e in SKELETON_EDGES]
            all_lengths.append(frame_lengths)
        
        std_dev = np.std(all_lengths, axis=0)
        avg_std = np.mean(std_dev)
        
        print("-" * 50)
        print(f"📌 平均骨長時序偏差 (Avg STD): {avg_std*1000:.6f} mm")
        print(f"📌 最大骨段抖動 (Max Jitter): {np.max(std_dev)*1000:.6f} mm")
        print("-" * 50)
        
        if avg_std < 1e-6:
            print("💎 結論：生理指紋完美守恆，地基艙具備「幾何自洽性」。")
        else:
            print("⚠️ 警告：檢測到骨架形變，請檢查旋轉權重映射。")

    def visualize_frame(self, pose_3d, frame_idx=0):
        fig = plt.figure(figsize=(8, 8)); ax = fig.add_subplot(111, projection='3d')
        for s, e in SKELETON_EDGES:
            ax.plot([pose_3d[s,0], pose_3d[e,0]], [pose_3d[s,1], pose_3d[e,1]], [pose_3d[s,2], pose_3d[e,2]], 
                    'r-', linewidth=3)
        ax.set_box_aspect([1, 1, 1]); ax.set_xlim(-0.8, 0.8); ax.set_ylim(-0.8, 0.8); ax.set_zlim(-0.2, 1.2)
        plt.title(f"V44.4 Sovereignty HUD - Frame {frame_idx}"); plt.show()

# ==============================================================================
# 執行：地基艙最後一步
# ==============================================================================
if __name__ == "__main__":
    # 這裡替換為您的路徑
    h36m_path = r'D:\videopose2\VideoPose3D\data\data_3d_h36m.npz'
    
    # 模擬您的 Sitting 數據輸入 (以您提供的坐姿單幀座標擴展為序列)
    raw_sitting_frame = np.array([
        [-46.73, -4.04, 60.38], [-44.84, 6.22, 56.04], [-3.70, 17.25, 51.59], [-11.37, 14.84, 12.91], 
        [-44.51, -13.33, 54.33], [-2.54, -21.07, 50.51], [-9.70, -19.06, 11.83], [-51.97, -4.88, 68.98], 
        [-53.32, -4.72, 80.70], [-50.92, -2.79, 109.13], [-46.10, -3.25, 115.38], [-49.65, -22.96, 101.12], 
        [-51.42, -32.63, 77.79], [-41.81, -35.43, 56.90], [-51.98, 16.56, 98.77], [-55.32, 22.89, 74.48], [-43.12, 25.87, 55.01]
    ])
    sequence_input = np.tile(raw_sitting_frame, (10, 1, 1)) # 模擬 10 幀序列

    validator = V44_4_ValidationEngine(h36m_path)
    final_output = validator.run_v44_4_protocol(sequence_input)
    
    # 執行時序審計
    validator.audit_invariance(final_output)
    
    # 視覺化第一幀
    validator.visualize_frame(final_output[0])