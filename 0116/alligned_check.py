import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ==============================================================================
# 1. 系統拓撲與動力學群組定義
# ==============================================================================
JOINT_NAMES = [
    "Hips", "R-Hip", "R-Knee", "R-Ankle", "L-Hip", "L-Knee", "L-Ankle",
    "Spine", "Thorax", "Neck", "Head", "L-Shoulder", "L-Elbow", "L-Wrist",
    "R-Shoulder", "R-Elbow", "R-Wrist"
]

SKELETON_EDGES = [
    (0, 7), (7, 8), (8, 9), (9, 10), (0, 1), (1, 2), (2, 3),
    (0, 4), (4, 5), (5, 6), (8, 11), (11, 12), (12, 13),
    (8, 14), (14, 15), (15, 16)
]

HIERARCHY = {
    0: [1, 4, 7], 1: [2], 2: [3], 4: [5], 5: [6],
    7: [8], 8: [9, 11, 14], 9: [10], 11: [12], 12: [13],
    14: [15], 15: [16]
}

BONE_RATIOS = {
    (0, 7): 0.855, (7, 8): 0.539, (8, 9): 1.258, (9, 10): 0.688,
    (1, 2): 0.907, (2, 3): 0.840, (4, 5): 0.907, (5, 6): 0.843,
    (11, 12): 0.855, (12, 13): 0.930, (14, 15): 0.854, (15, 16): 0.930,
    (0, 1): 0.911, (0, 4): 0.911
}

# ==============================================================================
# 2. 核心引擎：動力學鎖定與視覺化升級
# ==============================================================================
class V167_SovereigntyEngine:
    def __init__(self, h36m_path):
        print("\n>>> 啟動 V167：執行動力學鎖定與各向同性審計協定...")
        self.data = np.load(h36m_path, allow_pickle=True)['positions_3d'].item()

    def _propagate_offset(self, joints, start_node, offset):
        if start_node in HIERARCHY:
            for child in HIERARCHY[start_node]:
                joints[child] += offset
                self._propagate_offset(joints, child, offset)

    def apply_recursive_scaling(self, joints, current_node, ratios):
        if current_node not in HIERARCHY: return
        for child in HIERARCHY[current_node]:
            edge = (current_node, child)
            if edge in ratios:
                vec = joints[child] - joints[current_node]
                scaled_vec = vec / ratios[edge]
                old_pos = joints[child].copy()
                joints[child] = joints[current_node] + scaled_vec
                offset = joints[child] - old_pos
                self._propagate_offset(joints, child, offset)
            self.apply_recursive_scaling(joints, child, ratios)

    def compute_angle(self, a, b, c):
        ba, bc = a - b, c - b
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

    def procrustes_align(self, source, target):
        s_centered, t_centered = source - source[0], target - target[0]
        H = s_centered.T @ t_centered
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0: Vt[2, :] *= -1; R = Vt.T @ U.T
        return s_centered @ R.T, t_centered

    def process_ue_pose(self, raw_ue):
        conv = np.array(raw_ue) / 100.0
        conv[:, 0] = -conv[:, 0] 
        l_idx, r_idx = [4, 5, 6, 11, 12, 13], [1, 2, 3, 14, 15, 16]
        conv[l_idx + r_idx] = conv[r_idx + l_idx].copy() 
        conv[:, 1] = -conv[:, 1] 
        refined = conv - conv[0]
        self.apply_recursive_scaling(refined, 0, BONE_RATIOS)
        return refined

    def run_final_audit(self, ue_raw):
        # 獲取 GT 真值 (S9 Sitting 1)
        gt_pose = self.data['S9']['Sitting 1'][0].reshape(-1, 3)[[0,1,2,3,6,7,8,12,13,14,15,17,18,19,25,26,27]]
        proc_pose = self.process_ue_pose(ue_raw)
        
        # 1. 對齊與各向同性審計
        ue_aligned, gt_aligned = self.procrustes_align(proc_pose, gt_pose)
        errors_per_joint = np.linalg.norm(ue_aligned - gt_aligned, axis=1) * 1000
        mpjpe = np.mean(errors_per_joint)
        std_error = np.std(errors_per_joint)
        
        print("\n" + "="*50)
        print(f"📊 [V167 各向同性審計報告] - 關鍵指標輸出")
        print("="*50)
        for name, err in zip(JOINT_NAMES, errors_per_joint):
            print(f"📍 {name:12}: {err:6.2f} mm")
        print("-" * 50)
        print(f"🎯 平均誤差 (MPJPE): {mpjpe:.2f} mm")
        print(f"🔥 誤差標準差 (STD) : {std_error:.2f} mm")
        print("="*50)

        # 2. 語義夾角審計
        chains = {"R-Knee": (1,2,3), "L-Knee": (4,5,6), "R-Elbow": (14,15,16), "L-Elbow": (11,12,13)}
        print("\n📏 [V167 語義夾角校準結果]")
        for name, (a, b, c) in chains.items():
            ang_ue = self.compute_angle(ue_aligned[a], ue_aligned[b], ue_aligned[c])
            ang_gt = self.compute_angle(gt_aligned[a], gt_aligned[b], gt_aligned[c])
            print(f"✅ {name:8}: Diff {abs(ang_ue - ang_gt):5.2f}°")
        
        # 3. 視覺化對位 (核心修正：強制等比例鎖定)
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 繪製點位
        ax.scatter(ue_aligned[:,0], ue_aligned[:,1], ue_aligned[:,2], c='red', s=60, label='V167 Corrected', edgecolors='black')
        ax.scatter(gt_aligned[:,0], gt_aligned[:,1], gt_aligned[:,2], c='green', alpha=0.15, s=60, label='S9 GT')

        # 繪製骨架連線
        for i, j in SKELETON_EDGES:
            ax.plot([ue_aligned[i,0], ue_aligned[j,0]], 
                    [ue_aligned[i,1], ue_aligned[j,1]], 
                    [ue_aligned[i,2], ue_aligned[j,2]], color='red', alpha=0.6)

        # 強制 XYZ 比例為 1:1:1，防止「瓜感」或 Hips 頂出
        all_data = np.vstack([ue_aligned, gt_aligned])
        max_range = np.array([all_data[:,0].max()-all_data[:,0].min(), 
                             all_data[:,1].max()-all_data[:,1].min(), 
                             all_data[:,2].max()-all_data[:,2].min()]).max() / 2.0

        mid_x = (all_data[:,0].max()+all_data[:,0].min()) * 0.5
        mid_y = (all_data[:,1].max()+all_data[:,1].min()) * 0.5
        mid_z = (all_data[:,2].max()+all_data[:,2].min()) * 0.5

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        ax.set_box_aspect([1, 1, 1]) 
        
        ax.view_init(elev=20, azim=45) 
        ax.set_xlabel('X (Width)'); ax.set_ylabel('Y (Depth)'); ax.set_zlabel('Z (Height)')
        plt.title(f"V167 Audit: MPJPE={mpjpe:.1f}mm, STD={std_error:.1f}mm\n(Ratio Locked 1:1:1)")
        plt.legend(); plt.show()

# ==============================================================================
# 3. 點火啟動
# ==============================================================================
if __name__ == "__main__":
    UE_RAW = np.array([
        [0.0, 2.663, 99.163], [-9.816, 2.307, 93.570], [-12.360, -0.579, 50.914], 
        [-14.715, -5.041, 11.724], [9.816, 2.352, 93.570], [12.360, -0.766, 50.914], 
        [14.715, -3.873, 11.723], [0.0, 0.268, 120.943], [0.0, -1.204, 134.346], 
        [0.0, -2.862, 149.424], [0.0, -0.012, 156.808], [21.066, -2.920, 142.908], 
        [46.294, -4.302, 141.333], [69.418, -3.246, 142.257], [-21.066, -2.804, 142.908], 
        [-46.294, -4.340, 141.333], [-69.418, -3.192, 142.257]
    ])
    
    path = r'D:\videopose2\VideoPose3D\data\data_3d_h36m.npz'
    engine = V167_SovereigntyEngine(path)
    engine.run_final_audit(UE_RAW)