import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ==========================================
# 1. 系統定錨與剛體拓撲 (Logical Foundation) [cite: 2026-01-19]
# ==========================================
PARENT_MAP = {
    0: None, 1: 0, 2: 1, 3: 2, 4: 0, 5: 4, 6: 5,
    7: 0, 8: 7, 9: 8, 10: 9, 11: 8, 12: 11, 13: 12,
    14: 8, 15: 14, 16: 15
}

SKELETON_EDGES = [
    (0,7),(7,8),(8,9),(9,10), (0,1),(1,2),(2,3), (0,4),(4,5),(5,6),
    (8,11),(11,12),(12,13), (8,14),(14,15),(15,16)
]

H36M_TO_17_IDX = [0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27]

class KinematicEngineV16:
    def __init__(self, mean_lengths):
        self.mean_lengths = mean_lengths
        self.children_map = {i: [] for i in range(17)}
        for c, p in PARENT_MAP.items():
            if p is not None: self.children_map[p].append(c)

    def _recursive_lock(self, joints, current):
        """ 第一性原理：向量方向守恆，模長強制校準 [cite: 2026-01-19, 2026-01-20] """
        for child in self.children_map[current]:
            edge = (current, child)
            vec = joints[child] - joints[current]
            dist = np.linalg.norm(vec)
            if dist > 1e-6:
                # 剛體核心公式 [cite: 2026-01-20]
                joints[child] = joints[current] + (vec / dist) * self.mean_lengths[edge]
            self._recursive_lock(joints, child)

    def process(self, raw_ue):
        # A. 鏡像對齊與單位轉換 [cite: 2026-01-16]
        conv = np.array(raw_ue) / 100.0
        conv[:, 0] = -conv[:, 0]; conv[:, 1] = -conv[:, 1]
        
        # B. 建立相對座標系
        refined = conv - conv[0]
        
        # C. 執行動力學遞歸鎖定 (消滅所有物理雜訊) [cite: 2026-01-19]
        self._recursive_lock(refined, 0)
        
        # D. 全局解剖學平移 (不影響相對長度 Std) [cite: 2026-01-20]
        return refined + np.array([0, 0, 0.065])

# ==========================================
# 2. 跨動作壓力測試與對比審計 [cite: 2026-01-20]
# ==========================================
def run_stress_test(sequences, engine, means):
    print(f"\n{'='*25} V16 跨動作壓力測試報告 {'='*25}")
    print(f"{'Sequence Name':<15} | {'Physical Std':<15} | {'Max Vel (m/f)':<15} | {'Logic Check'}")
    print("-" * 75)

    processed_data = {}
    for name, path in sequences.items():
        if not os.path.exists(path):
            print(f"--- ❌ 跳過 {name}: 文件不存在")
            continue
        
        raw = np.load(path)
        proc = np.array([engine.process(f) for f in raw])
        
        # 物理審計 (Physical Audit) [cite: 2026-01-19]
        p_test = proc[len(proc)//2]
        errors = [np.linalg.norm(p_test[c] - p_test[p]) - means[(p, c)] for c, p in PARENT_MAP.items() if p is not None]
        std_val = np.std(errors) * 1000 # mm
        
        # 時域審計 (Temporal Audit) [cite: 2026-01-20]
        vel = np.linalg.norm(np.diff(proc, axis=0), axis=-1)
        max_vel = np.max(np.mean(vel, axis=1))
        
        status = "✅ STABLE" if std_val < 1e-6 else "⚠️ NOISY"
        print(f"{name:<15} | {std_val:<15.10f} | {max_vel:<15.6f} | {status}")
        processed_data[name] = proc

    # 視覺化對比：顯示兩者結構是否一致
    if processed_data:
        fig = plt.figure(figsize=(14, 7))
        for i, (name, data) in enumerate(processed_data.items()):
            ax = fig.add_subplot(1, len(processed_data), i+1, projection='3d')
            p = data[len(data)//2]
            for s, e in SKELETON_EDGES:
                ax.plot([p[s,0], p[e,0]], [p[s,1], p[e,1]], [p[s,2], p[e,2]], c='b', marker='o', markersize=3)
            ax.set_title(f"{name} Structural Check")
            ax.set_xlim(-0.7, 0.7); ax.set_ylim(-0.7, 0.7); ax.set_zlim(-0.7, 0.7)
        plt.show()

# ==========================================
# 3. 執行入口 (PDCA 循環)
# ==========================================
if __name__ == "__main__":
    H36M_PATH = r'D:\videopose2\VideoPose3D\data\data_3d_h36m.npz'
    TEST_FILES = {
        "Sitting": r"D:\videopose2\VideoPose3D\0116\ue_raw_sequences\sitting_01.npy",
        "Dancing": r"D:\videopose2\VideoPose3D\0116\ue_raw_sequences\rumba_dancing.npy"
    }

    # [1] 建立統計基準 [cite: 2026-01-19]
    h36m = np.load(H36M_PATH, allow_pickle=True)['positions_3d'].item()
    bone_stats = {(p, c): [] for c, p in PARENT_MAP.items() if p is not None}
    for sub in ['S1', 'S5', 'S6', 'S7', 'S8']:
        if sub in h36m:
            for act in h36m[sub].keys():
                poses = h36m[sub][act].reshape(-1, 32, 3)[:, H36M_TO_17_IDX]
                for c, p in PARENT_MAP.items():
                    if p is not None:
                        bone_stats[(p, c)].extend(np.linalg.norm(poses[:,c]-poses[:,p], axis=1).tolist())
    
    means_val = {e: np.mean(v) for e, v in bone_stats.items()}
    engine = KinematicEngineV16(means_val)

    # [2] 執行雙動作對比審計 [cite: 2026-01-20]
    run_stress_test(TEST_FILES, engine, means_val)