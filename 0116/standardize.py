import numpy as np
import os

# ==============================================================================
# 1. 系統拓撲與配置 (Topology & Config)
# ==============================================================================
# 建立 MECE 的動力學樹狀結構，確保所有點位無遺漏 [cite: 2026-01-19]
JOINT_NAMES = [
    'Pelvis', 'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle',
    'Spine', 'Thorax', 'Neck', 'Head', 'LShoulder', 'LElbow', 'LWrist',
    'RShoulder', 'RElbow', 'RWrist'
]

PARENT_MAP = {
    0: None, 1: 0, 2: 1, 3: 2, 4: 0, 5: 4, 6: 5,
    7: 0, 8: 7, 9: 8, 10: 9, 11: 8, 12: 11, 13: 12,
    14: 8, 15: 14, 16: 15
}

# 訓練集受試者：用於建立「統計平均基準」 [cite: 2026-01-19]
TRAIN_SUBJECTS = ['S1', 'S5', 'S6', 'S7', 'S8']
H36M_17_IDX = [0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27]

# ==============================================================================
# 2. 統計基準模組 (Statistical Baseline)
# ==============================================================================
def get_h36m_mean_lengths(data_path):
    """ 
    計算 H3.6M 訓練集的平均骨骼長度。
    邏輯：透過奧卡姆剃刀原則，將複雜的受試者差異簡化為一套「標準人」基準 [cite: 2026-01-19]。
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"找不到 H3.6M 數據: {data_path}")
        
    data = np.load(data_path, allow_pickle=True)['positions_3d'].item()
    bone_lengths = { (p, c): [] for c, p in PARENT_MAP.items() if p is not None }
    
    print(f"正在分析訓練集統計特徵: {TRAIN_SUBJECTS}...")
    for sub in TRAIN_SUBJECTS:
        if sub not in data: continue
        for act in data[sub].keys():
            # 獲取 17 點座標並計算序列平均姿勢
            poses = data[sub][act].reshape(-1, 32, 3)[:, H36M_17_IDX]
            mean_pose = np.mean(poses, axis=0) 
            for child, parent in PARENT_MAP.items():
                if parent is not None:
                    dist = np.linalg.norm(mean_pose[child] - mean_pose[parent])
                    bone_lengths[(parent, child)].append(dist)
    
    # 回傳每一根骨骼的平均物理長度 (單位：公尺) [cite: 2026-01-19]
    return {edge: np.mean(lengths) for edge, lengths in bone_lengths.items() if len(lengths) > 0}

# ==============================================================================
# 3. 生產級處理引擎 (The Production Engine)
# ==============================================================================
class ProductionPipeline:
    def __init__(self, mean_lengths):
        self.mean_lengths = mean_lengths
        self.children_map = {i: [] for i in range(17)}
        for c, p in PARENT_MAP.items():
            if p is not None: self.children_map[p].append(c)

    def _recursive_refine(self, joints, current, offset):
        """ 
        動力學遞歸：解決「骨架斷裂」與「局部/整體縮放衝突」的核心算法。
        公式：$P_{child} = P_{parent} + \text{Normalize}(\vec{V}_{orig}) \cdot L_{mean}$ [cite: 2026-01-19]
        """
        joints[current] += offset
        for child in self.children_map[current]:
            edge = (current, child)
            c_offset = offset.copy()
            if edge in self.mean_lengths:
                vec = joints[child] - joints[current]
                dist = np.linalg.norm(vec)
                # 強制賦值為 H3.6M 平均長度，同時保留 UE 的原始方向 [cite: 2026-01-19]
                target_dist = self.mean_lengths[edge]
                if dist > 0:
                    new_pos = joints[current] + (vec / dist) * target_dist
                    c_offset = new_pos - joints[child]
            self._recursive_refine(joints, child, c_offset)

    def process_frame(self, raw_ue, spine_offset_y=0.035, neck_offset_z=0.05):
        """ 
        處理單幀數據：包含座標對齊、遞歸精修與解剖補償 [cite: 2026-01-19]。
        """
        # A. 座標變換與手性修正 (CM -> M, 右手座標系對齊) [cite: 2026-01-16, 2026-01-19]
        conv = np.array(raw_ue) / 100.0
        conv[:, 0] = -conv[:, 0] # X 鏡像
        l, r = [4, 5, 6, 11, 12, 13], [1, 2, 3, 14, 15, 16]
        conv[l+r] = conv[r+l].copy() # 標籤交換
        conv[:, 1] = -conv[:, 1] # Y 朝向修正

        # B. 歸一化與遞歸精修
        refined = conv - conv[0]
        self._recursive_refine(refined, 0, np.zeros(3))
        
        # C. 解剖偏移補償 (Anatomical Offset)
        # 解決「點在背部 vs 點在中心」的語義衝突 [cite: 2026-01-19]
        refined[8, 1] -= spine_offset_y      # Thorax 深度補償
        refined[9, 1] -= spine_offset_y * 0.5 # Neck 深度補償
        refined[9, 2] += neck_offset_z       # Neck 高度補償 (針對 138mm 誤差) [cite: 2026-01-19]
        refined[10, 2] += neck_offset_z      # Head 高度同步平移
        
        return refined - refined[0]

# ==============================================================================
# 4. 自動化批次量產模組 (Batch Production)
# ==============================================================================
def batch_produce_dataset(h36m_path, ue_source_dir, output_path):
    """ 
    大數據生產線：將所有 .npy 原料轉化為標準 .npz 文件 [cite: 2026-01-19]。
    """
    # 建立生產環境
    mean_lengths = get_h36m_mean_lengths(h36m_path)
    pipeline = ProductionPipeline(mean_lengths)
    dataset = {}
    
    if not os.path.exists(ue_source_dir):
        print(f"警告：找不到原料目錄 {ue_source_dir}")
        return

    files = [f for f in os.listdir(ue_source_dir) if f.endswith('.npy')]
    print(f"開始量產，預計處理 {len(files)} 個序列...")

    for filename in files:
        filepath = os.path.join(ue_source_dir, filename)
        raw_seq = np.load(filepath)
        
        # 逐幀生產並確保 32-bit float 以符合模型要求 [cite: 2026-01-19]
        processed_seq = np.array([pipeline.process_frame(f) for f in raw_seq])
        dataset[filename.replace('.npy', '')] = processed_seq.astype(np.float32)
        print(f"  - 已完成: {filename}")
            
    # 封裝為標準 NPZ 格式
    np.savez(output_path, positions_3d=dataset)
    print(f"--- 量產完成！成果文件：{output_path} ---")

# ==============================================================================
# 5. 執行入口
# ==============================================================================
if __name__ == "__main__":
    # 配置路徑 (請根據你的實際目錄修改)
    H36M_NPZ_PATH = r'D:\videopose2\VideoPose3D\data\data_3d_h36m.npz'
    UE_RAW_DATA_DIR = r'D:\videopose2\VideoPose3D\0116\ue_raw_sequences'
    OUTPUT_FILENAME = "data_3d_ue_standardized_final.npz"
    
    batch_produce_dataset(H36M_NPZ_PATH, UE_RAW_DATA_DIR, OUTPUT_FILENAME)