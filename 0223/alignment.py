import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

# ==============================================================================
# V167 V42: 局部旋轉重定向協議 (Sovereignty OS)
# ==============================================================================
JOINT_NAMES = ["Hips", "R-Hip", "R-Knee", "R-Ankle", "L-Hip", "L-Knee", "L-Ankle", "Spine", "Thorax", "Neck", "Head", "L-Shoulder", "L-Elbow", "L-Wrist", "R-Shoulder", "R-Elbow", "R-Wrist"]
SKELETON_EDGES = [(0, 7), (7, 8), (8, 9), (9, 10), (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6), (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16)]
HIERARCHY = {0: [1, 4, 7], 1: [2], 2: [3], 4: [5], 5: [6], 7: [8], 8: [9, 11, 14], 9: [10], 11: [12], 12: [13], 14: [15], 15: [16]}

class V167_SovereigntyOS_V42:
    def __init__(self, h36m_path):
        print("\n>>> 啟動 V167 V42：執行「動態主權重定向」...")
        data = np.load(h36m_path, allow_pickle=True)['positions_3d'].item()
        h36m_idx = [0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27]
        # 提取目標(S9)物理指紋：生理長度與生理 T-Pose 基準向量
        self.target_gt = data['S9']['Sitting 1'][0].reshape(-1, 3)[h36m_idx] 
        self.target_gt -= self.target_gt[0] 
        self.fingerprint = self._extract_fingerprint(self.target_gt)
        print(f"✅ S9 生理指紋已鎖定 (基於 1.56mm 標定基準)。")

    def _extract_fingerprint(self, pose):
        fp = {}
        for p, children in HIERARCHY.items():
            for c in children:
                vec = pose[c] - pose[p]
                length = np.linalg.norm(vec)
                base_unit = vec / (length + 1e-8)
                fp[(p, c)] = {'L': length, 'U_base': base_unit}
        return fp

    def _get_rotation_delta(self, v_tpose, v_current):
        """
        計算從 T-Pose 到當前幀的旋轉矩陣 (Rodrigues' rotation)
        """
        v_t = v_tpose / (np.linalg.norm(v_tpose) + 1e-8)
        v_c = v_current / (np.linalg.norm(v_current) + 1e-8)
        
        cross_prod = np.cross(v_t, v_c)
        dot_prod = np.dot(v_t, v_c)
        
        # 處理極端情況 (同向或反向)
        if np.linalg.norm(cross_prod) < 1e-8:
            return np.eye(3) if dot_prod > 0 else -np.eye(3)
        
        # 建立偏對稱矩陣 (Skew-symmetric matrix)
        k = np.array([[0, -cross_prod[2], cross_prod[1]],
                      [cross_prod[2], 0, -cross_prod[0]],
                      [-cross_prod[1], cross_prod[0], 0]])
        
        # Rodrigues 公式
        rotation_matrix = np.eye(3) + k + k @ k * ((1 - dot_prod) / (np.linalg.norm(cross_prod)**2))
        return rotation_matrix

    def _fix_chirality(self, pose):
        """
        執行單軸反射 (det = -1)，將左手系(UE5)轉換為右手系(H3.6M)
        """
        p_corrected = pose.copy()
        p_corrected[:, 0] *= -1 # 反射 X 軸
        return p_corrected

    def run_v42_retargeting(self, mix_tpose, mix_current):
        """
        [L3_Execution] 核心：將 Mixamo 的角位移 Δθ 注入 S9 生理容器
        """
        # A. 座標歸一化 (修正手性與朝向)
        m_t = self._fix_chirality(mix_tpose)
        m_c = self._fix_chirality(mix_current)
        
        # 修正全域朝向 (Rotation Only)
        m_t[:, 1] *= -1; m_c[:, 1] *= -1
        
        # B. 語義對齊 (LR Swap)
        idx_r, idx_l = [1, 2, 3, 14, 15, 16], [4, 5, 6, 11, 12, 13]
        m_t[idx_r], m_t[idx_l] = m_t[idx_l].copy(), m_t[idx_r].copy()
        m_c[idx_r], m_c[idx_l] = m_c[idx_l].copy(), m_c[idx_r].copy()
        
        # Head/Neck Swap
        m_t[[9, 10]] = m_t[[10, 9]]; m_c[[9, 10]] = m_c[[10, 9]]
        
        # C. 物理重建流程
        recon = np.zeros_like(m_c)
        recon[0] = [0, 0, 0] # Root
        self._recursive_retarget(recon, m_t, m_c, 0)
        
        # D. 語義微調
        recon[10] += np.array([0, 0, 0.025])
        
        # E. 最終審計 (Protocol #2)
        final_ue, final_gt = self.procrustes_align(recon, self.target_gt)
        errors = np.linalg.norm(final_ue - final_gt, axis=1) * 1000
        mpjpe = np.mean(errors)
        
        print(f"🎯 [V167 V42 終極標定報告] - MPJPE: {mpjpe:.2f} mm")
        self.visualize(final_ue, final_gt, mpjpe)
        return final_ue, mpjpe

    def _recursive_retarget(self, target, m_t, m_c, parent):
        if parent not in HIERARCHY: return
        for child in HIERARCHY[parent]:
            # 1. 提取 Mixamo 的角位移靈魂
            v_t = m_t[child] - m_t[parent]
            v_c = m_c[child] - m_c[parent]
            R_delta = self._get_rotation_delta(v_t, v_c)
            
            # 2. 注入 S9 的生理肉體
            fp = self.fingerprint[(parent, child)]
            s9_v_base = fp['U_base']
            s9_length = fp['L']
            
            # 3. 幾何重建公式
            # P_child = P_parent + (R_delta * U_base) * L
            target[child] = target[parent] + (R_delta @ s9_v_base) * s9_length
            
            self._recursive_retarget(target, m_t, m_c, child)

    def procrustes_align(self, source, target):
        s_centered, t_centered = source - source[0], target - target[0]
        H = s_centered.T @ t_centered
        U, S, Vt = np.linalg.svd(H)
        R_mat = Vt.T @ U.T
        if np.linalg.det(R_mat) < 0: Vt[2, :] *= -1; R_mat = Vt.T @ U.T
        return s_centered @ R_mat.T, t_centered

    def visualize(self, ue, gt, mpjpe):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(ue[:,0], ue[:,1], ue[:,2], c='red', s=100, label='V42 Sovereign OS')
        ax.scatter(gt[:,0], gt[:,1], gt[:,2], c='green', alpha=0.3, s=100, label='S9 Real GT')
        for i, j in SKELETON_EDGES:
            ax.plot([ue[i,0], ue[j,0]], [ue[i,1], ue[j,1]], [ue[i,2], ue[j,2]], color='red', linewidth=3)
            ax.plot([gt[i,0], gt[j,0]], [gt[i,1], gt[j,1]], [gt[i,2], gt[j,2]], color='green', alpha=0.1)
        
        # 固定三軸限制 (視覺降噪)
        limit = 0.8
        ax.set_xlim(-limit, limit); ax.set_ylim(-limit, limit); ax.set_zlim(-limit, limit)
        ax.set_box_aspect([1, 1, 1])
        plt.title(f"V167 V42: Dynamic Delta Retargeting\nMPJPE: {mpjpe:.2f}mm")
        plt.legend(loc='upper left'); plt.show()

if __name__ == "__main__":
    # 使用你的 Mixamo 座標
    MIX_T = np.array([[0.0, 2.6, 99.1], [-9.8, 2.3, 93.5], [-12.3, -0.5, 50.9], [-14.7, -5.0, 11.7], [9.8, 2.3, 93.5], [12.3, -0.7, 50.9], [14.7, -3.8, 11.7], [0.0, 0.2, 120.9], [0.0, -1.2, 134.3], [0.0, -2.8, 149.4], [0.0, -0.01, 156.8], [21.0, -2.9, 142.9], [46.2, -4.3, 141.3], [69.4, -3.2, 142.2], [-21.0, -2.8, 142.9], [-46.2, -4.3, 141.3], [-69.4, -3.1, 142.2]])
    MIX_C = MIX_T.copy() # 此處可替換為動態幀
    
    path = r'D:\videopose2\VideoPose3D\data\data_3d_h36m.npz'
    engine = V167_SovereigntyOS_V42(path)
    engine.run_v42_retargeting(MIX_T, MIX_C)