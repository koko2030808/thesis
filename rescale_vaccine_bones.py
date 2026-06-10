import numpy as np

KEEP = [0,1,2,3,6,7,8,12,13,14,15,17,18,19,25,26,27]
BONES = [
    (1,0,'R_hip'),(2,1,'R_femur'),(3,2,'R_tibia'),
    (4,0,'L_hip'),(5,4,'L_femur'),(6,5,'L_tibia'),
    (7,0,'spine_lo'),(8,7,'spine_up'),(9,8,'neck'),(10,9,'head'),
    (11,8,'L_clav'),(12,11,'L_uarm'),(13,12,'L_farm'),
    (14,8,'R_clav'),(15,14,'R_uarm'),(16,15,'R_farm'),
]
PARENTS = [-1,0,1,2,0,4,5,0,7,8,9,8,11,12,8,14,15]

def mean_bone_lengths(poses):
    return np.array([np.linalg.norm(poses[:,c]-poses[:,p],axis=-1).mean() for c,p,_ in BONES])

# ── 載入 ──────────────────────────────────────────────────────────────
vacc_poses = np.load('data/data_3d_vaccine.npz', allow_pickle=True
    )['positions_3d'].item()['S_vaccine']['Sitting 1'].copy()

h36m_data = np.load('data/data_3d_h36m.npz', allow_pickle=True)['positions_3d'].item()
h36m_target = np.mean([
    mean_bone_lengths(np.concatenate([p[:,KEEP] for p in subj.values()], 0))
    for subj in h36m_data.values()
], axis=0)

# ── Step 1：修正 root（UE5 actor root → 合成骨盆中心）─────────────────
vacc_poses[:, 0] = (vacc_poses[:, 1] + vacc_poses[:, 4]) / 2

# ── Step 2：計算逐骨 scale factors ─────────────────────────────────────
vacc_bl = mean_bone_lengths(vacc_poses)
sf = h36m_target / vacc_bl

print(f"{'骨頭':<12} {'前(mm)':>8} {'目標(mm)':>10} {'×scale':>8}")
print('-' * 44)
for i, (*_, name) in enumerate(BONES):
    print(f"{name:<12} {vacc_bl[i]*1000:>8.1f} {h36m_target[i]*1000:>10.1f} {sf[i]:>8.3f}")

# ── Step 3：拓撲順序逐骨 rescale（方向保留，僅改長度）────────────────
rescaled = vacc_poses.copy()
for i in range(1, 17):
    p = PARENTS[i]
    vec = vacc_poses[:, i] - vacc_poses[:, p]   # 原始骨頭方向向量
    rescaled[:, i] = rescaled[:, p] + vec * sf[i-1]  # 新父節點位置 + 縮放後骨向量

# ── 驗證：殘差應 < 0.01mm ─────────────────────────────────────────────
after_bl = mean_bone_lengths(rescaled)
max_err = np.abs(after_bl - h36m_target).max() * 1000
print(f"\n最大殘差: {max_err:.4f}mm  ({'✅ OK' if max_err < 0.1 else '❌ 異常'})")

# ── 儲存 ──────────────────────────────────────────────────────────────
np.savez_compressed('data/data_3d_vaccine_rescaled.npz',
    positions_3d={'S_vaccine': {'Sitting 1': rescaled}})
print(f"✅ 儲存 → data/data_3d_vaccine_rescaled.npz  shape: {rescaled.shape}")
print("⚠️  下一步：用 rescaled npz 重跑 project_v167.py → 更新 data_2d_vaccine_ue5_rescaled.npz")
