import numpy as np

KEEP = [0,1,2,3,6,7,8,12,13,14,15,17,18,19,25,26,27]

BONES = [
    (1, 0, 'R_hip'),   (2, 1, 'R_femur'),  (3, 2, 'R_tibia'),
    (4, 0, 'L_hip'),   (5, 4, 'L_femur'),  (6, 5, 'L_tibia'),
    (7, 0, 'spine_lo'),(8, 7, 'spine_up'), (9, 8, 'neck'),
    (10,9, 'head'),    (11,8, 'L_clav'),   (12,11,'L_uarm'),
    (13,12,'L_farm'),  (14,8, 'R_clav'),   (15,14,'R_uarm'),
    (16,15,'R_farm'),
]

def mean_bone_lengths(poses, fix_root=False):
    """poses: (N,17,3) → 16 mean bone lengths (meters)"""
    if fix_root:
        # vaccine joint 0 = 世界原點，用 midpoint(R_hip, L_hip) 當合成骨盆
        poses = poses.copy()
        poses[:, 0] = (poses[:, 1] + poses[:, 4]) / 2
    return np.array([
        np.linalg.norm(poses[:, c] - poses[:, p], axis=-1).mean()
        for c, p, _ in BONES
    ])

# ── 載入疫苗 ──────────────────────────────────────────
vacc = np.load('data/data_3d_vaccine.npz', allow_pickle=True)
vacc_poses = vacc['positions_3d'].item()['S_vaccine']['Sitting 1']
print(f"疫苗 shape: {vacc_poses.shape}")
vacc_bl = mean_bone_lengths(vacc_poses, fix_root=True)   # ← 修正 root

# ── 載入 H36M，32→17 ─────────────────────────────────
h36m = np.load('data/data_3d_h36m.npz', allow_pickle=True)
h36m_data = h36m['positions_3d'].item()

subjects = sorted(h36m_data.keys())
h36m_bl_per_subj = {}
for subj in subjects:
    all_poses = np.concatenate([p[:, KEEP] for p in h36m_data[subj].values()], axis=0)
    h36m_bl_per_subj[subj] = mean_bone_lengths(all_poses)  # H36M root-centered，不需修正

h36m_mean_bl = np.mean(list(h36m_bl_per_subj.values()), axis=0)

# ── 對照表（mm）────────────────────────────────────────
print(f"\n{'骨頭':<12} {'疫苗(mm)':>10} {'H36M均(mm)':>12} {'差(mm)':>10} {'差(%)':>8}")
print('-' * 58)
for i, (*_, name) in enumerate(BONES):
    v, h = vacc_bl[i]*1000, h36m_mean_bl[i]*1000  # → mm
    print(f"{name:<12} {v:>10.1f} {h:>12.1f} {v-h:>10.1f} {(v-h)/h*100:>8.1f}%")

# ── Sanity check ──────────────────────────────────────
ridx = [n for *_, n in BONES].index('R_femur')
print(f"\n── R_femur sanity (期望：H36M≈420-450mm，疫苗≈500mm) ──")
for subj in subjects:
    print(f"  {subj}: {h36m_bl_per_subj[subj][ridx]*1000:.1f} mm")
print(f"  疫苗: {vacc_bl[ridx]*1000:.1f} mm")