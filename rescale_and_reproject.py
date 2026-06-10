"""
rescale_and_reproject.py
骨長 rescale（修正 j0 格式）+ 重投影 → 生成一致的 rescaled 2D/3D 疫苗對
"""
import json, numpy as np

KEEP = [0,1,2,3,6,7,8,12,13,14,15,17,18,19,25,26,27]
BONES = [
    (1,0,'R_hip'),(2,1,'R_femur'),(3,2,'R_tibia'),
    (4,0,'L_hip'),(5,4,'L_femur'),(6,5,'L_tibia'),
    (7,0,'spine_lo'),(8,7,'spine_up'),(9,8,'neck'),(10,9,'head'),
    (11,8,'L_clav'),(12,11,'L_uarm'),(13,12,'L_farm'),
    (14,8,'R_clav'),(15,14,'R_uarm'),(16,15,'R_farm'),
]
PARENTS = [-1,0,1,2,0,4,5,0,7,8,9,8,11,12,8,14,15]
IMAGE_W  = 1000
CAM_PATH = r'D:\04.09\VideoPose3D\vaccine_camera.json'

def mean_bone_lengths(poses):
    return np.array([np.linalg.norm(poses[:,c]-poses[:,p],axis=-1).mean() for c,p,_ in BONES])

# ── 載入 ──────────────────────────────────────────────────
vacc_poses = np.load('data/data_3d_vaccine.npz', allow_pickle=True
    )['positions_3d'].item()['S_vaccine']['Sitting 1'].copy()  # (T,17,3)

h36m_data = np.load('data/data_3d_h36m.npz', allow_pickle=True)['positions_3d'].item()
h36m_target = np.mean([
    mean_bone_lengths(np.concatenate([p[:,KEEP] for p in subj.values()],0))
    for subj in h36m_data.values()], axis=0)

# ── Step 1：隔離 j0（絕對相機座標，不是骨架）─────────────
j0_cam = vacc_poses[:, 0].copy()   # (T,3) 骨盆在相機空間的絕對位置
vacc_poses[:, 0] = 0               # 暫設為原點 → 才能正確計算骨長

# ── Step 2：scale factors ─────────────────────────────────
vacc_bl = mean_bone_lengths(vacc_poses)
sf = h36m_target / vacc_bl

print(f"{'骨頭':<12} {'前(mm)':>8} {'目標(mm)':>10} {'×scale':>8}")
print('-' * 44)
for i,(*_,name) in enumerate(BONES):
    print(f"{name:<12} {vacc_bl[i]*1000:>8.1f} {h36m_target[i]*1000:>10.1f} {sf[i]:>8.3f}")

# ── Step 3：拓撲 rescale ──────────────────────────────────
rescaled = vacc_poses.copy()       # j0 = 0（暫時）
for i in range(1, 17):
    p = PARENTS[i]
    vec = vacc_poses[:, i] - vacc_poses[:, p]
    rescaled[:, i] = rescaled[:, p] + vec * sf[i-1]

rescaled[:, 0] = j0_cam            # j0 還原為絕對相機座標

# 驗證
tmp = rescaled.copy(); tmp[:,0] = 0
after_bl = mean_bone_lengths(tmp)
max_err = np.abs(after_bl - h36m_target).max() * 1000
print(f"\n最大殘差: {max_err:.4f}mm  {'✅ OK' if max_err < 0.1 else '❌ 異常'}")

# ── 儲存 3D ───────────────────────────────────────────────
np.savez_compressed('data/data_3d_vaccine_rescaled.npz',
    positions_3d={'S_vaccine': {'Sitting 1': rescaled.astype(np.float32)}})
print(f"✅ 3D → data/data_3d_vaccine_rescaled.npz")

# ── Step 4：重投影 rescaled 3D → normalized 2D ────────────
with open(CAM_PATH, 'r') as f:
    cam = json.load(f)
cam0  = cam["data"][0] if "data" in cam else cam[0]
fx    = (IMAGE_W / 2.0) / np.tan(np.radians(cam0["fov"]) / 2.0)
cx    = cy = IMAGE_W / 2.0

# root-relative + j0_cam → 絕對相機座標
absolute          = rescaled.astype(np.float64).copy()
absolute[:, 1:] += j0_cam[:, np.newaxis, :]

Z   = absolute[:, :, 2]                         # depth
u   = fx * (absolute[:, :, 0] / Z) + cx         # pixel u
v   = fx * (absolute[:, :, 1] / Z) + cy         # pixel v
u_n = (u / (IMAGE_W / 2.0) - 1.0).astype(np.float32)
v_n = (v / (IMAGE_W / 2.0) - 1.0).astype(np.float32)
poses_2d = np.stack([u_n, v_n], axis=-1)        # (T,17,2)

print(f"2D 範圍：{poses_2d.min():.3f} ~ {poses_2d.max():.3f}（應在 -1~1）")

# ── 儲存 2D ───────────────────────────────────────────────
metadata = {'num_joints': 17, 'keypoints_symmetry': [[4,5,6,11,12,13],[1,2,3,14,15,16]]}
np.savez_compressed('data/data_2d_vaccine_ue5_rescaled.npz',
    positions_2d={'S_vaccine': {'Sitting 1': [poses_2d]}},
    metadata=metadata)
print(f"✅ 2D → data/data_2d_vaccine_ue5_rescaled.npz")
