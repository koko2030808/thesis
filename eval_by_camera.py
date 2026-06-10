"""
逐 Camera 評估腳本
==================
基於你的 run.py 架構，新增按 Camera 分類統計 MPJPE 與 CV。
執行：python eval_by_camera.py --evaluate pretrained_h36m_cpn.bin --dataset h36m --keypoints cpn_ft_h36m_dbb -arc 3,3,3,3,3

此腳本輸出論文第一層需要的核心表格：
- 各 Camera 的平均 MPJPE
- 各 Camera 的平均 CV（骨骼變異係數）
- Camera 3 vs 其他的差距
- Camera 3 最差動作 Top 5
"""

import numpy as np
from common.arguments import parse_args
import torch
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys

from common.camera import *
from common.model import *
from common.loss import *
from common.generators import UnchunkedGenerator
from common.utils import deterministic_random

# ==========================================
# 骨骼鏈定義（用於計算 CV）
# ==========================================
BONE_CHAINS = {
    'Right_Leg': (0,1,2,3),
    'Left_Leg':  (0,4,5,6),
    'Torso':     (0,7,8,9),
    'Left_Arm':  (8,11,12,13),
    'Right_Arm': (8,14,15,16),
}

def compute_cv(pos_3d):
    """骨骼變異係數：衡量骨骼長度在時序上的不穩定度"""
    total_cv = 0.0
    for joints in BONE_CHAINS.values():
        lengths = []
        for i in range(len(joints) - 1):
            diff = pos_3d[:, joints[i], :] - pos_3d[:, joints[i+1], :]
            lengths.append(np.linalg.norm(diff, axis=1))
        chain = np.sum(lengths, axis=0)
        total_cv += np.std(chain) / (np.mean(chain) + 1e-8)
    return total_cv / len(BONE_CHAINS)


# ==========================================
# 主程式：複用 run.py 的資料載入邏輯
# ==========================================
args = parse_args()

# 載入資料集
print('Loading dataset...')
dataset_path = 'data/data_3d_' + args.dataset + '.npz'
if args.dataset == 'h36m':
    from common.h36m_dataset import Human36mDataset
    dataset = Human36mDataset(dataset_path)
else:
    raise KeyError('此腳本僅支援 h36m')

print('Preparing data...')
for subject in dataset.subjects():
    for action in dataset[subject].keys():
        anim = dataset[subject][action]
        if 'positions' in anim:
            positions_3d = []
            for cam in anim['cameras']:
                pos_3d = world_to_camera(anim['positions'], R=cam['orientation'], t=cam['translation'])
                pos_3d[:, 1:] -= pos_3d[:, :1]
                positions_3d.append(pos_3d)
            anim['positions_3d'] = positions_3d

print('Loading 2D detections...')
keypoints = np.load('data/data_2d_' + args.dataset + '_' + args.keypoints + '.npz', allow_pickle=True)
keypoints_metadata = keypoints['metadata'].item()
keypoints_symmetry = keypoints_metadata['keypoints_symmetry']
kps_left, kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
joints_left, joints_right = list(dataset.skeleton().joints_left()), list(dataset.skeleton().joints_right())
keypoints = keypoints['positions_2d'].item()

# 裁切 & 歸一化
for subject in dataset.subjects():
    for action in dataset[subject].keys():
        if 'positions_3d' not in dataset[subject][action]:
            continue
        for cam_idx in range(len(keypoints[subject][action])):
            mocap_length = dataset[subject][action]['positions_3d'][cam_idx].shape[0]
            if keypoints[subject][action][cam_idx].shape[0] > mocap_length:
                keypoints[subject][action][cam_idx] = keypoints[subject][action][cam_idx][:mocap_length]

for subject in keypoints.keys():
    for action in keypoints[subject]:
        for cam_idx, kps in enumerate(keypoints[subject][action]):
            cam = dataset.cameras()[subject][cam_idx]
            kps[..., :2] = normalize_screen_coordinates(kps[..., :2], w=cam['res_w'], h=cam['res_h'])
            # 改成
            keypoints[subject][action][cam_idx] = kps[..., :2]  # strip score channel

# 載入模型
filter_widths = [int(x) for x in args.architecture.split(',')]
model_pos = TemporalModel(17, 2, dataset.skeleton().num_joints(),
                          filter_widths=filter_widths, causal=args.causal,
                          dropout=args.dropout, channels=args.channels, dense=args.dense)
receptive_field = model_pos.receptive_field()
pad = (receptive_field - 1) // 2
causal_shift = pad if args.causal else 0

if torch.cuda.is_available():
    model_pos = model_pos.cuda()

chk_filename = os.path.join(args.checkpoint, args.evaluate)
print('Loading checkpoint', chk_filename)
checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
model_pos.load_state_dict(checkpoint['model_pos'])
model_pos.eval()

subjects_test = args.subjects_test.split(',')

# ==========================================
# 逐 Camera 評估核心
# ==========================================
# results[cam_idx] = list of {subject, action, mpjpe, cv}
results = {i: [] for i in range(4)}

print('\n開始逐 Camera 評估...')

for subject in subjects_test:
    for action in sorted(dataset[subject].keys()):
        if 'positions_3d' not in dataset[subject][action]:
            continue

        poses_3d_list = dataset[subject][action]['positions_3d']  # list of 4 cameras
        poses_2d_list = keypoints[subject][action]                # list of 4 cameras

        for cam_idx in range(len(poses_3d_list)):
            poses_3d = poses_3d_list[cam_idx]  # (T, 17, 3)
            poses_2d = poses_2d_list[cam_idx]  # (T, 17, 2)

            # 建立 Generator
            gen = UnchunkedGenerator(
                None, [poses_3d], [poses_2d],
                pad=pad, causal_shift=causal_shift, augment=False,
                kps_left=kps_left, kps_right=kps_right,
                joints_left=joints_left, joints_right=joints_right
            )

            # 推理並收集預測結果
            all_pred = []
            all_gt   = []
            with torch.no_grad():
                for _, batch_3d, batch_2d in gen.next_epoch():
                    inp = torch.from_numpy(batch_2d.astype('float32'))
                    if torch.cuda.is_available():
                        inp = inp.cuda()
                    pred = model_pos(inp).cpu().numpy()  # (1, T', 17, 3)
                    pred = pred.squeeze(0)               # (T', 17, 3)

                    gt = batch_3d.squeeze(0)             # (T', 17, 3)
                    gt[:, 0] = 0                         # root 歸零（與 run.py 一致）

                    all_pred.append(pred)
                    all_gt.append(gt)

            pred_all = np.concatenate(all_pred, axis=0)  # (T', 17, 3)
            gt_all   = np.concatenate(all_gt,   axis=0)

            # MPJPE（已是相機空間、root-relative）
            mpjpe = np.mean(np.linalg.norm(pred_all - gt_all, axis=-1)) * 1000

            # CV（骨骼變異係數）
            cv = compute_cv(pred_all)

            results[cam_idx].append({
                'subject': subject,
                'action':  action,
                'mpjpe':   mpjpe,
                'cv':      cv,
            })

            print(f'  {subject} | {action:<20} | Cam{cam_idx} | '
                  f'MPJPE={mpjpe:6.2f}mm | CV={cv:.4f}')

# ==========================================
# 輸出論文表格
# ==========================================
CAM_NAMES = ['Cam0', 'Cam1', 'Cam2', 'Cam3(俯視)']

print('\n' + '='*72)
print('  論文第一層核心結果：逐 Camera 的 MPJPE 與 CV')
print('='*72)
print(f"  {'Camera':<18} {'MPJPE (mm)':>12} {'CV':>10} {'樣本數':>8}")
print('-'*72)

cam_summary = {}
for cam_idx in range(4):
    data = results[cam_idx]
    if not data:
        continue
    mpjpe_avg = np.mean([d['mpjpe'] for d in data])
    cv_avg    = np.mean([d['cv']    for d in data])
    n         = len(data)
    cam_summary[cam_idx] = (mpjpe_avg, cv_avg, n)
    marker = '  ← 論文主角（俯視）' if cam_idx == 3 else ''
    print(f'  {CAM_NAMES[cam_idx]:<18} {mpjpe_avg:>12.2f} {cv_avg:>10.4f} {n:>8}{marker}')

print('='*72)

# Camera 3 vs 其他
if 3 in cam_summary and len(cam_summary) > 1:
    c3 = cam_summary[3]
    others = [cam_summary[i] for i in range(3) if i in cam_summary]
    avg_mpjpe = np.mean([o[0] for o in others])
    avg_cv    = np.mean([o[1] for o in others])
    print(f'\n  Camera 3（俯視）vs 其他相機平均：')
    print(f'    MPJPE：{c3[0]:.2f}mm vs {avg_mpjpe:.2f}mm（差距 {c3[0]-avg_mpjpe:+.2f}mm）')
    print(f'    CV：  {c3[1]:.4f} vs {avg_cv:.4f}（差距 {c3[1]-avg_cv:+.4f}）')

# Camera 3 最差動作 Top 5
if results[3]:
    print(f'\n  Camera 3 最差動作 Top 5（依 CV 排序）：')
    print(f"  {'Subject':<8} {'Action':<22} {'MPJPE':>10} {'CV':>8}")
    print('  ' + '-'*52)
    for r in sorted(results[3], key=lambda x: x['cv'], reverse=True)[:5]:
        print(f"  {r['subject']:<8} {r['action']:<22} {r['mpjpe']:>10.2f} {r['cv']:>8.4f}")

print('\n' + '='*72)
print('  逐動作平均 MPJPE（跨所有 Camera）')
print('='*72)
action_data = {}
for cam_idx in range(4):
    for r in results[cam_idx]:
        a = r['action']
        if a not in action_data:
            action_data[a] = {'mpjpe': [], 'cv': []}
        action_data[a]['mpjpe'].append(r['mpjpe'])
        action_data[a]['cv'].append(r['cv'])

for action in sorted(action_data.keys()):
    avg_mpjpe = np.mean(action_data[action]['mpjpe'])
    avg_cv = np.mean(action_data[action]['cv'])
    marker = ' ← 病灶' if 'Sitting' in action else ''
    print(f'  {action:<28} {avg_mpjpe:>8.2f}mm  CV={avg_cv:.4f}{marker}')

print('\n[✅] 評估完成。以上是論文第一層的 Baseline 數字。')