"""
Fine-tuning 腳本 V3：Detectron2 / COCO 路線
===================================================
基於 finetune_v2.py，三處改動切換到 Detectron 路線：

  改動 1（命令參數）：--keypoints detectron_pt_coco
                      --resume pretrained_h36m_detectron_coco.bin
  改動 2（疫苗 2D）：  data_2d_vaccine_detectron.npz（COCO 格式）
  改動 3（幀數對齊）：  3D(6920) 和 2D(6919) 取較短

執行：
python finetune_v3.py --dataset h36m --keypoints detectron_pt_coco \
  --resume pretrained_h36m_detectron_coco.bin --epochs 20 -arc 3,3,3,3,3

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AI BUILDER NOTE：為什麼 v3 不能直接和 v2 比？
  v2 baseline（CPN）= 46.8mm
  v3 baseline（Detectron）= ~51.6mm（Detectron 比 CPN 不準）
  → v3 要和「Detectron baseline」比，不是和 v2 比
  → 跑 v3 評估前，先跑 detectron baseline 當對照組
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import numpy as np
from common.arguments import parse_args
import torch
import torch.optim as optim
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import errno

from common.camera import *
from common.model import *
from common.loss import *
from common.generators import ChunkedGenerator, UnchunkedGenerator
from time import time
from common.utils import deterministic_random

args = parse_args()
print(args)

os.makedirs(args.checkpoint, exist_ok=True)

# ============================================================
# 原始 run.py 的數據載入流程（完全不動）
# ============================================================
print('Loading dataset...')
dataset_path = 'data/data_3d_' + args.dataset + '.npz'
from common.h36m_dataset import Human36mDataset
dataset = Human36mDataset(dataset_path)

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

# 【改動 1 自動生效】--keypoints detectron_pt_coco 會載入 COCO 格式的 2D
# 它的 metadata 裡是 COCO symmetry，kps_left/kps_right 自動正確
print('Loading 2D detections...')
keypoints = np.load('data/data_2d_' + args.dataset + '_' + args.keypoints + '.npz', allow_pickle=True)
keypoints_metadata = keypoints['metadata'].item()
keypoints_symmetry = keypoints_metadata['keypoints_symmetry']
kps_left, kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
joints_left, joints_right = list(dataset.skeleton().joints_left()), list(dataset.skeleton().joints_right())
keypoints = keypoints['positions_2d'].item()

for subject in dataset.subjects():
    for action in dataset[subject].keys():
        if 'positions_3d' not in dataset[subject][action]:
            continue
        for cam_idx in range(len(keypoints[subject][action])):
            mocap_length = dataset[subject][action]['positions_3d'][cam_idx].shape[0]
            if keypoints[subject][action][cam_idx].shape[0] > mocap_length:
                keypoints[subject][action][cam_idx] = keypoints[subject][action][cam_idx][:mocap_length]

# for subject in keypoints.keys():
#     for action in keypoints[subject]:
#         for cam_idx, kps in enumerate(keypoints[subject][action]):
#             cam = dataset.cameras()[subject][cam_idx]
#             kps[..., :2] = normalize_screen_coordinates(kps[..., :2], w=cam['res_w'], h=cam['res_h'])
#             keypoints[subject][action][cam_idx] = kps
for subject in keypoints.keys():
    for action in keypoints[subject]:
        for cam_idx, kps in enumerate(keypoints[subject][action]):
            cam = dataset.cameras()[subject][cam_idx]
            kps[..., :2] = normalize_screen_coordinates(kps[..., :2], w=cam['res_w'], h=cam['res_h'])
            # 【核心修正】：強制切片，丟棄 COCO 的第三維度 (Score)，保持與預訓練權重一致的 2 維
            keypoints[subject][action][cam_idx] = kps[..., :2]

subjects_train = args.subjects_train.split(',')
subjects_test = args.subjects_test.split(',')

# 原始 fetch 函數（完全不動）
def fetch(subjects, action_filter=None, subset=1, parse_3d_poses=True):
    out_poses_3d = []
    out_poses_2d = []
    out_camera_params = []
    for subject in subjects:
        for action in keypoints[subject].keys():
            if action_filter is not None:
                found = False
                for a in action_filter:
                    if action.startswith(a):
                        found = True
                        break
                if not found:
                    continue
            poses_2d = keypoints[subject][action]
            for i in range(len(poses_2d)):
                out_poses_2d.append(poses_2d[i])
            if subject in dataset.cameras():
                cams = dataset.cameras()[subject]
                for cam in cams:
                    if 'intrinsic' in cam:
                        out_camera_params.append(cam['intrinsic'])
            if parse_3d_poses and 'positions_3d' in dataset[subject][action]:
                poses_3d = dataset[subject][action]['positions_3d']
                assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
                for i in range(len(poses_3d)):
                    out_poses_3d.append(poses_3d[i])
    if len(out_camera_params) == 0:
        out_camera_params = None
    if len(out_poses_3d) == 0:
        out_poses_3d = None
    stride = args.downsample
    if subset < 1:
        for i in range(len(out_poses_2d)):
            n_frames = int(round(len(out_poses_2d[i])//stride * subset)*stride)
            start = deterministic_random(0, len(out_poses_2d[i]) - n_frames + 1, str(len(out_poses_2d[i])))
            out_poses_2d[i] = out_poses_2d[i][start:start+n_frames:stride]
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][start:start+n_frames:stride]
    elif stride > 1:
        for i in range(len(out_poses_2d)):
            out_poses_2d[i] = out_poses_2d[i][::stride]
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][::stride]
    return out_camera_params, out_poses_3d, out_poses_2d

cameras_valid, poses_valid, poses_valid_2d = fetch(subjects_test)
cameras_train, poses_train, poses_train_2d = fetch(subjects_train, subset=args.subset)

# ============================================================
# 注入疫苗數據（改動 2 + 改動 3）
# ============================================================
print('\n[疫苗] 載入合成坐姿數據（Detectron2 COCO 格式）...')
d3 = np.load('data/data_3d_vaccine.npz', allow_pickle=True)

# 【改動 2】2D 來源換成 Detectron2 偵測結果（COCO 格式）
d2 = np.load('data/data_2d_vaccine_detectron.npz', allow_pickle=True)

p3d = d3['positions_3d'].item()
p2d = d2['positions_2d'].item()

vaccine_3d = []
vaccine_2d = []
for subject in p3d.keys():
    for action in p3d[subject].keys():
        seq_3d = np.array(p3d[subject][action])
        seq_2d_list = p2d[subject][action]
        for seq_2d in seq_2d_list:
            # 【核心修正】：確保疫苗數據也強制只取 X, Y 兩維
            seq_2d = np.array(seq_2d)[..., :2]

            # 【改動 3】幀數對齊：3D(6920) 和 2D(6919) 取較短
            # AI BUILDER NOTE：ChunkedGenerator 要求同一序列的 3D 和 2D
            # 幀數一致，否則切 batch 時會錯位。Detectron2 比 3D GT 少 1 幀
            # （UE5 渲染 0000~6918，但 3D 抓到第 6919 幀），所以截掉。
            min_len = min(len(seq_3d), len(seq_2d))
            seq_3d_use = seq_3d[:min_len]
            seq_2d = seq_2d[:min_len]

            # 確保長度超過感受野
            while len(seq_3d_use) < 300:
                seq_3d_use = np.concatenate([seq_3d_use, seq_3d_use], axis=0)
                seq_2d = np.concatenate([seq_2d, seq_2d], axis=0)

            vaccine_3d.append(seq_3d_use)
            vaccine_2d.append(seq_2d)

print(f'[疫苗] 對齊後每序列幀數: {len(vaccine_3d[0])}')

REPEAT = 10
for _ in range(REPEAT):
    poses_train.extend(vaccine_3d)
    poses_train_2d.extend(vaccine_2d)
    if cameras_train is not None:
        cameras_train.extend([cameras_train[0]] * len(vaccine_3d))

print(f'[疫苗] 注入 {len(vaccine_3d) * REPEAT} 個序列（重複 {REPEAT} 次）')
print(f'[訓練集] 總序列數：{len(poses_train)}')

# ============================================================
# 原始 run.py 的模型建立與訓練（完全不動）
# ============================================================
filter_widths = [int(x) for x in args.architecture.split(',')]
model_pos_train = TemporalModelOptimized1f(
    poses_valid_2d[0].shape[-2], poses_valid_2d[0].shape[-1],
    dataset.skeleton().num_joints(),
    filter_widths=filter_widths, causal=args.causal,
    dropout=args.dropout, channels=args.channels
)
model_pos = TemporalModel(
    poses_valid_2d[0].shape[-2], poses_valid_2d[0].shape[-1],
    dataset.skeleton().num_joints(),
    filter_widths=filter_widths, causal=args.causal,
    dropout=args.dropout, channels=args.channels
)

receptive_field = model_pos.receptive_field()
print('INFO: Receptive field: {} frames'.format(receptive_field))
pad = (receptive_field - 1) // 2
causal_shift = pad if args.causal else 0

if torch.cuda.is_available():
    model_pos = model_pos.cuda()
    model_pos_train = model_pos_train.cuda()

# 載入預訓練權重（改動 1：detectron 模型由命令參數 --resume 指定）
chk_filename = os.path.join(args.checkpoint, args.resume)
print('Loading checkpoint', chk_filename)
checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
print('This model was trained for {} epochs'.format(checkpoint['epoch']))
model_pos_train.load_state_dict(checkpoint['model_pos'])
model_pos.load_state_dict(checkpoint['model_pos'])

optimizer = optim.Adam(model_pos_train.parameters(), lr=args.learning_rate, amsgrad=True)
lr = args.learning_rate
lr_decay = args.lr_decay

train_generator = ChunkedGenerator(
    args.batch_size // args.stride, cameras_train, poses_train, poses_train_2d, args.stride,
    pad=pad, causal_shift=causal_shift, shuffle=True, augment=args.data_augmentation,
    kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right
)
test_generator = UnchunkedGenerator(
    cameras_valid, poses_valid, poses_valid_2d,
    pad=pad, causal_shift=causal_shift, augment=False,
    kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right
)
print('INFO: Training on {} frames'.format(train_generator.num_frames()))
print('INFO: Testing on {} frames'.format(test_generator.num_frames()))

initial_momentum = 0.1
final_momentum = 0.001

print('\n開始 Fine-tuning（V3 Detectron 路線）...')
for epoch in range(args.epochs):
    start_time = time()
    epoch_loss_3d_train = 0
    N = 0

    model_pos_train.train()
    for _, batch_3d, batch_2d in train_generator.next_epoch():
        inputs_3d = torch.from_numpy(batch_3d.astype('float32'))
        inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
        if torch.cuda.is_available():
            inputs_3d = inputs_3d.cuda()
            inputs_2d = inputs_2d.cuda()
        inputs_3d[:, :, 0] = 0
        optimizer.zero_grad()
        predicted_3d_pos = model_pos_train(inputs_2d)
        loss_3d_pos = mpjpe(predicted_3d_pos, inputs_3d)
        epoch_loss_3d_train += inputs_3d.shape[0] * inputs_3d.shape[1] * loss_3d_pos.item()
        N += inputs_3d.shape[0] * inputs_3d.shape[1]
        loss_3d_pos.backward()
        optimizer.step()

    # 驗證
    with torch.no_grad():
        model_pos.load_state_dict(model_pos_train.state_dict())
        model_pos.eval()
        epoch_loss_3d_valid = 0
        N_val = 0
        for _, batch, batch_2d in test_generator.next_epoch():
            inputs_3d = torch.from_numpy(batch.astype('float32'))
            inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
            if torch.cuda.is_available():
                inputs_3d = inputs_3d.cuda()
                inputs_2d = inputs_2d.cuda()
            inputs_3d[:, :, 0] = 0
            predicted_3d_pos = model_pos(inputs_2d)
            loss_3d_pos = mpjpe(predicted_3d_pos, inputs_3d)
            epoch_loss_3d_valid += inputs_3d.shape[0] * inputs_3d.shape[1] * loss_3d_pos.item()
            N_val += inputs_3d.shape[0] * inputs_3d.shape[1]

    elapsed = (time() - start_time) / 60
    lr *= lr_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] *= lr_decay
    momentum = initial_momentum * np.exp(-(epoch+1)/args.epochs * np.log(initial_momentum/final_momentum))
    model_pos_train.set_bn_momentum(momentum)

    print('[%d] time %.2f lr %f 3d_train %f 3d_valid %f' % (
        epoch + 1, elapsed, lr,
        epoch_loss_3d_train / N * 1000,
        epoch_loss_3d_valid / N_val * 1000
    ))

    # V3 用不同檔名，不覆蓋 v2 的 checkpoint
    chk_path = os.path.join(args.checkpoint, f'finetuned_sitting_v3_epoch{epoch+1}.bin')
    torch.save({
        'epoch': epoch + 1,
        'model_pos': model_pos_train.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr': lr,
    }, chk_path)

print(f'\n[✅] V3 Fine-tuning 完成！')
print(f'\n=== 評估步驟（注意：要和 Detectron baseline 比，不是和 v2 比）===')
print(f'# 1. 先跑 Detectron baseline 當對照組')
print(f'python eval_by_camera.py --evaluate pretrained_h36m_detectron_coco.bin --dataset h36m --keypoints detectron_pt_coco -arc 3,3,3,3,3')
print(f'# 2. 再跑 v3 fine-tuned')
print(f'python eval_by_camera.py --evaluate finetuned_sitting_v3_epoch{args.epochs}.bin --dataset h36m --keypoints detectron_pt_coco -arc 3,3,3,3,3')