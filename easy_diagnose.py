"""
VideoPose3D 論文第一層評估腳本 v3
====================================
修正：直接存取 dataset_3d._data 避免 KeyError
執行：python diagnose_camera_v3.py
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
import sys

sys.path.append(os.getcwd())
from common.model import TemporalModel
from common.camera import normalize_screen_coordinates
from common.h36m_dataset import Human36mDataset

# ==========================================
# 設定區
# ==========================================
CHK_PATH = r'D:\videopose2\VideoPose3D\checkpoint\pretrained_h36m_cpn.bin'
DATA_2D  = r'data\data_2d_h36m_cpn_ft_h36m_dbb.npz'
DATA_3D  = r'data\data_3d_h36m.npz'

SUBJECTS     = ['S9', 'S11']
CAMERA_NAMES = ['Cam0', 'Cam1', 'Cam2', 'Cam3(俯視)']

BONE_CHAINS = {
    'Right_Leg': (0,1,2,3),
    'Left_Leg':  (0,4,5,6),
    'Torso':     (0,7,8,9),
    'Left_Arm':  (8,11,12,13),
    'Right_Arm': (8,14,15,16),
}


def compute_mpjpe(pred, gt):
    return np.mean(np.linalg.norm(pred - gt, axis=-1)) * 1000


def compute_cv(pos_3d):
    total_cv = 0.0
    for joints in BONE_CHAINS.values():
        lengths = []
        for i in range(len(joints) - 1):
            diff = pos_3d[:, joints[i], :] - pos_3d[:, joints[i+1], :]
            lengths.append(np.linalg.norm(diff, axis=1))
        chain = np.sum(lengths, axis=0)
        total_cv += np.std(chain) / (np.mean(chain) + 1e-8)
    return total_cv / len(BONE_CHAINS)


def run_evaluation():
    # 載入模型
    print("[*] 載入模型...")
    model = TemporalModel(17, 2, 17,
                          filter_widths=[3,3,3,3,3],
                          causal=False, channels=1024)
    ckpt = torch.load(CHK_PATH, map_location='cpu')
    model.load_state_dict(ckpt['model_pos'])
    model.eval()
    print("    ✅ 模型載入完成")

    # 載入 3D GT
    print("[*] 載入 3D GT...")
    dataset_3d = Human36mDataset(DATA_3D)
    
    # 直接存取 _data 字典，避免 __getitem__ 的 KeyError
    gt_data = dataset_3d._data
    
    # Sanity Check：印出結構
    print(f"    GT subjects: {list(gt_data.keys())}")
    first_sub = list(gt_data.keys())[0]
    first_actions = list(gt_data[first_sub].keys())
    print(f"    {first_sub} actions (前5個): {first_actions[:5]}")
    sample = gt_data[first_sub][first_actions[0]]['positions']
    print(f"    positions shape: {sample.shape}")
    print(f"    positions 值域: {sample.min():.4f} ~ {sample.max():.4f}")

    # 載入 2D 數據
    print("[*] 載入 2D 數據...")
    d2d = np.load(DATA_2D, allow_pickle=True)['positions_2d'].item()
    print("    ✅ 2D 數據載入完成")

    results = {i: [] for i in range(4)}

    print("\n[*] 開始逐 Camera 評估...\n")

    for sub in SUBJECTS:
        if sub not in gt_data:
            print(f"    ⚠️  {sub} 不在 GT 數據中，跳過")
            continue
        if sub not in d2d:
            print(f"    ⚠️  {sub} 不在 2D 數據中，跳過")
            continue

        for action in sorted(d2d[sub].keys()):
            cam_list = d2d[sub][action]

            # 找對應的 GT（處理 action 名稱帶尾綴的情況）
            gt_all = None
            for gt_action in gt_data[sub].keys():
                if gt_action == action or action.startswith(gt_action):
                    gt_all = gt_data[sub][gt_action]['positions']
                    break
            
            if gt_all is None:
                # 嘗試去掉尾綴
                base = action.replace(' 1', '').replace(' 2', '')
                if base in gt_data[sub]:
                    gt_all = gt_data[sub][base]['positions']
                else:
                    print(f"    ⚠️  找不到 {sub}/{action} 的 GT，跳過")
                    continue

            for cam_idx, kps_raw in enumerate(cam_list):
                if kps_raw is None or len(kps_raw) == 0:
                    continue

                # 歸一化 2D
                kps_norm = normalize_screen_coordinates(
                    kps_raw.astype(np.float32), w=1000, h=1002
                )

                # 推理
                with torch.no_grad():
                    inp  = torch.from_numpy(kps_norm).float().unsqueeze(0)
                    pred = model(inp).numpy()[0]

                # 時序對齊
                pad     = (kps_norm.shape[0] - pred.shape[0]) // 2
                gt_trim = gt_all[pad: pad + pred.shape[0]]
                min_len = min(pred.shape[0], gt_trim.shape[0])
                pred    = pred[:min_len]
                gt_trim = gt_trim[:min_len]

                # Root-Relative 對齊
                pred_rel = pred    - pred[:, :1, :]
                gt_rel   = gt_trim - gt_trim[:, :1, :]

                mpjpe = compute_mpjpe(pred_rel, gt_rel)
                cv    = compute_cv(pred_rel)

                results[cam_idx].append({
                    'subject': sub,
                    'action':  action,
                    'mpjpe':   mpjpe,
                    'cv':      cv,
                })

                print(f"    {sub} | {action:<20} | Cam{cam_idx} | "
                      f"MPJPE={mpjpe:6.2f}mm | CV={cv:.4f}")

    # 輸出表格
    print("\n" + "="*72)
    print("  論文第一層核心結果：逐 Camera 的 MPJPE 與 CV")
    print("="*72)
    print(f"  {'Camera':<18} {'MPJPE (mm)':>12} {'CV':>10} {'樣本數':>8}")
    print("-"*72)

    cam_summary = {}
    for cam_idx in range(4):
        data = results[cam_idx]
        if not data:
            print(f"  {CAMERA_NAMES[cam_idx]:<18} {'無數據':>12}")
            continue
        mpjpe_avg = np.mean([d['mpjpe'] for d in data])
        cv_avg    = np.mean([d['cv']    for d in data])
        n         = len(data)
        cam_summary[cam_idx] = (mpjpe_avg, cv_avg, n)
        marker = "  ← 論文主角（俯視）" if cam_idx == 3 else ""
        print(f"  {CAMERA_NAMES[cam_idx]:<18} {mpjpe_avg:>12.2f} "
              f"{cv_avg:>10.4f} {n:>8}{marker}")

    print("="*72)

    if 3 in cam_summary and len(cam_summary) > 1:
        c3_mpjpe = cam_summary[3][0]
        c3_cv    = cam_summary[3][1]
        others   = [cam_summary[i] for i in range(3) if i in cam_summary]
        avg_mpjpe = np.mean([o[0] for o in others])
        avg_cv    = np.mean([o[1] for o in others])
        print(f"\n  Camera 3（俯視）vs 其他相機平均：")
        print(f"    MPJPE：{c3_mpjpe:.2f}mm vs {avg_mpjpe:.2f}mm"
              f"（差距 {c3_mpjpe - avg_mpjpe:+.2f}mm）")
        print(f"    CV：  {c3_cv:.4f} vs {avg_cv:.4f}"
              f"（差距 {c3_cv - avg_cv:+.4f}）")

    if results[3]:
        print(f"\n  Camera 3 最差動作 Top 5（依 CV 排序）：")
        print(f"  {'Subject':<8} {'Action':<22} {'MPJPE':>10} {'CV':>8}")
        print("  " + "-"*52)
        for r in sorted(results[3], key=lambda x: x['cv'], reverse=True)[:5]:
            print(f"  {r['subject']:<8} {r['action']:<22} "
                  f"{r['mpjpe']:>10.2f} {r['cv']:>8.4f}")

    print("\n[✅] 評估完成。以上是論文第一層的 Baseline 數字。")


if __name__ == "__main__":
    run_evaluation()