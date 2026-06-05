"""
VideoPose3D 論文第一層評估腳本
================================
輸出：每個 Camera 的 MPJPE 和 CV，特別凸顯 Camera 3（俯視角）
執行：python diagnose_camera.py
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
import sys

sys.path.append(os.getcwd())
from common.model import TemporalModel
from common.camera import normalize_screen_coordinates

# 請確保這三個檔案在您的硬碟中真實存在，並填入絕對路徑
CHK_PATH    = r'D:\04.09\VideoPose3D\checkpoint\pretrained_h36m_cpn.bin' # 你的預訓練模型在哪？
DATA_2D     = r'D:\04.09\VideoPose3D\data\data_2d_h36m_cpn_ft_h36m_dbb.npz' # 你的官方 2D 預測檔在哪？
DATA_3D     = r'D:\04.09\VideoPose3D\data\data_3d_h36m.npz' # 你的官方 3D 真值檔在哪？

H36M_17J    = [0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27]
SUBJECTS    = ['S9', 'S11']
CAMERA_NAMES = ['Cam0', 'Cam1', 'Cam2', 'Cam3(俯視)']

BONE_CHAINS = {
    'Right_Leg': (0,1,2,3), 'Left_Leg': (0,4,5,6),
    'Torso':     (0,7,8,9), 'Left_Arm': (8,11,12,13),
    'Right_Arm': (8,14,15,16)
}


# ==========================================
# 工具函數
# ==========================================
def compute_mpjpe(pred, gt):
    """Mean Per Joint Position Error，單位 mm"""
    return np.mean(np.linalg.norm(pred - gt, axis=-1)) * 1000


def compute_cv(pos_3d):
    """骨骼變異係數：衡量骨骼長度在時序上的不穩定度"""
    total_cv = 0
    for joints in BONE_CHAINS.values():
        lengths = []
        for i in range(len(joints) - 1):
            p1 = pos_3d[:, joints[i], :]
            p2 = pos_3d[:, joints[i+1], :]
            lengths.append(np.linalg.norm(p1 - p2, axis=1))
        chain_len = np.sum(lengths, axis=0)
        total_cv += np.std(chain_len) / (np.mean(chain_len) + 1e-8)
    return total_cv / len(BONE_CHAINS)


# ==========================================
# 主評估流程
# ==========================================
def run_evaluation():
    # 載入模型
    print("[*] 載入模型...")
    model = TemporalModel(17, 2, 17,
                          filter_widths=[3,3,3,3,3],
                          causal=False, channels=1024)
    ckpt = torch.load(CHK_PATH, map_location='cpu')
    model.load_state_dict(ckpt['model_pos'])
    model.eval()

    # 載入數據
    print("[*] 載入數據集...")
    d2d = np.load(DATA_2D, allow_pickle=True)['positions_2d'].item()
    d3d = np.load(DATA_3D, allow_pickle=True)['positions_3d'].item()

    # 結果容器：results[cam_idx] = list of (mpjpe, cv, subject, action)
    results = {i: [] for i in range(4)}

    print("[*] 開始逐 Camera 評估...\n")

    for sub in SUBJECTS:
        for action in sorted(d2d[sub].keys()):
            cam_list = d2d[sub][action]

            for cam_idx, kps_raw in enumerate(cam_list):
                if kps_raw is None:
                    continue

                # 取對應的 3D GT
                gt_3d_all = d3d[sub][action][:, H36M_17J, :]

                # 歸一化 2D
                kps_norm = normalize_screen_coordinates(
                    kps_raw.astype(np.float32), w=1000, h=1002
                )

                # 推理
                with torch.no_grad():
                    inp = torch.from_numpy(kps_norm).float().unsqueeze(0)
                    pred = model(inp).numpy()[0]

                # 時序對齊
                pad = (kps_norm.shape[0] - pred.shape[0]) // 2
                gt   = gt_3d_all[pad: pad + pred.shape[0]]

                # 對齊長度防呆
                min_len = min(pred.shape[0], gt.shape[0])
                pred = pred[:min_len]
                gt   = gt[:min_len]

                # 計算指標
                mpjpe = compute_mpjpe(pred, gt)
                cv    = compute_cv(pred)

                results[cam_idx].append({
                    'subject': sub,
                    'action':  action,
                    'mpjpe':   mpjpe,
                    'cv':      cv,
                })

    # ==========================================
    # 輸出結果表
    # ==========================================
    print("\n" + "="*70)
    print("  論文第一層核心結果：逐 Camera 的 MPJPE 與 CV")
    print("="*70)
    print(f"  {'Camera':<15} {'MPJPE (mm)':>12} {'CV':>10} {'樣本數':>8}")
    print("-"*70)

    cam_summary = {}
    for cam_idx in range(4):
        data = results[cam_idx]
        if not data:
            continue
        mpjpe_avg = np.mean([d['mpjpe'] for d in data])
        cv_avg    = np.mean([d['cv']    for d in data])
        n         = len(data)
        cam_summary[cam_idx] = (mpjpe_avg, cv_avg, n)

        marker = "  ← 俯視角（論文主角）" if cam_idx == 3 else ""
        print(f"  {CAMERA_NAMES[cam_idx]:<15} {mpjpe_avg:>12.2f} {cv_avg:>10.4f} {n:>8}{marker}")

    print("="*70)

    # Camera 3 vs 其他的對比
    if 3 in cam_summary:
        cam3_mpjpe = cam_summary[3][0]
        cam3_cv    = cam_summary[3][1]
        other_mpjpe = np.mean([cam_summary[i][0] for i in range(3) if i in cam_summary])
        other_cv    = np.mean([cam_summary[i][1] for i in range(3) if i in cam_summary])

        print(f"\n  Camera 3 vs 其他相機平均：")
        print(f"    MPJPE：Camera3 = {cam3_mpjpe:.2f}mm，其他平均 = {other_mpjpe:.2f}mm"
              f"，差距 = {cam3_mpjpe - other_mpjpe:+.2f}mm")
        print(f"    CV：  Camera3 = {cam3_cv:.4f}，其他平均 = {other_cv:.4f}"
              f"，差距 = {cam3_cv - other_cv:+.4f}")

    # 最差動作排行（Camera 3 專屬）
    print(f"\n  Camera 3 最差動作 Top 5（CV 排行）：")
    print(f"  {'Subject':<8} {'Action':<20} {'MPJPE':>10} {'CV':>8}")
    print("  " + "-"*50)
    cam3_sorted = sorted(results[3], key=lambda x: x['cv'], reverse=True)
    for r in cam3_sorted[:5]:
        print(f"  {r['subject']:<8} {r['action']:<20} "
              f"{r['mpjpe']:>10.2f} {r['cv']:>8.4f}")

    print("\n[✅] 評估完成，以上數字即為論文第一層的 Baseline 結果。")


if __name__ == "__main__":
    run_evaluation()
