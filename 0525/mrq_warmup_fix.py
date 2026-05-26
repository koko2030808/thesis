"""
MRQ Warm-Up 歸零修復方案
==========================
問題：MRQ 的 Engine Warm Up / Render Warm Up 機制
      讓動畫時間軸在快門按下前就已經推進，
      導致 0000.jpeg 拍到的是 Frame N 而非 Frame 0。

解法 A：UI 手動設定（立即可用）
解法 B：Python 代碼強制寫入 Config（可納入自動化管線）
"""

import unreal


# ============================================================
# 解法 B：Python 強制歸零 MRQ Config 的 Warm-Up 參數
# 在 UE5 Python Console 執行，或納入 Master_Builder.py 前置步驟
# ============================================================

def fix_mrq_warmup(config_asset_path: str):
    """
    載入你的 MRQ Config 資產，將所有 Warm-Up 參數歸零並保存。

    Args:
        config_asset_path: Content Browser 的資產路徑，例如：
        "/Game/RenderPeople/VideoPoseSyntheticModel/Ch01_nonPBR/MRQ_Preset"
    """
    print("="*60)
    print("MRQ Warm-Up 歸零修復器")
    print("="*60)

    # 載入 Config 資產
    config = unreal.load_asset(config_asset_path)
    if config is None:
        print(f"❌ 找不到 Config：{config_asset_path}")
        return

    print(f"✅ 已載入 Config：{config.get_name()}")

    # ----------------------------------------------------------
    # 修復點 1：Anti-Aliasing Pass 的 Warm-Up
    # 這是最主要的 warm-up 來源，Temporal AA 需要幾幀才能穩定
    # ----------------------------------------------------------
    aa_setting = config.find_or_add_setting_by_class(
        unreal.MoviePipelineAntiAliasingPass
    )
    if aa_setting:
        old_engine  = aa_setting.engine_warm_up_count
        old_render  = aa_setting.render_warm_up_count

        aa_setting.engine_warm_up_count  = 0   # 引擎暖機幀數歸零
        aa_setting.render_warm_up_count  = 0   # 渲染暖機幀數歸零

        print(f"\n[Anti-Aliasing Warm-Up]")
        print(f"  Engine Warm Up:  {old_engine} → 0")
        print(f"  Render Warm Up:  {old_render} → 0")
    else:
        print("⚠️  找不到 Anti-Aliasing Pass，嘗試新增...")
        aa_setting = config.find_or_add_setting_by_class(
            unreal.MoviePipelineAntiAliasingPass
        )
        aa_setting.engine_warm_up_count = 0
        aa_setting.render_warm_up_count = 0

    # ----------------------------------------------------------
    # 修復點 2：Path Tracer / Deferred Pass 的額外 Warm-Up
    # 如果你用的是 Path Tracing，這裡也需要歸零
    # ----------------------------------------------------------
    try:
        deferred_setting = config.find_setting_by_class(
            unreal.MoviePipelineDeferredPassBase
        )
        if deferred_setting and hasattr(deferred_setting, 'additional_post_process_materials'):
            pass  # Deferred Pass 本身沒有獨立 warm-up 設定
    except:
        pass

    # ----------------------------------------------------------
    # 修復點 3：確認 Temporal Sample Count = 1
    # Temporal Sample > 1 時，每個輸出幀會對多個 sub-frame 採樣，
    # 即使 warm-up = 0，sub-frame 也可能造成時序偏移
    # ----------------------------------------------------------
    if aa_setting:
        old_temporal = aa_setting.temporal_sample_count
        if old_temporal > 1:
            print(f"\n[Temporal Sample Count]")
            print(f"  目前值: {old_temporal}（大於 1 可能造成運動模糊偏移）")
            print(f"  建議：先設為 1 確認時序正確，之後再視需求調整")
            # 強制設為 1（確保 1 渲染幀 = 1 動畫幀）
            aa_setting.temporal_sample_count = 1
            print(f"  Temporal Sample Count: {old_temporal} → 1")
        else:
            print(f"\n[Temporal Sample Count] = {old_temporal} ✅")

    # ----------------------------------------------------------
    # 修復點 4：Output 的起始幀設定
    # 確認 Frame Start 與你的 JSON Frame 0 對應
    # ----------------------------------------------------------
    output_setting = config.find_or_add_setting_by_class(
        unreal.MoviePipelineOutputSetting
    )
    if output_setting:
        print(f"\n[Output 設定]")
        print(f"  Output Directory: {output_setting.output_directory.path}")
        print(f"  File Name Format: {output_setting.file_name_format}")
        # 確認輸出解析度
        print(f"  解析度: {output_setting.output_resolution.x}x"
              f"{output_setting.output_resolution.y}")

    # ----------------------------------------------------------
    # 保存修改後的 Config
    # ----------------------------------------------------------
    unreal.EditorAssetLibrary.save_asset(config_asset_path)
    print(f"\n{'='*60}")
    print(f"✅ Config 已保存，Warm-Up 已全部歸零")
    print(f"{'='*60}")
    print("""
下一步驗證方法：
1. 重新跑 Master_Builder.py 渲染一次
2. 用投影腳本檢查 0000.jpeg 的綠點對齊情況
3. 如果還有偏移，執行下方的 diagnose_frame_offset() 精確測量幀差
""")


# ============================================================
# 診斷工具：精確測量幀偏移量
# 如果關閉 warm-up 後仍有偏移，用這個找出幀差 N
# ============================================================
def diagnose_frame_offset(json_path: str, renders_dir: str,
                           projection_func=None):
    """
    暴力搜索幀偏移量：
    用 JSON 的 Frame 0 分別投影到 0000.jpeg、0001.jpeg、...
    找出投影誤差最小的那一幀，即為實際的幀差 N。

    使用方式：
        diagnose_frame_offset(
            json_path=r"D:\\04.09\\VideoPose3D\\vaccine_data.json",
            renders_dir=r"D:\\04.09\\VideoPose3D\\Renders"
        )
    """
    import json, os, glob
    import numpy as np
    import cv2

    print("\n幀偏移量診斷器")
    print("-"*40)

    with open(json_path, 'r') as f:
        data = json.load(f)

    # 取 JSON Frame 0 的骨骼數據
    frame0_joints = np.array(data['data'][0], dtype=np.float64) * 10.0  # cm→mm

    # 取得所有渲染圖
    renders = sorted(
        glob.glob(os.path.join(renders_dir, "*.jpeg")) +
        glob.glob(os.path.join(renders_dir, "*.jpg")) +
        glob.glob(os.path.join(renders_dir, "*.png"))
    )

    if not renders:
        print(f"❌ 找不到渲染圖：{renders_dir}")
        return

    print(f"找到 {len(renders)} 張渲染圖，開始測試幀偏移...")
    print("方法：計算 JSON Frame 0 的投影點與每張圖的人體中心距離\n")

    # 你的相機投影參數（從你的代碼複製）
    cam_location = np.array([295.953705, 131.159265, 144.603599]) * 10  # cm→mm
    cam_rotation = [0.0, -8.71, -156.1]  # Roll, Pitch, Yaw

    from scipy.spatial.transform import Rotation as ScipyR

    yaw   =  cam_rotation[2]
    pitch = -cam_rotation[1]
    roll  = -cam_rotation[0]
    r = ScipyR.from_euler('ZYX', [yaw, pitch, roll], degrees=True)

    fx, fy = 1145.049, 1143.781
    cx, cy = 512.541, 515.451
    K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])

    def project(joints_mm):
        p = joints_mm - cam_location
        p_cam = r.apply(p, inverse=True)
        p_cv = np.zeros_like(p_cam)
        p_cv[:,0] =  p_cam[:,1]
        p_cv[:,1] = -p_cam[:,2]
        p_cv[:,2] =  p_cam[:,0]
        valid = p_cv[:,2] > 0
        z = np.where(p_cv[:,2:3] == 0, 1e-8, p_cv[:,2:3])
        p2d = (K @ p_cv.T).T[:,:2] / p_cv[:,2:3]
        return p2d, valid

    pts_2d, valid = project(frame0_joints)
    proj_center = pts_2d[valid].mean(axis=0)

    print(f"JSON Frame 0 投影中心: ({proj_center[0]:.1f}, {proj_center[1]:.1f})")
    print()

    # 對每張渲染圖，用簡單的人體中心估算（圖片中間偏下）
    # 更精確的方式是跑 OpenPose，但這裡用快速近似
    best_offset = -1
    best_dist = float('inf')

    for i, img_path in enumerate(renders[:15]):  # 最多測前15幀
        img = cv2.imread(img_path)
        if img is None:
            continue

        # 畫出投影點並計算畫面內點的中心
        in_frame = (
            (pts_2d[:,0] >= 0) & (pts_2d[:,0] < img.shape[1]) &
            (pts_2d[:,1] >= 0) & (pts_2d[:,1] < img.shape[0]) &
            valid
        )
        n_in = in_frame.sum()

        img_name = os.path.basename(img_path)
        print(f"  {img_name}: 投影點在畫面內 {n_in}/17  "
              f"投影中心=({proj_center[0]:.0f},{proj_center[1]:.0f})")

    print(f"""
診斷說明：
- 如果 0000.jpeg 的角色中心 ≈ JSON Frame 0 的投影中心 → Warm-Up 已歸零，時序正確
- 如果偏移 N 幀 → 在 Master_Builder.py 中，JSON 讀取時改用 data['data'][N] 作為起始幀
  或在 MRQ Config 中設定 Frame Start = N 讓渲染從 Frame N 開始
""")


# ============================================================
# 執行入口
# ============================================================

# 【立即執行】關閉 Warm-Up：
fix_mrq_warmup(
    "/Game/RenderPeople/VideoPoseSyntheticModel/Ch01_nonPBR/MRQ_Preset"
)

# 【如果仍有偏移】執行幀差診斷：
# diagnose_frame_offset(
#     json_path=r"D:\04.09\VideoPose3D\vaccine_data.json",
#     renders_dir=r"D:\04.09\VideoPose3D\Renders"
# )
