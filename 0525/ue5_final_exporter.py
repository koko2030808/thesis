import unreal
import json
import os

# ============================================================
# 配置區：V167 Socket 量產專用
# ============================================================
SEQUENCE_ASSET_PATH = "/Game/RenderPeople/VideoPoseSyntheticModel/Ch01_nonPBR/Seq_Vaccine_01"
OUTPUT_JSON_PATH    = r"D:\04.09\VideoPose3D\vaccine_data.json"
FRAME_START         = 0
FRAME_END           =  6919  # 11 幀 (0~10)
SEQUENCE_FPS        = 30

# 已依據 UI 截圖完全對齊 17 個實體 Socket 名稱
SOCKET_MAP = {
    0:  "H36M_00_Pelvis",
    1:  "H36M_01_R_Hip",
    2:  "H36M_02_R_Knee",
    3:  "H36M_03_R_Ankle",
    4:  "H36M_04_L_Hip",
    5:  "H36M_05_L_Knee",
    6:  "H36M_06_L_Ankle",
    7:  "H36M_07_Spine",
    8:  "H36M_08_Thorax",
    9:  "H36M_09_Nose",
    10: "H36M_10_Head",
    11: "H36M_11_L_Shoulder",
    12: "H36M_12_L_Elbow",
    13: "H36M_13_L_Wrist",
    14: "H36M_14_R_Shoulder",
    15: "H36M_15_R_Elbow",
    16: "H36M_16_R_Wrist",
}

# ============================================================
# 步驟一：驗證插槽 (Sockets) 是否存在
# ============================================================
def verify_sockets(smc):
    print("\n" + "="*60)
    print(f"驗證 SOCKET_MAP（共 {len(SOCKET_MAP)} 個插槽）...")
    invalid_sockets = []
    
    for joint_idx, socket_name in SOCKET_MAP.items():
        if not smc.does_socket_exist(socket_name):
            invalid_sockets.append(f"  [{joint_idx}] '{socket_name}' 不存在")

    if invalid_sockets:
        print("❌ 致命錯誤：以下插槽名稱不存在於 Skeletal Mesh 上：")
        for msg in invalid_sockets:
            print(msg)
        return False
        
    print("✅ 所有插槽名稱驗證通過")
    return True

# ============================================================
# 步驟二：Sequencer Scrub 逐幀導出
# ============================================================
def export_via_sequencer_scrub():
    print("\n" + "="*60)
    print("V167 數位雙生 - 終極插槽導出器 啟動")
    print("="*60)

    seq_asset = unreal.load_asset(SEQUENCE_ASSET_PATH)
    if seq_asset is None:
        print(f"❌ 找不到 Sequence：{SEQUENCE_ASSET_PATH}")
        return

    smc = _find_skeletal_mesh_component()
    if smc is None:
        return

    if not verify_sockets(smc):
        return

    all_frames = []
    unreal.LevelSequenceEditorBlueprintLibrary.open_level_sequence(seq_asset)

    print(f"\n開始逐幀採樣（Frame {FRAME_START} → {FRAME_END}）...\n")

    for frame_num in range(FRAME_START, FRAME_END + 1):
        _evaluate_frame(frame_num)
        
        joints_cm, diagnostics = _read_sockets_world_space(smc)
        joints_list = [joints_cm.get(i, [0.0, 0.0, 0.0]) for i in range(17)]

        frame_entry = {
            "frame":          frame_num,
            "joints_world_cm": joints_list,
            "a_pose_flag":    diagnostics["is_a_pose"],
        }
        all_frames.append(frame_entry)

        _print_frame_diagnostic(frame_num, diagnostics)

    _save_json(all_frames)
    print(f"\n✅ JSON 已成功量產保存至：{OUTPUT_JSON_PATH}")
    print("="*60)

# ============================================================
# 內部工具函數
# ============================================================
def _evaluate_frame(frame_num):
    """V167 防禦型時間軸推進器：安全解算動畫幀"""
    try:
        # 強制推進時間軸並暫停，這足以觸發骨架動畫解算
        unreal.LevelSequenceEditorBlueprintLibrary.set_current_time(frame_num)
        unreal.LevelSequenceEditorBlueprintLibrary.pause()
    except Exception:
        try:
            unreal.LevelSequenceEditorBlueprintLibrary.jump_to_frame(frame_num)
        except Exception as e:
            print(f"  ⚠️ Frame {frame_num} 推進失敗: {e}")

    # 移除會引發崩潰的 flush_level_streaming 舊版幻覺代碼
    # 改用安全的編輯器刷新 (若版本不支持亦安全跳過)
    try:
        unreal.LevelSequenceEditorBlueprintLibrary.refresh_current_level_sequence()
    except Exception:
        pass

def _read_sockets_world_space(smc):
    joints_cm = {}
    for joint_idx, socket_name in SOCKET_MAP.items():
        try:
            # 【關鍵修復】：拔除會引發崩潰的 Enum 參數。
            # UE 底層 API 預設 (Default) 即為 World Space，直接呼叫最安全。
            transform = smc.get_socket_transform(socket_name)
            loc = transform.translation
            joints_cm[joint_idx] = [
                round(loc.x, 3),
                round(loc.y, 3),
                round(loc.z, 3),
            ]
        except Exception as e:
            # 強制印出真實報錯，拒絕靜默坍縮
            print(f"❌ 警告：讀取 {socket_name} 失敗 -> {e}")
            joints_cm[joint_idx] = [0.0, 0.0, 0.0]

    diag = _compute_diagnostics(joints_cm)
    return joints_cm, diag  


def _compute_diagnostics(joints_cm):
    diag = {
        "wrist_y_spread": 0.0, "knee_z_asymmetry": 0.0,
        "is_a_pose": False,
    }
    if 13 in joints_cm and 16 in joints_cm:
        diag["wrist_y_spread"] = abs(joints_cm[13][1] - joints_cm[16][1])
    if 2 in joints_cm and 5 in joints_cm:
        diag["knee_z_asymmetry"] = abs(joints_cm[2][2] - joints_cm[5][2])
    
    if diag["wrist_y_spread"] > 70 and diag["knee_z_asymmetry"] < 5:
        diag["is_a_pose"] = True
    return diag

def _print_frame_diagnostic(frame_num, diag):
    status = "⚠️ A-POSE" if diag["is_a_pose"] else "✅ ANIM  "
    print(f"  [{status}] Frame {frame_num:3d} | Wrist_Y_Spread: {diag['wrist_y_spread']:5.1f}cm | Knee_Z_Asym: {diag['knee_z_asymmetry']:5.1f}cm")

def _save_json(all_frames):
    os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)
    data_array = [f["joints_world_cm"] for f in all_frames]
    output = {
        "metadata": {
            "method": "sequencer_scrub_socket_pose",
            "coordinate_space": "UE5_World_Space_LeftHand",
            "unit": "cm",
            "frame_count": len(all_frames),
            "fps": SEQUENCE_FPS,
        },
        "data": [data_array],
    }
    with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

def _find_skeletal_mesh_component():
    """V167 精準狙擊：鎖定 Outliner 顯示標籤"""
    editor_actor_subs = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    actors = editor_actor_subs.get_all_level_actors()
    
    for actor in actors:
        # 【關鍵修復】：使用 get_actor_label() 讀取你在介面上看到的名字
        actor_label = actor.get_actor_label()
        
        # 強迫字串比對，只允許目標人物通過
        if "Ch01_nonPBR" in actor_label:
            try:
                smc = actor.get_component_by_class(unreal.SkeletalMeshComponent)
                if smc and smc.skeletal_mesh:
                    print(f"✅ 精準鎖定目標實體：{actor_label}")
                    return smc
            except Exception:
                continue
                
    print("❌ 致命錯誤：場景中找不到標籤包含 'Ch01_nonPBR' 的實體。")
    print("   請確認目標人物在 Outliner 中的確切命名。")
    return None
# ============================================================
# 執行
# ============================================================
export_via_sequencer_scrub()