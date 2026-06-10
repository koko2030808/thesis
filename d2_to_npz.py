"""
d2_to_npz.py
============
Detectron2 output → data_2d_vaccine_detectron.npz

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AI BUILDER MECE：這支腳本在管線中的角色
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

這支腳本是「格式轉換器」，不是「演算法」。
它唯一的責任是：把 Detectron2 的輸出，
轉換成 VideoPose3D fine-tuning 期望的格式。

INPUT  → vaccine_video.mp4.npz   (Detectron2 格式，pixel座標)
OUTPUT → data_2d_vaccine_detectron.npz  (VideoPose3D 格式，normalized)

判斷這支腳本寫得好不好的標準：
  ✅ 輸出能被 finetune_v2.py 直接讀取，不報錯
  ✅ 值域在 [-1, 1]，和 H36M 訓練數據一致
  ✅ 幀數對齊 3D GT，不差幀

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
三個核心設計決策（AI Builder 必須理解的 WHY）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

決策 1：normalize 公式 x/500-1
  WHY：H36M 的 normalize_screen_coordinates 公式是 X/w*2 - [1, h/w]
       我們的圖像是 1000×1000，所以 h/w=1
       代入：X/1000*2 - 1 = X/500 - 1
  FAILURE：如果用錯公式（例如 x/1000），
           模型會收到分佈偏移的輸入，所有預測會系統性跑偏

決策 2：取「最高信心」的人，而非「最大 bounding box」的人
  WHY：Detectron2 可能偵測到背景雜物。信心分數更可靠。
       我們的場景只有 1 人，所以 argmax 就是那個人。
  FAILURE：如果場景有多人，這個邏輯會挑到不一致的人，
           導致 2D 序列在不同幀之間跳到不同人身上

決策 3：空幀用「前一幀插值」而非填零
  WHY：VideoPose3D 是時序模型（感受野 243 幀）。
       填零會在序列中插入一個不真實的 spike，
       讓模型在空幀前後的預測都受污染。
       前一幀插值假設「短暫遮擋時姿勢不變」，更合理。
  FAILURE：如果連續多幀都是空幀（例如鏡頭切換），
           插值會把舊姿勢拉到很遠的地方，這時需要後插值（用後一幀）
"""

import numpy as np

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 路徑設定
# AI BUILDER NOTE：把所有路徑集中在一個地方。
# 不要把路徑硬寫在函數裡面。
# 原因：要換環境（例如從 D 槽換到 E 槽）時，
#       只需要改這裡，不需要在整個 code 裡搜尋。
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
D2_NPZ  = r'D:\04.09\VideoPose3D\Detections\vaccine_video.mp4.npz'
REF_NPZ = r'D:\04.09\VideoPose3D\data\data_2d_vaccine_ue5.npz'
OUT_NPZ = r'D:\04.09\VideoPose3D\data\data_2d_vaccine_detectron.npz'
IMAGE_W = IMAGE_H = 1000


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 1：理解 REF 格式
#
# AI BUILDER PATTERN：「先讀合約，再寫輸出」
#
# 我們不知道 data_2d_vaccine_ue5.npz 的確切結構，
# 不要猜。先讀它，讓程式自己告訴我們格式。
# 然後我們輸出同樣格式，保證下游可以讀。
#
# FAILURE MODE：如果跳過這步直接存，
# 格式不對時 finetune_v2.py 會在讀取時爆掉，
# 而且錯誤訊息很難追蹤（因為是在完全不同的腳本裡）。
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print('=== [1/3] 讀取現有格式 ===')
ref     = np.load(REF_NPZ, allow_pickle=True)
ref_key = ref.files[0]
ref_val = ref[ref_key]

# numpy 把 dict 包在 0-dim array 裡，.item() 解包回 Python dict
# AI BUILDER NOTE：這是 numpy savez 儲存 dict 時的設計限制，
# 不是 bug，是 feature。記住這個 pattern。
if hasattr(ref_val, 'item'):
    ref_val = ref_val.item()

print(f'REF key: {ref_key}')


def print_struct(d, indent=0):
    """遞迴印出 nested dict 的結構。
    AI BUILDER：這種 debug 函數在你開發 data pipeline 時
    比 print(d) 有用 10 倍。把它放進你的工具箱。"""
    if isinstance(d, dict):
        for k, v in d.items():
            print(' ' * indent + f'[{k}]')
            print_struct(v, indent + 4)
    elif isinstance(d, list):
        print(' ' * indent + f'list (len={len(d)})')
        if len(d) > 0:
            print_struct(d[0], indent + 4)
    elif hasattr(d, 'shape'):
        print(' ' * indent + f'array shape={d.shape} dtype={d.dtype}')
    else:
        print(' ' * indent + f'type={type(d).__name__}')

print_struct(ref_val)


def get_leaf_path(d, path=()):
    """遞迴找到最深的 numpy array 和它的路徑。
    
    AI BUILDER CONCEPT：「leaf node」
    在 nested dict 裡，只有最深層的 array 才是真正的數據。
    上層的 dict keys 只是索引（subject / action / camera）。
    我們要換的是 leaf，不是整棵樹。
    
    注意：H36M 格式的相機維度是 list，不是 dict，所以要處理 list。
    """
    if isinstance(d, dict):
        for k, v in d.items():
            result = get_leaf_path(v, path + (k,))
            if result is not None:
                return result
    elif isinstance(d, list):
        for idx, v in enumerate(d):
            result = get_leaf_path(v, path + (idx,))
            if result is not None:
                return result
    elif hasattr(d, 'shape'):
        return path, d
    return None

leaf_path, leaf_arr = get_leaf_path(ref_val)
print(f'\n找到 leaf: path={leaf_path}, shape={leaf_arr.shape}')


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 2：萃取 Detectron2 關節點
#
# AI BUILDER PATTERN：「明確追蹤 data shape」
#
# Detectron2 輸出的 keypoints 格式：
#   kps_all[frame_i]        → list of 2 items: [class_0_empty, class_1_data]
#   kps_all[frame_i][1]     → 真正的資料（class 1 = person）
#   kps_all[frame_i][1]     → shape: (n_people, 4, 17)
#                                      ↑        ↑   ↑
#                                   偵測到幾人  通道  COCO關節數
#
# 4 個通道的含義：
#   [0] = x 座標（pixel）
#   [1] = y 座標（pixel）
#   [2] = logit（我們不用，這是 Detectron1 相容用的 dummy）
#   [3] = prob（信心分數，我們用這個選人）
#
# FAILURE MODE：如果搞錯 axis，
# 例如 kps[:, :, 3] 而非 kps[:, 3, :]，
# 結果不會報錯，但會挑到完全不同的數字（靜默錯誤，最危險）。
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print('\n=== [2/3] 萃取 Detectron2 關節點 ===')
d2      = np.load(D2_NPZ, allow_pickle=True)
kps_all = d2['keypoints']
N       = len(kps_all)
print(f'Detectron2 幀數: {N}')

poses_2d     = np.zeros((N, 17, 2), dtype=np.float32)
empty_frames = []

for i, frame_kps in enumerate(kps_all):
    cls_keyps = frame_kps[1]           # 取 class 1（person），跳過 class 0（背景）

    # ── 空幀處理 ──────────────────────────────────────────────────
    # AI BUILDER NOTE：空幀是 pipeline 的「例外情況」。
    # 每個 pipeline 都要明確定義例外情況的處理策略。
    # 策略選項：填零 / 前一幀插值 / 後一幀插值 / 線性插值
    # 我們選「前一幀插值」，理由見檔頭 MECE。
    if len(cls_keyps) == 0:
        empty_frames.append(i)
        if i > 0:
            poses_2d[i] = poses_2d[i - 1]
        continue

    kps = np.array(cls_keyps)          # shape: (n_people, 4, 17)

    # ── 多人場景：選最高信心的人 ─────────────────────────────────
    # AI BUILDER NOTE：即使你的場景只有 1 人，
    # 也要寫「最高信心」的邏輯，而非 kps[0]。
    # 因為 Detectron2 有時會把椅子、桌子誤判為人。
    # argmax 在 1 人時等於 kps[0]，但在誤判時會救你。
    mean_conf = kps[:, 3, :].mean(axis=1)     # axis=1 = 對 17 關節取平均
    best      = np.argmax(mean_conf)
    xy        = kps[best, :2, :].T            # shape: (2, 17) → 轉置 → (17, 2)

    # ── Normalize：pixel座標 → [-1, 1] ───────────────────────────
    # 公式推導：H36M normalize_screen_coordinates = X/w*2 - [1, h/w]
    # 代入 w=1000, h=1000 → h/w=1 → x方向: X/500-1, y方向: Y/500-1
    # 兩個方向公式相同（因為正方形圖像）
    # 如果圖像不是正方形，x 和 y 的分母會不同！
    xy[:, 0] = xy[:, 0] / (IMAGE_W / 2.0) - 1.0
    xy[:, 1] = xy[:, 1] / (IMAGE_H / 2.0) - 1.0

    poses_2d[i] = xy

# ── 值域驗證 ────────────────────────────────────────────────────────
# AI BUILDER PATTERN：「pipeline 的每個輸出都要有值域驗證」
# 如果值域不在 [-1, 1]，下游模型收到的輸入會超出訓練分佈。
# 這不會報錯，但 MPJPE 會莫名其妙變差。
print(f'空幀數: {len(empty_frames)} → 已用前一幀插值')
print(f'poses_2d shape: {poses_2d.shape}')
print(f'值域: [{poses_2d.min():.3f}, {poses_2d.max():.3f}]')
if poses_2d.min() < -1.5 or poses_2d.max() > 1.5:
    print('⚠️  值域超出預期！請檢查 normalize 公式或 IMAGE_W 設定')
else:
    print('✅ 值域正常')


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 3：用和 REF 完全相同的結構存檔
#
# AI BUILDER PATTERN：「格式繼承，而非格式複製」
#
# 錯誤做法：手動寫 {'S_vaccine': {'vaccine_sitting': {'cam': poses_2d}}}
#   問題：如果 json_to_npz_v2.py 改變了 key 名稱，這裡就會不一致。
#
# 正確做法：讀 REF 的結構，只替換最深的 array。
#   replace_leaf 把 dict 的結構複製過來，只換數據。
#   這樣 key 永遠和 REF 一致，不管 REF 怎麼變。
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print('\n=== [3/3] 存檔 ===')


def replace_leaf(d, new_arr):
    """遞迴把 nested dict 最深的 array 換成 new_arr。
    
    AI BUILDER CONCEPT：「結構不變，數據替換」
    這是 data pipeline 中非常常見的 pattern：
    你想改變數據，但保持外層的 metadata 結構。
    把這個 pattern 記起來，你會一直用到它。
    
    注意：H36M 相機維度是 list，所以也要處理 list。
    """
    if isinstance(d, dict):
        return {k: replace_leaf(v, new_arr) for k, v in d.items()}
    elif isinstance(d, list):
        return [replace_leaf(v, new_arr) for v in d]
    elif hasattr(d, 'shape'):
        return new_arr    # 找到 leaf，換掉
    return d

out_val = replace_leaf(ref_val, poses_2d)

# dtype=object：儲存 nested dict 時的必要設定
# AI BUILDER NOTE：numpy 預設只能存形狀規則的 array。
# 嵌套 dict 是不規則結構，必須用 dtype=object 繞過這個限制。
np.savez_compressed(OUT_NPZ, **{ref_key: np.array(out_val, dtype=object)})


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 驗證：重新讀取，確認輸出和預期一致
#
# AI BUILDER PATTERN：「存完立刻讀回來驗證」
# 不要相信寫入成功就代表讀取成功。
# np.savez 寫入成功，不代表 np.load 讀到你以為的格式。
# 特別是 dtype=object 的情況，格式容易在存取之間悄悄變形。
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
check     = np.load(OUT_NPZ, allow_pickle=True)
check_val = check[ref_key]
if hasattr(check_val, 'item'):
    check_val = check_val.item()

_, leaf_arr2 = get_leaf_path(check_val)
print(f'驗證 → shape={leaf_arr2.shape}，期望={poses_2d.shape}')
assert leaf_arr2.shape == poses_2d.shape, '❌ 存取後 shape 不一致！'
print(f'[✅] 已存至: {OUT_NPZ}')


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 幀數對齊報告（給下一步 fine-tune 的指示）
#
# AI BUILDER PATTERN：「腳本應該告訴你下一步要做什麼」
# 好的 pipeline 腳本不只做事，還輸出狀態報告。
# 這樣你不需要記住每個腳本的輸出含義。
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print('\n=== 幀數對齊報告 ===')
n_3d = leaf_arr.shape[0]
n_2d = N
print(f'  3D GT（REF）:     {n_3d} 幀')
print(f'  Detectron2（OUT）: {n_2d} 幀')

if n_3d == n_2d:
    print('  ✅ 幀數一致，fine-tune 時不需要截幀')
else:
    diff = abs(n_3d - n_2d)
    shorter = 'Detectron2' if n_2d < n_3d else '3D GT'
    print(f'  ⚠️  差 {diff} 幀（{shorter} 較短）')
    print(f'  → 在 finetune_v2.py 裡，將較長的那份截成 {min(n_3d, n_2d)} 幀')
    print(f'     seq_3d = seq_3d[:{min(n_3d, n_2d)}]')
    print(f'     seq_2d = seq_2d[:{min(n_3d, n_2d)}]')