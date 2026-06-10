"""
count_frames.py
===============
目的：自動統計所有 Sitting 坐姿動畫的幀數。
支援 Binary FBX（Mixamo 預設格式）和 ASCII FBX。

MECE 架構：
    第一層：設定（路徑、參數）
    第二層：核心函數（讀取 Binary FBX 的幀數）
    第三層：主流程（批次處理所有檔案）
    第四層：輸出（統計結果 + 是否足夠的判斷）

Binary FBX 原理：
    FBX Binary 格式在檔案特定位置儲存時間資訊（KTime 單位）。
    透過搜尋特定 byte pattern 找到時間數值，轉換成幀數。
    1 秒 = 46186158000 KTime，30fps 時 1 幀 = 1539538600 KTime。
"""

import os
import glob
import struct
import re

# ============================================================
# 第一層：設定
# ============================================================

FOLDER = r"C:\GPS_Gaussian\Content\RenderPeople\VideoPoseSyntheticModel\Ch01_nonPBR"
FPS = 30
REPEAT = 10
H36M_TRAIN_FRAMES = 3126272

# ============================================================
# 第二層：核心函數
# ============================================================

def get_fbx_frames_binary(filepath, fps=30):
    """
    讀取 Binary FBX 檔案的幀數。

    原理：
        Binary FBX 把時間資訊存成 int64 數值（8 bytes）。
        搜尋 'LocalTime' 這個字串，它後面跟著的 int64 就是結束時間（KTime）。
        結束時間 / 每幀KTime = 總幀數。
    """
    KTIME_PER_SEC = 46186158000
    ktime_per_frame = KTIME_PER_SEC / fps

    try:
        with open(filepath, 'rb') as f:
            data = f.read()

        # 方法一：搜尋 "LocalTime" 字串，後面跟著 int64 結束時間
        keyword = b'LocalTime'
        idx = data.find(keyword)
        while idx != -1:
            # keyword 後面跳過一些 header bytes，找到 int64 數值
            for offset in range(1, 30):
                try:
                    pos = idx + len(keyword) + offset
                    if pos + 8 > len(data):
                        break
                    val = struct.unpack_from('<q', data, pos)[0]
                    # 合理的 KTime 範圍：0.5秒 到 300秒 的動畫
                    if KTIME_PER_SEC * 0.5 < val < KTIME_PER_SEC * 300:
                        frames = round(val / ktime_per_frame)
                        if 10 < frames < 10000:  # 合理幀數範圍
                            return frames
                except:
                    pass
            idx = data.find(keyword, idx + 1)

        # 方法二：搜尋所有合理的 int64 KTime 數值
        # 策略：找所有在 15fps~120fps、1秒~120秒 範圍內的數值
        candidates = []
        for fps_check in [30, 60, 24, 25]:
            ktime_f = KTIME_PER_SEC / fps_check
            for i in range(0, min(len(data) - 8, 500000), 4):
                try:
                    val = struct.unpack_from('<q', data, i)[0]
                    frames = val / ktime_f
                    if 30 < frames < 3000:  # 1秒到100秒，合理範圍
                        candidates.append(round(frames))
                except:
                    pass

        if candidates:
            # 取最常出現的幀數（通常是正確答案）
            from collections import Counter
            most_common = Counter(candidates).most_common(1)[0][0]
            return most_common

    except Exception as e:
        pass

    return None


def get_fbx_frames_ascii(filepath, fps=30):
    """讀取 ASCII FBX 檔案的幀數（備用）。"""
    KTIME_PER_SEC = 46186158000
    ktime_per_frame = KTIME_PER_SEC / fps
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read(100000)
        match = re.search(r'LocalTime:\s*\d+,(\d+)', content)
        if match:
            return round(int(match.group(1)) / ktime_per_frame)
        match = re.search(r'ReferenceTime:\s*\d+,(\d+)', content)
        if match:
            return round(int(match.group(1)) / ktime_per_frame)
    except:
        pass
    return None


def get_fbx_frames(filepath, fps=30):
    """
    自動判斷 FBX 格式（Binary 或 ASCII），回傳幀數。
    Binary FBX 的前 4 bytes 是 'Kaydara' 開頭的 magic number。
    """
    try:
        with open(filepath, 'rb') as f:
            magic = f.read(7)
        if magic == b'Kaydara':
            # Binary FBX
            return get_fbx_frames_binary(filepath, fps)
        else:
            # ASCII FBX
            return get_fbx_frames_ascii(filepath, fps)
    except:
        return None


# ============================================================
# 第三層：主流程
# ============================================================

def main():
    files = glob.glob(os.path.join(FOLDER, "*[Ss]it*.fbx"))
    files = sorted(set(files))

    if not files:
        print(f"找不到任何 Sitting 動畫，請確認路徑：{FOLDER}")
        return

    total_frames = 0
    results = []

    for f in files:
        name = os.path.basename(f)
        frames = get_fbx_frames(f, fps=FPS)
        if frames is not None:
            total_frames += frames
            results.append((name, frames))
        else:
            results.append((name, None))

    # ============================================================
    # 第四層：輸出
    # ============================================================

    print(f"\n找到 {len(files)} 個 Sitting 動畫\n")
    print(f"{'動畫名稱':<55} {'幀數':>6}")
    print("-" * 65)

    for name, frames in results:
        if frames is not None:
            print(f"{name:<55} {frames:>6} 幀")
        else:
            print(f"{name:<55} {'無法讀取':>6}")

    print("-" * 65)
    print(f"{'原始合計':<55} {total_frames:>6} 幀")

    repeated_frames = total_frames * REPEAT
    ratio = repeated_frames / H36M_TRAIN_FRAMES * 100

    print(f"\nREPEAT={REPEAT} 後：{repeated_frames} 幀")
    print(f"佔 H36M 訓練集的比例：{ratio:.2f}%")
    print()

    if ratio < 1.0:
        print("比例低於 1%，fine-tuning 效果有限，建議繼續增加動畫")
    elif ratio < 5.0:
        print("比例介於 1-5%，可以試跑，效果可能有限")
    else:
        print("比例超過 5%，預期可以看到明顯改善")


if __name__ == "__main__":
    main()
