import numpy as np

NPZ_PATH = r'D:\04.09\VideoPose3D\Detections\vaccine_video.mp4.npz'

d = np.load(NPZ_PATH, allow_pickle=True)

print('=== 基本資訊 ===')
print('keys:', d.files)

meta = d['metadata'].item()
print(f'解析度: w={meta["w"]} h={meta["h"]}')

kps_all  = d['keypoints']
boxes_all = d['boxes']
print(f'總幀數: {len(kps_all)}')

print()
print('=== 偵測統計 ===')
detected = 0
empty    = 0
for frame_kps in kps_all:
    if len(frame_kps[1]) > 0:
        detected += 1
    else:
        empty += 1

print(f'有偵測到人的幀: {detected}')
print(f'沒有偵測到人的幀: {empty}')
print(f'偵測率: {detected/len(kps_all)*100:.1f}%')

print()
print('=== 第一個有人的幀 ===')
for i, frame_kps in enumerate(kps_all):
    if len(frame_kps[1]) > 0:
        arr = np.array(frame_kps[1])
        print(f'幀 {i}，shape: {arr.shape}')
        print('  → 偵測到人數:', arr.shape[0])
        print('  → 格式 [x, y, logit, prob] × 17 COCO joints')
        print('  第一個人的前3個關節 (nose, left_eye, right_eye):')
        print(arr[0, :, :3])  # x,y,prob for first 3 joints
        break

print()
print('=== 最後一幀 ===')
last = kps_all[-1]
if len(last[1]) > 0:
    print(f'最後一幀有偵測到人，shape: {np.array(last[1]).shape}')
else:
    print('最後一幀沒有偵測到人')