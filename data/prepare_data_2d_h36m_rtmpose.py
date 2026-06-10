import os
import numpy as np
from glob import glob
from data_utils import suggest_metadata

# 完全繼承 VideoPose3D 的硬編碼相機映射
cam_map = {
    '54138969': 0,
    '55011271': 1,
    '58860488': 2,
    '60457274': 3,
}

def main():
    # 你的 RTMPose .npy 輸出路徑
    input_dir = 'D:/H36M/RTMPose_NPY'
    output_filename = 'data_2d_h36m_rtmpose.npz'

    # 宣告為 COCO 格式，讓網路自動學習映射，並啟用正確的左右翻轉對稱性
    metadata = suggest_metadata('coco')
    
    print(f'開始聚合 2D 檢測結果自: {input_dir}')
    output = {}
    
    file_list = glob(os.path.join(input_dir, 'S*', '*.npy'))
    for f in file_list:
        path, fname = os.path.split(f)
        subject = os.path.basename(path)
        
        # 檔名範例: "Eating 2.54138969.npy"
        stem = os.path.splitext(fname)[0]
        parts = stem.split('.')
        if len(parts) != 2:
            continue
            
        action, camera = parts[0], parts[1]
        
        # 壞軌剔除機制
        if subject == 'S11' and action == 'Directions':
            continue 
            
        # 動作名稱標準化 (對齊 3D GT)
        canonical_name = action.replace('TakingPhoto', 'Photo') \
                               .replace('WalkingDog', 'WalkDog')
                               
        camera_idx = cam_map.get(camera)
        if camera_idx is None:
            continue

        # 讀取 RTMPose 生成的 NumPy 陣列 shape: (N, 17, 2)
        keypoints = np.load(f)
        
        # 確保字典結構存在
        if subject not in output:
            output[subject] = {}
        if canonical_name not in output[subject]:
            output[subject][canonical_name] = [None, None, None, None]
            
        output[subject][canonical_name][camera_idx] = keypoints.astype('float32')

    print('寫入壓縮檔中...')
    np.savez_compressed(output_filename, positions_2d=output, metadata=metadata)
    print(f'成功生成 {output_filename}！')

if __name__ == '__main__':
    main()