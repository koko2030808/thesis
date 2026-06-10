import numpy as np

def atom_action_01_2_scan(path):
    print("\n>>> 原子行動 01.2：正在掃描全量序列...")
    data = np.load(path, allow_pickle=True)['positions_3d'].item()
    
    # 根據你的審計，這裡已經是 (3071, 32, 3)
    seq_32 = data['S9']['Sitting 1']
    total_frames = seq_32.shape[0]
    
    print(f"{'幀索引 (Index)':<15} | {'Hips Z 高度 (mm)':<15}")
    print("-" * 35)
    
    # 每隔 100 幀打印一次，讓你鎖定「高度暴跌」的區間
    for i in range(0, total_frames, 100):
        # Index 0 在 H3.6M 中是 Hips
        z_height = seq_32[i, 0, 2] 
        print(f"{i:<15d} | {z_height:<15.2f}")

if __name__ == "__main__":
    path = r'D:\videopose2\VideoPose3D\data\data_3d_h36m.npz'
    atom_action_01_2_scan(path)