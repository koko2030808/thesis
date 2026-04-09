import numpy as np
import matplotlib.pyplot as plt

def visual_sovereignty_audit(path):
    print("\n>>> 啟動視覺化審計：正在提取 S9 物理數據...")
    data = np.load(path, allow_pickle=True)['positions_3d'].item()
    seq_32 = data['S9']['Sitting 1'] # (3071, 32, 3)
    
    # A. 繪製 Hips Z 軸高度曲線 (1D 趨勢驗證)
    z_heights = seq_32[:, 0, 2] # 提取所有幀的 Hips Z
    
    plt.figure(figsize=(12, 5))
    plt.plot(z_heights, color='blue', label='Hips Z-Height (m)')
    plt.axvline(x=0, color='green', linestyle='--', label='T-Pose (Frame 0)')
    plt.axvline(x=1200, color='red', linestyle='--', label='Target Sitting (Frame 1200)')
    plt.title("S9 Sitting 1: Hips Vertical Trajectory")
    plt.xlabel("Frame Index"); plt.ylabel("Height (meters)")
    plt.grid(True); plt.legend(); plt.show()
    
    # B. 繪製 3D 骨架預覽 (3D 姿態驗證)
    # 定義 32 關節中的核心連結 (簡化版供預覽)
    edges = [(0,1), (1,2), (2,3), (0,4), (4,5), (5,6), (0,7), (7,8), (8,9), (8,11), (11,12), (12,13), (8,14), (14,15), (15,16)]
    
    fig = plt.figure(figsize=(12, 6))
    for i, idx in enumerate([0, 1200]):
        ax = fig.add_subplot(1, 2, i+1, projection='3d')
        pose = seq_32[idx]
        ax.scatter(pose[:,0], pose[:,1], pose[:,2], c='blue', s=20)
        for e in edges:
            ax.plot(pose[e, 0], pose[e, 1], pose[e, 2], color='black')
        ax.set_title(f"Frame {idx}: {'T-Pose' if idx==0 else 'Sitting'}")
        ax.set_zlim(0, 1.8); ax.set_box_aspect([1,1,1])
    plt.show()

if __name__ == "__main__":
    path = r'D:\videopose2\VideoPose3D\data\data_3d_h36m.npz'
    visual_sovereignty_audit(path)