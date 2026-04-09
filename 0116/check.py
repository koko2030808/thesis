import numpy as np
import matplotlib.pyplot as plt

def temporal_rigorous_audit(data_path):
    data = np.load(data_path, allow_pickle=True)['positions_3d'].item()
    sequence = data[list(data.keys())[0]]
    
    # 計算全序列速度 (影格間位移) [cite: 2026-01-20]
    velocity = np.linalg.norm(np.diff(sequence, axis=0), axis=-1)
    avg_velocity = np.mean(velocity, axis=1) # 17個點的平均速度
    
    plt.figure(figsize=(10, 4))
    plt.plot(avg_velocity, label='Mean Velocity (m/frame)')
    plt.axhline(y=np.mean(avg_velocity) + 3*np.std(avg_velocity), color='r', linestyle='--', label='3-Sigma Threshold')
    plt.title("Temporal Continuity Audit (Motion Stability)")
    plt.xlabel("Frame Index")
    plt.ylabel("Velocity")
    plt.legend()
    plt.show()

    # 如果沒有超過 3-Sigma 的突跳，代表數據極其穩定 [cite: 2026-01-20]
    print(f">>> 時域穩定性評估完成。最高速度突跳: {np.max(avg_velocity):.4f} m/frame")

temporal_rigorous_audit("final_v9_rigid_dataset.npz")