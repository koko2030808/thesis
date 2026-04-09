import math
import numpy as np
from scipy.spatial.transform import Rotation as R

# ==========================================
# 1. H3.6M 原始內參矩陣 (遺傳密碼)
# ==========================================
h36m_cameras_intrinsic_params = [
    {
        'id': '54138969',
        'focal_length': [1145.0494, 1143.7811],
        'res_w': 1000, 'res_h': 1002,
    },
    {
        'id': '55011271',
        'focal_length': [1149.6756, 1147.5916],
        'res_w': 1000, 'res_h': 1000,
    },
    {
        'id': '58860488',
        'focal_length': [1149.1407, 1148.7989],
        'res_w': 1000, 'res_h': 1000,
    },
    {
        'id': '60457274',
        'focal_length': [1145.5113, 1144.7739],
        'res_w': 1000, 'res_h': 1002,
    },
]

# ==========================================
# 2. H3.6M 原始外參矩陣 (S1-S11)
# ==========================================
h36m_cameras_extrinsic_params = {
    'S1': [
        {'orientation': [0.1407, -0.1500, -0.7552, 0.6223], 'translation': [1841.107, 4955.284, 1563.445]},
        {'orientation': [0.6157, -0.7648, -0.1483, 0.1179], 'translation': [1761.278, -5078.006, 1606.265]},
        {'orientation': [0.1465, -0.1464, 0.7653, -0.6094], 'translation': [-1846.777, 5215.046, 1491.972]},
        {'orientation': [0.5834, -0.7853, 0.1454, -0.1474], 'translation': [-1794.789, -3722.698, 1574.892]},
    ],
    'S5': [
        {'orientation': [0.1467, -0.1623, -0.7551, 0.6178], 'translation': [2097.391, 4880.944, 1605.732]},
        {'orientation': [0.6159, -0.7626, -0.1572, 0.1189], 'translation': [2031.700, -5167.933, 1612.923]},
        {'orientation': [0.1429, -0.1290, 0.7678, -0.6110], 'translation': [-1620.594, 5171.658, 1496.437]},
        {'orientation': [0.5920, -0.7814, 0.1274, -0.1503], 'translation': [-1637.173, -3867.317, 1547.033]},
    ],
    'S6': [
        {'orientation': [0.1337, -0.1569, -0.7571, 0.6198], 'translation': [1935.451, 4950.245, 1618.083]},
        {'orientation': [0.6147, -0.7628, -0.1617, 0.1181], 'translation': [1969.803, -5128.738, 1632.778]},
        {'orientation': [0.1529, -0.1352, 0.7646, -0.6112], 'translation': [-1769.596, 5185.361, 1476.993]},
        {'orientation': [0.5916, -0.7804, 0.1283, -0.1561], 'translation': [-1721.668, -3884.131, 1540.487]},
    ],
    'S7': [
        {'orientation': [0.1435, -0.1631, -0.7548, 0.6188], 'translation': [1974.512, 4926.354, 1597.832]},
        {'orientation': [0.6141, -0.7638, -0.1596, 0.1177], 'translation': [1937.058, -5119.790, 1631.566]},
        {'orientation': [0.1455, -0.1287, 0.7660, -0.6127], 'translation': [-1741.811, 5208.249, 1464.824]},
        {'orientation': [0.5912, -0.7821, 0.1244, -0.1519], 'translation': [-1734.710, -3832.421, 1548.583]},
    ],
    'S8': [
        {'orientation': [0.1411, -0.1558, -0.7561, 0.6196], 'translation': [2150.651, 4896.161, 1611.904]},
        {'orientation': [0.6169, -0.7647, -0.1484, 0.1115], 'translation': [2219.965, -5148.453, 1613.044]},
        {'orientation': [0.1471, -0.1337, 0.7670, -0.6100], 'translation': [-1571.221, 5137.018, 1498.176]},
        {'orientation': [0.5927, -0.7825, 0.1214, -0.1463], 'translation': [-1476.913, -3896.741, 1547.972]},
    ],
    'S9': [
        {'orientation': [0.1554, -0.1554, -0.7532, 0.6199], 'translation': [2044.458, 4935.117, 1481.227]},
        {'orientation': [0.6187, -0.7634, -0.1413, 0.1193], 'translation': [1990.959, -5123.810, 1568.804]},
        {'orientation': [0.1335, -0.1367, 0.7689, -0.6100], 'translation': [-1670.992, 5211.985, 1528.387]},
        {'orientation': [0.5879, -0.7823, 0.1427, -0.1479], 'translation': [-1696.043, -3827.099, 1591.412]},
    ],
    'S11': [
        {'orientation': [0.1523, -0.1544, -0.7547, 0.6191], 'translation': [2098.440, 4926.554, 1500.278]},
        {'orientation': [0.6189, -0.7600, -0.1530, 0.1255], 'translation': [2083.182, -4912.172, 1561.078]},
        {'orientation': [0.1494, -0.1565, 0.7681, -0.6026], 'translation': [-1609.815, 5177.335, 1537.896]},
        {'orientation': [0.5894, -0.7818, 0.1399, -0.1471], 'translation': [-1590.738, -3854.168, 1578.017]},
    ],
}

# ==========================================
# 3. 核心運算引擎 (First Principles)
# ==========================================
def solve_ue5_camera_v12(ext_data, int_data):
    # --- 內參換算 (解決透視不對稱病灶) ---
    sensor_width_mm = 36.0 # UE5 標準 Sensor 寬度 [cite: 2026-01-28]
    res_w = int_data['res_w']
    res_h = int_data['res_h']
    f_px = int_data['focal_length'][0]
    
    # 物理焦距 mm 與 Aspect Ratio 定影 [cite: 2026-01-28]
    ue_focal_length = f_px * (sensor_width_mm / res_w)
    aspect_ratio = res_w / res_h

    # --- 外參定影 (座標重映射) ---
    t = ext_data['translation']
    ue_loc = [t[2]/10.0, t[0]/10.0, -t[1]/10.0] # Z, X, -Y [cite: 2026-01-28]

    # --- 旋轉定影 (向量投影法) ---
    q = ext_data['orientation']
    r = R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix() # X, Y, Z, W [cite: 2026-01-28]
    
    # 提取方向向量並映射至 UE 左手系 [cite: 2026-01-28]
    fwd_ue = np.array([r[2,2], r[0,2], -r[1,2]])
    yaw = math.atan2(fwd_ue[1], fwd_ue[0]) * (180 / math.pi)
    pitch = math.atan2(fwd_ue[2], math.sqrt(fwd_ue[0]**2 + fwd_ue[1]**2)) * (180 / math.pi)

    return {
        'id': int_data['id'],
        'loc': ue_loc,
        'rot': (0.0, pitch, yaw), # 強制 Roll 歸零策略 [cite: 2026-01-28]
        'focal': ue_focal_length,
        'aspect': aspect_ratio
    }

# ==========================================
# 4. 批量執行程序
# ==========================================
print(f"--- [V12.0 畢業總表：全 28 台相機部署清單] ---")
subjects = ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11']

for sub in subjects:
    print(f"\n===== Subject: {sub} =====")
    for i in range(4):
        res = solve_ue5_camera_v12(h36m_cameras_extrinsic_params[sub][i], h36m_cameras_intrinsic_params[i])
        print(f"Cam_{i+1} (ID:{res['id']}):")
        print(f"  Location (cm): X={res['loc'][0]:.2f}, Y={res['loc'][1]:.2f}, Z={res['loc'][2]:.2f}")
        print(f"  Rotation (deg): R=0.00, P={res['rot'][1]:.2f}, Y={res['rot'][2]:.2f}")
        print(f"  Lens: Focal={res['focal']:.3f}mm | AspectRatio={res['aspect']:.3f}")