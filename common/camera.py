# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
import torch

from common.utils import wrap
from common.quaternion import qrot, qinverse

def normalize_screen_coordinates(X, w, h): 
    assert X.shape[-1] == 2
    
    # Normalize so that [0, w] is mapped to [-1, 1], while preserving the aspect ratio
    return X/w*2 - [1, h/w]

    
def image_coordinates(X, w, h):
    assert X.shape[-1] == 2
    
    # Reverse camera frame normalization
    return (X + [1, h/w])*w/2
    

def world_to_camera(X, R, t):
    Rt = wrap(qinverse, R) # Invert rotation
    return wrap(qrot, np.tile(Rt, (*X.shape[:-1], 1)), X - t) # Rotate and translate

    
def camera_to_world(X, R, t):
    return wrap(qrot, np.tile(R, (*X.shape[:-1], 1)), X) + t

    
def project_to_2d(X, camera_params):
    """
    Project 3D points to 2D using the Human3.6M camera projection function.
    This is a differentiable and batched reimplementation of the original MATLAB script.
    
    Arguments:
    X -- 3D points in *camera space* to transform (N, *, 3)
    camera_params -- intrinsic parameteres (N, 2+2+3+2=9)
    """
    assert X.shape[-1] == 3
    assert len(camera_params.shape) == 2
    assert camera_params.shape[-1] == 9
    assert X.shape[0] == camera_params.shape[0]
    
    while len(camera_params.shape) < len(X.shape):
        camera_params = camera_params.unsqueeze(1)
        
    f = camera_params[..., :2]
    c = camera_params[..., 2:4]
    k = camera_params[..., 4:7]
    p = camera_params[..., 7:]
    
    XX = torch.clamp(X[..., :2] / X[..., 2:], min=-1, max=1)
    r2 = torch.sum(XX[..., :2]**2, dim=len(XX.shape)-1, keepdim=True)

    radial = 1 + torch.sum(k * torch.cat((r2, r2**2, r2**3), dim=len(r2.shape)-1), dim=len(r2.shape)-1, keepdim=True)
    tan = torch.sum(p*XX, dim=len(XX.shape)-1, keepdim=True)

    XXX = XX*(radial + tan) + p*r2
    
    return f*XXX + c

def project_to_2d_linear(X, camera_params):
    """
    Project 3D points to 2D using only linear parameters (focal length and principal point).
    
    Arguments:
    X -- 3D points in *camera space* to transform (N, *, 3)
    camera_params -- intrinsic parameteres (N, 2+2+3+2=9)
    """
    assert X.shape[-1] == 3
    assert len(camera_params.shape) == 2
    assert camera_params.shape[-1] == 9
    assert X.shape[0] == camera_params.shape[0]
    
    while len(camera_params.shape) < len(X.shape):
        camera_params = camera_params.unsqueeze(1)
        
    f = camera_params[..., :2]
    c = camera_params[..., 2:4]
    
    XX = torch.clamp(X[..., :2] / X[..., 2:], min=-1, max=1)
    
    return f*XX + c

def normalize_sequence_fixed_scale(X_seq, ref_frame_idx=0):
    """
    【新部署邏輯】序列固定基準歸一化
    X_seq shape: (num_frames, num_joints, 2)
    """
    assert X_seq.shape[-1] == 2
    
    # 1. 取得基準影格（通常是第一幀站立姿態）計算縮放因子
    ref_frame = X_seq[ref_frame_idx]
    y_min_ref, y_max_ref = ref_frame[:, 1].min(), ref_frame[:, 1].max()
    ref_height = y_max_ref - y_min_ref
    
    # 2. 定義統一的縮放比例（將基準高度縮放到 2.0）
    fixed_scale = 2.0 / (ref_height + 1e-6)
    
    # 3. 對整段序列套用相同比例
    X_normalized = np.copy(X_seq)
    for f in range(X_seq.shape[0]):
        # 以每幀的 Hip (索引 0) 為中心平移，消除絕對座標位移
        root_x, root_y = X_seq[f, 0, 0], X_seq[f, 0, 1]
        
        # 套用固定比例，保留下蹲導致的高度變化
        X_normalized[f, :, 0] = (X_seq[f, :, 0] - root_x) * fixed_scale
        X_normalized[f, :, 1] = (X_seq[f, :, 1] - root_y) * fixed_scale
        
    return X_normalized
