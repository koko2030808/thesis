import argparse
import numpy as np
from pathlib import Path
from mmpose.apis import MMPoseInferencer
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subjects', nargs='+', required=True, help='指派的受試者列表')
    parser.add_argument('--device', type=str, default='cuda:0', help='指定 GPU 節點')
    args = parser.parse_args()

    # 替換為你的 H36M 實際路徑
    H36M_ROOT = Path("D:/H36M") 
    out_root = H36M_ROOT / "RTMPose_NPY"
    out_root.mkdir(exist_ok=True)

    print(f"[{args.device}] 載入模型權重中...")
    inferencer = MMPoseInferencer(
        pose2d='rtmpose-m_8xb256-420e_coco-256x192', 
        det_model='rtmdet-m',
        device=args.device
    )

    for subj in args.subjects:
        video_dir = H36M_ROOT / subj / "Videos"
        if not video_dir.exists():
            continue
            
        subj_out_dir = out_root / subj
        subj_out_dir.mkdir(exist_ok=True)
        
        # 核心過濾機制：徹底濾除 _ALL 總集篇，防算力浪費
        videos = [v for v in video_dir.glob("*.mp4") if not v.name.startswith("_ALL")] + \
                 [v for v in video_dir.glob("*.avi") if not v.name.startswith("_ALL")]
        
        for vid_path in tqdm(videos, desc=f"[{args.device}] 處理 {subj}"):
            out_npy = subj_out_dir / f"{vid_path.stem}.npy"
            
            # 斷點續傳機制
            if out_npy.exists():
                continue
                
            result_generator = inferencer(str(vid_path), return_vis=False, show_progress=False)
            vid_kpts = []
            
            for frame_result in result_generator:
                preds = frame_result['predictions'][0]
                if len(preds) > 0:
                    vid_kpts.append(np.array(preds[0]['keypoints']))
                else:
                    vid_kpts.append(vid_kpts[-1] if vid_kpts else np.zeros((17, 2)))
                    
            np.save(out_npy, np.array(vid_kpts))

if __name__ == "__main__":
    main()