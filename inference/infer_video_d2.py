# Copyright (c) 2018-present, Facebook, Inc.
# Modified: 加入 os.makedirs 防呆 + dtype=object 修正

"""
=== MECE 架構分析 ===

任務本質：把影片的每一幀送進 Detectron2，取得 2D 關節點，存成 NPZ。

拆解成四層：
  Layer 1（輸入）：讀影片 → 逐幀 BGR 圖像
  Layer 2（推理）：Detectron2 → bounding box + 17個 COCO 關節點
  Layer 3（整理）：把不規則資料包成 numpy object array
  Layer 4（輸出）：存成 .npz，供後續 prepare_data_2d_custom.py 使用

已知限制：
  - 每幀只取最高信心的人（best_match）
  - 沒偵測到人的幀：keypoints = []，bbox = []（後續會做插值）
  - 輸出格式模仿 Detectron1（cls_boxes, cls_keyps）
"""

import detectron2
from detectron2.utils.logger import setup_logger
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor

import subprocess as sp
import numpy as np
import time
import argparse
import sys
import os
import glob


# ============================================================
# Layer 0：參數解析
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end inference')
    parser.add_argument('--cfg', dest='cfg',
                        help='cfg model file，例如 COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml',
                        default=None, type=str)
    parser.add_argument('--output-dir', dest='output_dir',
                        help='輸出目錄（自動建立）',
                        default='/tmp/infer_simple', type=str)
    parser.add_argument('--image-ext', dest='image_ext',
                        help='影片副檔名（處理資料夾時用）',
                        default='mp4', type=str)
    parser.add_argument('im_or_folder',
                        help='影片路徑 或 影片資料夾路徑',
                        default=None)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


# ============================================================
# Layer 1：影片讀取工具
# ============================================================
def get_resolution(filename):
    """用 ffprobe 取得影片解析度"""
    command = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
               '-show_entries', 'stream=width,height', '-of', 'csv=p=0', filename]
    pipe = sp.Popen(command, stdout=sp.PIPE, bufsize=-1)
    for line in pipe.stdout:
        w, h = line.decode().strip().split(',')
        return int(w), int(h)


def read_video(filename):
    """用 ffmpeg 逐幀讀取影片，每幀回傳 BGR numpy array"""
    w, h = get_resolution(filename)
    command = ['ffmpeg', '-i', filename,
               '-f', 'image2pipe', '-pix_fmt', 'bgr24',
               '-vsync', '0', '-vcodec', 'rawvideo', '-']
    pipe = sp.Popen(command, stdout=sp.PIPE, bufsize=-1)
    while True:
        data = pipe.stdout.read(w * h * 3)
        if not data:
            break
        yield np.frombuffer(data, dtype='uint8').reshape((h, w, 3))


# ============================================================
# Layer 2：Detectron2 推理（主函數）
# ============================================================
def main(args):

    # 【修正點 1】自動建立輸出目錄，防止 FileNotFoundError
    os.makedirs(args.output_dir, exist_ok=True)

    # 載入 Detectron2 模型
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(args.cfg))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(args.cfg)
    predictor = DefaultPredictor(cfg)

    # 決定要處理哪些影片
    if os.path.isdir(args.im_or_folder):
        im_list = glob.iglob(args.im_or_folder + '/*.' + args.image_ext)
    else:
        im_list = [args.im_or_folder]

    for video_name in im_list:
        out_name = os.path.join(args.output_dir, os.path.basename(video_name))
        print('Processing {}'.format(video_name))

        boxes = []
        segments = []
        keypoints = []

        # Layer 2：逐幀推理
        for frame_i, im in enumerate(read_video(video_name)):
            t = time.time()
            outputs = predictor(im)['instances'].to('cpu')
            print('Frame {} processed in {:.3f}s'.format(frame_i, time.time() - t))

            has_bbox = False
            if outputs.has('pred_boxes'):
                bbox_tensor = outputs.pred_boxes.tensor.numpy()
                if len(bbox_tensor) > 0:
                    has_bbox = True
                    scores = outputs.scores.numpy()[:, None]
                    bbox_tensor = np.concatenate((bbox_tensor, scores), axis=1)

            if has_bbox:
                kps = outputs.pred_keypoints.numpy()
                kps_xy = kps[:, :, :2]
                kps_prob = kps[:, :, 2:3]
                kps_logit = np.zeros_like(kps_prob)  # Dummy（Detectron1 相容）
                kps = np.concatenate((kps_xy, kps_logit, kps_prob), axis=2)
                kps = kps.transpose(0, 2, 1)
            else:
                # 沒偵測到人：空 list，後續 prepare_data_2d_custom.py 會做插值
                kps = []
                bbox_tensor = []

            # 模仿 Detectron1 格式：[[], actual_data]
            cls_boxes = [[], bbox_tensor]
            cls_keyps = [[], kps]

            boxes.append(cls_boxes)
            segments.append(None)
            keypoints.append(cls_keyps)

        # Layer 3：整理成可存儲格式
        metadata = {
            'w': im.shape[1],
            'h': im.shape[0],
        }

        # 【修正點 2】boxes/keypoints 是不規則形狀（有幀有人、有幀沒人）
        # 原版直接 savez 會爆，需先包成 dtype=object 的 numpy array
        datas = {
            'boxes':     boxes,
            'segments':  segments,
            'keypoints': keypoints,
            'metadata':  metadata,
        }
        for key, value in datas.items():
            datas[key] = np.array(value, dtype=object)

        # Layer 4：存成 NPZ
        np.savez_compressed(out_name, **datas)
        print(f'[✅] 已存至：{out_name}.npz')


# ============================================================
# 執行
# ============================================================
if __name__ == '__main__':
    setup_logger()
    args = parse_args()
    main(args)