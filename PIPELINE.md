# PIPELINE.md — Repo 地圖
更新：2026/06/11 ｜ 放在 D:\04.09\VideoPose3D\ 根目錄
用途：回答「這個檔案是什麼、還能不能刪、改它會不會死」三個問題。

---

## B｜現役管線（B-explicit，刪掉會擋住畢業）

| 檔案 | 角色 | 狀態 |
|---|---|---|
| run_rtmpose_h36m.py | 雙卡 RTMPose 跑 H36M 2D 特徵 | 🔄 執行中 |
| prepare_data_2d_h36m_rtmpose.py | .npy 聚合 → data_2d_h36m_rtmpose.npz | ⏳ RTMPose 跑完後 |
| rescale_and_reproject.py | 疫苗骨長 rescale + 重投影（2D/3D 一致） | ⏳ 下一步 |
| bone_campare.py | 骨長對照表（③ gap 數據，答洪老師）⚠️檔名 typo | ✅ 已產出 |
| eval_by_camera.py | 逐相機 MPJPE/CV 評估主力（口委數據來源） | 常駐 |
| json_to_npz_v2.py | 疫苗 JSON→NPZ（j0 格式語義定義在此：j0=絕對相機座標，j1-16=root-relative） | 常駐參考 |
| finetune_v2.py | v2 成功配方 → W3 改造成 B-explicit fine-tune 的模板 | W3 主角 |
| run.py | 官方入口：W2 重訓 baseline + W4 --render 比較影片 | W2/W4 主角 |

## C｜UE5 疫苗生產線（已完成，需要重生成疫苗時才碰）

| 檔案 | 角色 |
|---|---|
| 0525/ue5_final_exporter.py | Sequencer 逐幀收割 17 socket → vaccine_data.json |
| 0525/vacine.py | 3D 投影 → vaccine_data_2d.json（量產模式）⚠️檔名 typo |
| project_2d_json.py | 同上功能的另一版本（重複，留一個即可） |
| 0525/project_v167.py | 投影視覺驗證（綠點貼渲染圖） |
| render_headless.py | UE5 無頭渲染產線 |
| 0525/mrq_warmup_fix.py | MRQ 暖機歸零（時序對齊修復） |
| 0525/Validator_3D.py | 疫苗 3D 骨架視覺驗證 |
| fix_camera.py | 相機 json 複製 6920 幀（一次性，已完成） |

## D｜化石層（指向 D:\videopose2 舊環境，物理上不可執行 → archive/）

**0107-0112｜情況 A 時代（= 主題 B 答洪老師的歷史素材）**
- 0107/check_cpn_skeleton.py — COCO→H36M mapping 首次驗證 ⭐ 此 mapping 函數 W2 直接複用
- 0108/check_new_normalization.py — 空檔案
- 0112/check_normalization.py — 情況 A 歸一化對比（皮爾森相關驗證）
- 0112/verify_squat.py — 蹲下動作三聯動渲染驗證

**0116｜UE5→H36M 對齊演進（V16~V167）**
- alligned_check.py / produced_alligned_system.py / produced_alligned_system_check.py — 對齊引擎三版本
- standardize.py — 逐骨強制改 H36M 平均長度 ⭐ 今天 rescale 的前身（重新發明的證據）
- ue_h36m_bridge.py — 逐骨比例縮放橋接
- check.py / comparison.py / vqr.py — 時序審計 + 對比 + 動畫驗證

**0127-0129｜相機與重投影校準（V99→V145）**
- 0127/s9_params.py — H36M 28 台相機參數 → UE5 部署表（數位雙生基石）
- 0129/check_ue_projection.py / draw_points.py / draw_points_s9.py — 投影校準演進

**0223｜Mixamo→S9 retargeting（V42/V44）**
- alignment.py / s9_sitting_alignment.py / s9_sitting_alignment_vis.py / sitting_validation.py — retargeting 演進
- s9_sitting.py / s9_sitting_visualization.py — S9 數據掃描與視覺化
- s9_sitting_video.py — 影片幀定位工具（HUD）

**0411｜Socket 方案設計期**
- extract_s9_bones.py / s9_bones.py — S9 骨長 + socket 偏移計算（兩檔高度重複）
- check_mixamo_to_s9.py / check_official_s9.py — 骨長驗證
- ue_collect.py — UE5 單幀 socket 收割
- blueprint_parts/blueprint.py — 藍圖文本切分

**v3 死路三件套（負面結果 = 論文素材，勿刪）**
- finetune_v3.py / d2_to_npz.py / verify_detections.py / inference/infer_video_d2.py

**診斷考古（邏輯已被 eval_by_camera.py 吸收）**
- diagnose.py / diagnose_all.py / diagnose_camera.py / easy_diagnose.py

**一次性探針**
- probe.py / probe2.py / checking_sitting.py / inspect_skeleton.py / count_frames.py

**UE5 逆向工具**
- AST_Parser.py — 藍圖 T3D 剪貼簿解析
- Log_to_JSON.py — Log 收割（已被 ue5_final_exporter.py 取代）

## E｜危險項（拆彈清單）

| 檔案 | 危險 | 動作 |
|---|---|---|
| finetune_part.py | 檔名說 finetune，**實際是生成病灶 GIF**——誤跑會以為在訓練 | 改名 make_worstcase_gif.py |
| rescale_vaccine_bones.py | 已知 j0 格式 bug 的舊版，與新版並存 | 刪（git 保底） |
| common/camera.py | 被注入 normalize_sequence_fixed_scale = **情況 A 的物理殘留** | W2 拆除清單第 1 項 |
| run.py L121 | 被註解的 normalize_sequence_fixed_scale 呼叫 = 情況 A 犯罪現場 | W2 確認保持註解/刪除 |

## A｜官方核心（永不修改）
run.py（入口）、common/ 13 檔、data/prepare_*.py、inference/infer_video.py

---

## 維護協議（控熵於源頭，零邊際成本）

1. 每個新腳本第一行：`# ROLE: pipeline | probe | tool | archive`
2. 每週五跟 Ben 報告前：`git add -A && git commit -m "週次快照"`
3. 一個功能第二次被寫出來時，停下來搜 archive/——SOP 第三步「找最近現有資源」的觸發器
