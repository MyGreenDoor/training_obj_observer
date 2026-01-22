# SDF ベース姿勢推定の現状整理（詳細版）

## 目的・問題意識
- **目標**：latent から SDF を構成し，CAD 由来 SDF と比較して姿勢差分（R,t）を推定する．
- **現実**：推論時に pose を出すには，SDF（または SDF volume 相当の表現）を介する必要がある．
- **懸念**：`pos_map` が示す座標原点と，latent→SDF の原点が一致しない可能性がある．  

## 現在の処理フロー（主に `train_stereo_la_with_instance_seg_latent.py`）
- 入力：`stereo`, `depth`, `disparity`, `semantic_seg`, `instance_seg`（必要なら `SDFs`, `SDFs_meta`）
- モデル：`PanopticStereoMultiHeadLatent`
  - 出力：`disp_*`, `sem_logits`, `pos_mu(_norm)`, `pos_logvar(_norm)`, `aff_logits`, `emb`, `latent_map`, `sdf_map`, `sdf_logvar`
- 損失：
  - 既存：`disp`, `sem`, `pos`, `aff`, `emb`
  - 追加：SDF loss（CAD SDF との一致）
  - 追加：SDF pose loss（SDF ペアから R,t 差分推定）

## SDF loss（per-pixel）
- **入力**：`pred["point_map_1x"]`, GT `pos_map`, `rot_map`, `SDFs`, `SDFs_meta`
- **座標変換**：
  - `vec = point_map - pos_map`
  - `obj_coords = R^T * vec` （object 座標）
- **SDF sampling**：
  - `SDFs_meta` の `bbox_min/max` または `grid_*` を使い `[-1,1]^3` に正規化
  - `grid_sample` で trilinear sampling
- **損失**：heteroscedastic Laplace NLL  
  - `|e| * exp(-0.5 * logvar) + 0.5 * logvar`
- **有効条件**：
  - instance mask 内
  - 正規化範囲内  

## SDF pose 推定（R,t）
- **目的**：SDF ペアから姿勢差分（回転＋並進）を推定し，`obj_in_camera` を構成
- **SDF ペアの生成**：
  - `pred` 側：`latent_map` を instance マスクで集約 → `SDFVolumeDecoder` で SDF volume 化
  - `cad` 側：`SDFs` の canonical volume
- **ネットワーク**：`SDFPoseDeltaNet`
  - 入力：`(N,2,D,H,W)` の SDF ペア
  - 出力：`rotvec` + `Δt`
  - 回転：`rot_utils.so3_exp_batch(rotvec)`
  - 並進：`t_pred + Δt`（`t_pred` は pos_map 由来）
- **対称性**：
  - `align_pose_by_symmetry_min_rotation` で GT を pred に合わせる

## `obj_in_camera` の算出
- `R_pred` と `t_pred + Δt` から `T_cam_obj` を生成  
  `obj_in_camera_sdf = compose_T_from_Rt(R_pred, t_pred + Δt)`

## 主要関数・パス
- `train_stereo_la_with_instance_seg.py`
  - `_compute_sdf_loss`
  - `_build_sdf_pose_pairs`
  - `_compute_sdf_pose_delta`
- `train_stereo_la_with_instance_seg_latent.py`
  - `build_model`（`sdf_pose_net`, `sdf_decoder` を attach）
- `models/sdf_pose_net.py`
  - `SDFPoseDeltaNet`
  - `SDFVolumeDecoder`
- `models/panoptic_stereo.py`
  - `sdf_map`, `sdf_logvar` の head

## 形状とテンソル
- `latent_map`: `(B, C, H, W)`
- `wks_inst`: `(B, K, 1, H, W)`（instance mask）
- `latent_k`: `(B, K, C)`（instance 集約）
- `pred_vol`: `(N, 1, D, H, W)`
- `cad_vol`: `(N, 1, D, H, W)`
- `sdf_pair`: `(N, 2, D, H, W)`
- `rotvec`: `(N, 3)`
- `Δt`: `(N, 3)`

## 設定キー（例）
- `loss.w_sdf`
- `loss.w_sdf_pose`  
- `loss.w_sdf_pose_rot`
- `loss.w_sdf_pose_trans`
- `loss.sdf_max_points`
- `loss.sdf_charb_eps`
- `sdf_pose.base_ch`
- `sdf_pose.num_down`
- `sdf_pose.hidden_ch`
- `sdf_pose.out_scale_rot`
- `sdf_pose.out_scale_trans`
- `sdf_pose.decoder_base_ch`
- `sdf_pose.decoder_base_res`

## 前提・制約
- `SDFs` の順序は instance id（1..K）に一致する前提
- SDF volume の shape はバッチ内で一致する前提
- 正規化座標は `[-1,1]^3` を想定
- `la_loader` に依存（学習はローカルのみ）

## 既知の懸念
- **pos_map 原点と SDF 原点の不一致**
  - 現状：`Δt` に吸収される可能性はあるが，座標系の混乱が起きやすい
  - 対策候補：latent から中心オフセット `Δc` を回帰し，`t_pred + Δt - Δc` で整合

## 改善候補
- SDF volume 解像度の coarse-to-fine 化
- `Δt` 予測の安定化（logvar 追加，正則化）
- 対称物体の multi-hypothesis 化

## 実行・検証（注意）
- `la_loader` が必要（ローカルのみ）
- クラウド環境では静的解析のみ
