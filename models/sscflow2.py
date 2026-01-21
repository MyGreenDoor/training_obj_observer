
from typing import Tuple, Dict, Callable, List, Union, Optional, Sequence

from pytorch3d.structures import Meshes
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.multi_head import (
    LiteFPNMultiTaskHead,
    TransformerMultiTaskHead,
    extract_instances_from_head_vec,
    pos_mu_to_pointmap,
)
from models.stereo_disparity import (
    scale_intrinsics_pair_for_feature,
    disparity_to_pointmap_from_Kpair,
    disparity_to_pointmap_from_Kpair_with_logvar,
    make_gn,
    ContextUNet14,
    build_corr_pyramid,
    sample_corr_pyramid_bilinear,
    sample_corr_pyramid_bilinear_2d,
    UpdateBlock,
)
from utils.projection import SilhouetteDepthRenderer
from utils import rot_utils
from utils.logging_utils import section_timer
from utils.normalization_utils import normalize_point_map, denormalize_pos_mu, denormalize_pos_logvar
from models.updater import DeltaPoseRegressorMapHourglass4L, DeltaPoseRegressorUNet, DeltaPoseRegressor, DeltaPoseRegressor_CF
from models.raft_base_updater import DeltaPoseRegressorRAFTLike


class SSCFlow2(nn.Module):
    def __init__(self,
                 levels: int = 4,
                 norm_layer: Callable[[int], nn.Module] = make_gn(16),
                 l2_normalize_feature: bool = True,
                 use_ctx_aspp: bool = True,
                 lookup_mode: str = "1d",      # "1d" | "2d" | "a2d"
                 radius_w: int = 4,            # 横方向半径
                 radius_h: int = 0,            # 縦方向半径（2D/異方性2Dで使用）
                 num_classes: int = 1,         # ← 追加: 検出クラス数
                 use_pixelshuffle_head: bool = True,
                 hidden_ch: int = 128,      # ← 追加: マルチタスクヘッドの隠れ層次元
                 context_ch: int = 128,     # ← 追加: マルチタスクヘッドのコンテキスト次元
                 topk: int = 20,
                 nms_radius: int = 4,
                 center_thresh: float = 0.3,
                 rot_repr: str = "r6d", 
                 faces_per_pixel: int = 8,
                 blur_sigma: float = 1e-4,
                 blur_gamma: float = 1e-4,
                 delta_base_ch: int = 128,                # 追加オプション
                 shape_constraint_ema: float = 0.4,
                 use_gate: bool = True,
                 use_so3_log_xyz: bool = True,        #: 回転=so3ログ, 並進=Δx,Δy,Δz
                 use_so3_log_ratio_normal: bool = True,
                 raft_like_updater: bool = True,
                 point_map_norm_mean: Optional[Sequence[float]] = None,
                 point_map_norm_std: Optional[Sequence[float]] = None,
                 point_map_norm_eps: float = 1e-6,
                 use_pos_logz: bool = True,
                 ):
        super().__init__()
        self.levels = levels
        self.lookup_mode = lookup_mode.lower()

        # 近傍設定
        if self.lookup_mode == "1d":
            self.radius_w = radius_w
            self.radius_h = 0
        elif self.lookup_mode == "2d":
            self.radius_w = radius_w
            self.radius_h = radius_w
        elif self.lookup_mode in ("a2d", "anisotropic2d", "anisotropic_2d"):
            self.radius_w = radius_w
            self.radius_h = radius_h
        else:
            raise ValueError(f"Unknown lookup_mode={lookup_mode}")

        # 近傍数
        if self.radius_h == 0:
            k_per_level = 2 * self.radius_w + 1
        else:
            k_per_level = (2 * self.radius_w + 1) * (2 * self.radius_h + 1)

        # Encoderたち
        self.feature = ContextUNet14(norm_layer, use_aspp=use_ctx_aspp, l2_normalize=l2_normalize_feature)
        self.context = ContextUNet14(norm_layer, use_aspp=use_ctx_aspp, out_ch=context_ch + hidden_ch)
        self.update  = UpdateBlock(hidden_ch=hidden_ch, context_ch=context_ch,
                                   k_per_level=k_per_level, levels=levels, norm=norm_layer)

        # ▼ 新規: マルチタスク・ヘッドを追加（pose_headは削除）
        self.head = TransformerMultiTaskHead(
            norm_layer=norm_layer,
            num_classes=num_classes,
            ctx_ch=context_ch,
            hidden_ch=hidden_ch,
            use_pixelshuffle=use_pixelshuffle_head,
            rot_repr=rot_repr,
        )
        self.render = SilhouetteDepthRenderer(
            # faces_per_pixel=faces_per_pixel,
            # blur_sigma=blur_sigma,
            # blur_gamma=blur_gamma,
        )
        c_in = 3 + 1
        if use_so3_log_xyz:
            c_in += 3
        if use_so3_log_ratio_normal:
            c_in += 3
        if not raft_like_updater:
            c_in += 3 + 1
            c_in += 6 + context_ch + hidden_ch
            self.delta_pose_updater = DeltaPoseRegressor(
                c_in=c_in, base_ch=delta_base_ch, use_gate=use_gate
            )
        else:
            c_in += 1 # depth_conf
            self.delta_pose_updater = DeltaPoseRegressorRAFTLike(
                ctx_in_ch=c_in, base_ch=delta_base_ch
            )
        self.topk = topk
        self.nms_radius = nms_radius
        self.center_thresh = center_thresh
        self.shape_constraint_ema = shape_constraint_ema
        self.use_so3_log_xyz = use_so3_log_xyz
        self.use_so3_log_ratio_normal = use_so3_log_ratio_normal
        self.raft_like_updater = raft_like_updater
        self.point_map_norm_eps = float(point_map_norm_eps)
        self.use_pos_logz = bool(use_pos_logz)
        self.use_point_map_norm = point_map_norm_mean is not None and point_map_norm_std is not None
        if self.use_point_map_norm:
            mean = torch.as_tensor(point_map_norm_mean, dtype=torch.float32)
            std = torch.as_tensor(point_map_norm_std, dtype=torch.float32)
            if mean.numel() != 3 or std.numel() != 3:
                raise ValueError("point_map_norm_mean/std must have 3 values for (X/Z, Y/Z, logZ).")
            mean = mean.view(1, 3, 1, 1)
            std = std.view(1, 3, 1, 1).clamp_min(self.point_map_norm_eps)
            self.register_buffer("point_map_norm_mean", mean)
            self.register_buffer("point_map_norm_std", std)
        else:
            self.point_map_norm_mean = None
            self.point_map_norm_std = None

    def extract(self, stereo: torch.Tensor) -> Dict[str, torch.Tensor]:
        assert stereo.size(1) == 6
        B = stereo.size(0)
        L, R = stereo[:, :3], stereo[:, 3:]
        both = torch.cat([L, R], dim=0)
        feats = self.feature(both)             # (2B,128,H/4,W/4)
        featL, featR = feats[:B], feats[B:]
        ctx = self.context(L)                  # (B,256,H/4,W/4)
        hidden0  = torch.tanh(ctx[:, :128])    # (B,128,H/4,W/4)
        context0 = F.relu(ctx[:, 128:], inplace=False)  # (B,128,H/4,W/4)
        return {"featL_1_4": featL, "featR_1_4": featR, "ctx_1_4": ctx,
                "hidden0": hidden0, "context0": context0}

    def _normalize_point_map(self, point_map: torch.Tensor) -> torch.Tensor:
        """Normalize point map to standardized (X/Z, Y/Z, logZ) when enabled."""
        if not self.use_point_map_norm:
            return point_map
        return normalize_point_map(
            point_map,
            eps=self.point_map_norm_eps,
            mean=self.point_map_norm_mean,
            std=self.point_map_norm_std,
        )

    def forward(
        self,
        stereo: torch.Tensor,                 # (B,6,H,W)
        left_mask: torch.Tensor,              # (B,1,H,W)
        K_pair_1x: torch.Tensor,              # (B,2,3,3)
        baseline_mm: Union[float, torch.Tensor],
        meshes_flat: Meshes,
        iters: int = 8,
        disp_init: torch.Tensor = None,
        use_gt_peaks: bool = False,  
        gt_center_1_4: torch.Tensor = torch.empty(0),  # (B,1,H/4,W/4) or empty
        gt_Wk_1_4: torch.Tensor = torch.empty(0),      # (B,K,1,H/4,W/4) or empty
        gt_mask_1_4: torch.Tensor = torch.empty(0),    # (B,1,H/4,W/4) or empty
        with_shape_constraint: bool = True,
        enable_pose_update: bool = True,
    ) -> Dict[str, torch.Tensor]:

        stuff = self.extract(stereo)
        featL, featR = stuff["featL_1_4"], stuff["featR_1_4"]
        hidden, context0 = stuff["hidden0"], stuff["context0"]
        B, C, H4, W4 = featL.shape
        fp_dtype = featL.dtype

        # full corr & pyramid
        corr_full = torch.einsum('bchi,bchj->bjhi', featL, featR)
        corr_pyr  = build_corr_pyramid(corr_full, num_levels=self.levels)

        # 初期視差
        sc_disp = torch.zeros(B, 1, H4, W4, device=featL.device, dtype=featL.dtype) if disp_init is None else disp_init
        disp_preds: List[torch.Tensor] = []
        disp_log_var_preds: List[torch.Tensor] = []
        

        # 入力マスクを1/4へ（輪郭保持のため nearest）
        left_mask_1_4 = F.interpolate(left_mask.float(), size=(H4, W4), mode="nearest")

        # iter0でだけ使う head 出力（プレースホルダ）
        mask_logits = center_logits = cls_logits = None
        pos_mu = pos_logvar = None
        rot_mat = rot_logvar_theta = None
        depth_1_4 = None
        rot_maps: List[torch.Tensor] = []
        pos_maps: List[torch.Tensor] = []
        pos_logvar_maps: List[torch.Tensor] = []
        rot_logvar_theta_maps: List[torch.Tensor] = []
        Z_mm_list: List[torch.Tensor] = []  # 各iterの深度マップを保存
        Rk_list: List[torch.Tensor] = []
        tk_list: List[torch.Tensor] = []
        point_map_rend_cur = None

        for t in range(iters):
            # 近傍相関サンプリング
            if self.lookup_mode == "1d":
                corr_nb = sample_corr_pyramid_bilinear(corr_pyr, disp_1_4=sc_disp, radius=self.radius_w)
            else:
                corr_nb = sample_corr_pyramid_bilinear_2d(
                    corr_pyr, disp_1_4=disp, radius_w=self.radius_w, radius_h=self.radius_h
                )

            # update
            hidden, delta, disp_log_var = self.update(hidden, context0, corr_nb, sc_disp)
            disp = sc_disp + delta
            disp_preds.append(disp)
            disp_log_var_preds.append(disp_log_var)
            sc_disp = disp.detach()
        point_map_cur, K_pair_14, depth_log_var = disparity_to_pointmap_from_Kpair_with_logvar(
            disp, disp_log_var_preds[-1],K_pair_1x=K_pair_1x, baseline_mm=baseline_mm,
            downsample=4, eps=1e-6
        )  # (B,3,H/4,W/4)
        depth_1_4 = point_map_cur[:, 2:3]
        point_map_cur_in = self._normalize_point_map(point_map_cur)
    
        head_out = self.head(
            ctx_14=context0,          # 勾配はここへ
            hidden_14=hidden,
            mask_14=left_mask_1_4,    # 入力インスタンスマスク（そのまま）
            point_map_14=point_map_cur_in
        )
        pos_mu_raw = head_out["pos_mu"]
        pos_logvar_raw = head_out["pos_logvar"]
        if self.use_pos_logz:
            pos_mu = denormalize_pos_mu(pos_mu_raw)
            pos_logvar = denormalize_pos_logvar(pos_logvar_raw, pos_mu_raw)
        else:
            pos_mu = pos_mu_raw
            pos_logvar = pos_logvar_raw
        with torch.no_grad():
            inst = extract_instances_from_head_vec(
                head_out["center_logits"],
                head_out["mask_logits"],
                pos_mu,
                head_out["rot_mat"],
                head_out["cls_logits"],
                K_pair_1x[:, 0],
                use_gt_peaks,
                gt_center_1_4,   # Optional[Tensor] (確率/ヒート)
                gt_Wk_1_4,       # Optional[Tensor]
                self.topk,
                self.nms_radius,
                self.center_thresh,
                True,                            # use_uncertainty
                pos_logvar,
                head_out.get("rot_logvar_theta", None),
                2.0, 4,                          # gauss_sigma, gauss_radius
                True                             # class_from_logits
            )
        Rk_list.append(inst['R'])
        tk_list.append(inst['t'])
        # 単発ヘッド出力に格納
        mask_logits        = head_out["mask_logits"]
        center_logits      = head_out["center_logits"]
        rot_mat            = head_out["rot_mat"]
        rot_logvar_theta   = head_out["rot_logvar_theta"]
        cls_logits         = head_out["cls_logits"]
        assert pos_mu.shape[1] == 3
        assert pos_logvar.shape[1] == 3
        current_pos_map = pos_mu_to_pointmap(pos_mu, K_pair_1x[:, 0], downsample=4)
        current_rot_map = rot_mat
        flow_pred_list = []
        if enable_pose_update:
            with torch.no_grad():
                render_out = _render_t0_outputs(
                    renderer=self.render,
                    meshes_flat=meshes_flat,
                    K_left_14=K_pair_14[:, 0],        # (B,3,3)
                    instances=inst,
                    valid_k=inst["valid"],                   # (B,K) ????OK??atch ?? valid ?????????valid_k ???????????????
                    image_size=featL.shape[-2:],
                )
            depth_rend = render_out["depth_pred"]
            point_map_rend_cur = rot_utils.depth_to_pointmap_from_K(depth_rend, K_pair_14[:, 0])
            cur_pos_logvar_map = pos_logvar
            cur_rot_logvar_theta_map = rot_logvar_theta
            for i in range(2):
                # --- ????????iter?? ??ose ??????????? ---
                mask_14 = mask_logits.detach()
                extra   = stuff["hidden0"].detach()
                depth_conf =  torch.exp(-0.5 * depth_log_var).detach()
                point_map_rend_in = self._normalize_point_map(point_map_rend_cur)
                ctx_in = [
                    point_map_cur_in.detach(),
                    mask_14,
                    depth_conf,
                ]
                if self.use_so3_log_xyz:
                    R_log = rot_utils.so3_log_map(current_rot_map.detach())            # (B,3,H,W)
                    ctx_in.append(R_log)
                else:
                    R_log = None

                add_feats = [
                    cur_pos_logvar_map.detach(),
                    cur_rot_logvar_theta_map.detach(),
                ]
                if self.use_so3_log_ratio_normal:
                    Zc = point_map_cur[:, 2:3].clamp_min(1e-6)
                    Zr = point_map_rend_cur[:, 2:3].clamp_min(1e-6)
                    dz_ratio = torch.log(Zc / Zr)                    # (B,1,H,W)

                    xz_c = point_map_cur[:, 0:1] / Zc
                    yz_c = point_map_cur[:, 1:2] / Zc
                    xz_r = point_map_rend_cur[:, 0:1] / Zr
                    yz_r = point_map_rend_cur[:, 1:2] / Zr
                    dn_x = xz_c - xz_r
                    dn_y = yz_c - yz_r
                    add_feats.extend([dz_ratio.detach(), dn_x.detach(), dn_y.detach()])
                    ctx_in.extend([dz_ratio.detach(), dn_x.detach(), dn_y.detach()])
                if self.raft_like_updater:
                    delta_pose_map, flow_pred, pos_logvar_map, rot_logvar_theta_map = self.delta_pose_updater(
                        point_map_cur_in.detach(), point_map_rend_in.detach(),
                        current_pos_map.detach(), R_log.detach(),
                        cur_pos_logvar_map.detach(), cur_rot_logvar_theta_map.detach(),
                        torch.cat(ctx_in, dim=1),
                    )
                    flow_pred_list.append(flow_pred)
                else:
                    delta_pose_map, pos_logvar_map, rot_logvar_theta_map = self.delta_pose_updater(
                        point_map_cur_in.detach(), point_map_rend_in.detach(), current_pos_map.detach(), R_cur_log=R_log,
                        context=context0.detach(), mask=mask_14, extra_feats=extra,
                        add_feats=add_feats
                    )
                # # SE(3)?????? ??????????????????
                R_delta, t_delta = rot_utils.se3_from_delta_map(delta_pose_map)     # (B,3,3,H/4,W/4), (B,3,H/4,W/4)
                # # ????????/?????????????
                current_rot_map, current_pos_map = rot_utils.update_pose_maps_se3(
                    current_rot_map.detach(),              # (B,3,3,H/4,W/4)
                    current_pos_map.detach(),              # (B,3,H/4,W/4)
                    R_delta,                               # (B,3,3,H/4,W/4)
                    t_delta,                               # (B,3,H/4,W/4)
                )
                rot_maps.append(current_rot_map)
                pos_maps.append(current_pos_map)
                pos_logvar_maps.append(pos_logvar_map)
                rot_logvar_theta_maps.append(rot_logvar_theta_map)
                cur_pos_logvar_map = pos_logvar_map
                cur_rot_logvar_theta_map = rot_logvar_theta_map

            
                        
                # ==== ここから各 iter で「レンダ側 point map」を“インスタンス単位”に更新 ====
                if use_gt_peaks:
                    Wk = gt_Wk_1_4
                    # fg_w = gt_mask_1_4
                    fg_w = torch.ones_like(gt_mask_1_4)
                else:
                    # 1) インスタンス重み（Wk）と前景重みを準備
                    Wk = inst["weight_map"]        # (B,K,1,H/4,W/4)  ※extract_instancesの戻りを保持している想定
                    fg_w = torch.sigmoid(mask_logits).clamp(0, 1) # (B,1,H/4,W/4)    ※他でも可。学習安定優先なら detach

                # 2) 現在の rot/pos マップから (R_k, t_k) を抽出
                Rk, tk, is_valid, _, _ = rot_utils.pose_from_maps_auto(
                    rot_map=current_rot_map,       # (B,3,3,H/4,W/4)
                    pos_map=current_pos_map,       # (B,3,H/4,W/4)
                    Wk_1_4=Wk,                     # (B,K,1,H/4,W/4)
                    wfg=fg_w > 0.5,
                    min_px=10,
                    min_wsum=1e-6,
                )
            
                Rk_list.append(Rk)
                tk_list.append(tk)
            
                # 3) レンダリングベースで再描画（各 iter）
                    # (a) インスタンスごとの SE(3) を T に組み立てる
                    #     compose_T_from_Rt(R, t, valid) の挙動に合わせて valid をそのまま渡す
                T_upd = rot_utils.compose_T_from_Rt(Rk, tk, is_valid)   # (B,K,4,4)
                with torch.no_grad():
                    # (b) 更新ポーズで再レンダ
                    render_out_iter = self.render(
                        meshes_flat=meshes_flat,             # flatten済み Meshes（B×K_valid の並び想定）
                        T_cam_obj=T_upd,                     # (B,K,4,4)
                        K_left=K_pair_14[:, 0],              # (B,3,3)   ※1/4 K
                        valid_k=is_valid,                    # (B,K)     ※無効は内部でスキップ
                        image_size=featL.shape[-2:],         # (H/4, W/4) と同じ空間解像度でOK
                    )

                # (c) レンダ深度 → point-map（1/4解像度）
                depth_rend_iter = render_out_iter["depth"]              # (B,1,H/4,W/4)
                point_map_rend_cur = rot_utils.depth_to_pointmap_from_K(
                    depth_rend_iter, K_pair_14[:, 0]
                )  # (B,3,H/4,W/4)
                Z_mm = point_map_rend_cur[:, 2:3]         # (B,1,H/4,W/4)
                Z_mm_list.append(Z_mm.clone())
        else:
            point_map_rend_cur = point_map_cur.detach()

        

        # 返却
        dst: Dict[str, torch.Tensor] = {
            **stuff,
            "corr_pyr": corr_pyr,
            "disp_preds": disp_preds,  # (B,1,H/4,W/4)
            "disp_log_var_preds": disp_log_var_preds,  # (B,1,H/4,W/4)
            # iter0 のヘッド出力（単発）
            "mask_logits":  mask_logits,        # (B,1,H/4,W/4)
            "center_logits":center_logits,      # (B,1,H/4,W/4)
            "pos_mu":       pos_mu,             # (B,3,H/4,W/4)
            "pos_logvar":   pos_logvar,         # (B,3,H/4,W/4)
            "rot_mat":      rot_mat,            # (B,3,3,H/4,W/4)
            "rot_maps": rot_maps,
            "pos_maps": pos_maps,
            "rot_logvar_theta": rot_logvar_theta, # (B,1,H/4,W/4)
            "cls_logits":   cls_logits,         # (B,C,H/4,W/4)
            "depth_1_4":    depth_1_4,          # 便宜上
            "left_mask_1_4": left_mask_1_4,     # lossのbootstrap用
            "point_map_rend_last": point_map_rend_cur,    # 最終反復のレンダ側ポイントマップ
            "instances": inst,
            "Z_mm_list": Z_mm_list,
            "Rk_list": Rk_list,
            "tk_list": tk_list,
            "pos_logvar_maps": pos_logvar_maps,
            "rot_logvar_theta_maps": rot_logvar_theta_maps,
        }
        if self.raft_like_updater and len(flow_pred_list) > 0:
            flow_preds = torch.cat(flow_pred_list, dim=1)
            dst['flow_preds'] = flow_preds
            dst['flow_pred'] = flow_preds[:, -1]
        return dst

    # def forward(
    #     self,
    #     stereo: torch.Tensor,                 # (B,6,H,W)
    #     left_mask: torch.Tensor,              # (B,1,H,W)
    #     K_pair_1x: torch.Tensor,              # (B,2,3,3)
    #     baseline_mm: Union[float, torch.Tensor],
    #     meshes_flat: Meshes,
    #     iters: int = 8,
    #     disp_init: torch.Tensor = None,
    #     use_gt_peaks: bool = False,  
    #     gt_center_1_4: torch.Tensor = torch.empty(0),  # (B,1,H/4,W/4) or empty
    #     gt_Wk_1_4: torch.Tensor = torch.empty(0),      # (B,K,1,H/4,W/4) or empty
    #     gt_mask_1_4: torch.Tensor = torch.empty(0),    # (B,1,H/4,W/4) or empty
    #     with_shape_constraint: bool = True,
    # ) -> Dict[str, torch.Tensor]:

    #     stuff = self.extract(stereo)
    #     featL, featR = stuff["featL_1_4"], stuff["featR_1_4"]
    #     hidden, context0 = stuff["hidden0"], stuff["context0"]
    #     B, C, H4, W4 = featL.shape
    #     fp_dtype = featL.dtype

    #     # full corr & pyramid
    #     corr_full = torch.einsum('bchi,bchj->bjhi', featL, featR)
    #     corr_pyr  = build_corr_pyramid(corr_full, num_levels=self.levels)

    #     # 初期視差
    #     sc_disp = torch.zeros(B, 1, H4, W4, device=featL.device, dtype=featL.dtype) if disp_init is None else disp_init
    #     disp_preds: List[torch.Tensor] = []
        

    #     # 入力マスクを1/4へ（輪郭保持のため nearest）
    #     left_mask_1_4 = F.interpolate(left_mask.float(), size=(H4, W4), mode="nearest")

    #     # iter0でだけ使う head 出力（プレースホルダ）
    #     mask_logits = center_logits = cls_logits = None
    #     pos_mu = pos_logvar = None
    #     rot_mat = rot_logvar_theta = None
    #     depth_1_4 = None
    #     rot_maps: List[torch.Tensor] = []
    #     pos_maps: List[torch.Tensor] = []
    #     Z_mm_list: List[torch.Tensor] = []  # 各iterの深度マップを保存
    #     point_map_rend_cur = None

    #     for t in range(iters):
    #         # 近傍相関サンプリング
    #         if self.lookup_mode == "1d":
    #             corr_nb = sample_corr_pyramid_bilinear(corr_pyr, disp_1_4=sc_disp, radius=self.radius_w)
    #         else:
    #             corr_nb = sample_corr_pyramid_bilinear_2d(
    #                 corr_pyr, disp_1_4=disp, radius_w=self.radius_w, radius_h=self.radius_h
    #             )

    #         # update
    #         hidden, delta = self.update(hidden, context0, corr_nb, sc_disp)
    #         disp = sc_disp + delta
    #         disp_preds.append(disp)
    #         point_map_cur, K_pair_14 = disparity_to_pointmap_from_Kpair(
    #             disp, K_pair_1x=K_pair_1x, baseline_mm=baseline_mm,
    #             downsample=4, eps=1e-6
    #         )  # (B,3,H/4,W/4)
    #         depth_1_4 = point_map_cur[:, 2:3]

    #         # ★ iter0 だけ MultiHead 実行（update 後の disp を使用）
    #         if t == 0:
    #             head_out = self.head(
    #                 ctx_14=context0,          # 勾配はここへ
    #                 hidden_14=hidden,
    #                 mask_14=left_mask_1_4,    # 入力インスタンスマスク（そのまま）
    #                 point_map_14=point_map_cur
    #             )
    #             with torch.no_grad():
    #                 inst = extract_instances_from_head_vec(
    #                     head_out["center_logits"],
    #                     head_out["mask_logits"],
    #                     head_out["pos_mu"],
    #                     head_out["rot_mat"],
    #                     head_out["cls_logits"],
    #                     K_pair_1x[:, 0],
    #                     use_gt_peaks,
    #                     gt_center_1_4,   # Optional[Tensor] (確率/ヒート)
    #                     gt_Wk_1_4,       # Optional[Tensor]
    #                     self.topk,
    #                     self.nms_radius,
    #                     self.center_thresh,
    #                     True,                            # use_uncertainty
    #                     head_out.get("pos_logvar", None),
    #                     head_out.get("rot_logvar_theta", None),
    #                     2.0, 4,                          # gauss_sigma, gauss_radius
    #                     True                             # class_from_logits
    #                 )
    #             # 単発ヘッド出力に格納
    #             mask_logits        = head_out["mask_logits"]
    #             center_logits      = head_out["center_logits"]
    #             pos_mu             = head_out["pos_mu"]
    #             pos_logvar         = head_out["pos_logvar"]
    #             rot_mat            = head_out["rot_mat"]
    #             rot_logvar_theta   = head_out["rot_logvar_theta"]
    #             cls_logits         = head_out["cls_logits"]
    #             current_pos_map = rot_utils.depth_to_pointmap_from_K(head_out["pos_mu"], K_pair_14[:, 0])
    #             current_rot_map = rot_mat
                
    #             render_out = _render_t0_outputs(
    #                 renderer=self.render,
    #                 meshes_flat=meshes_flat,
    #                 K_left_14=K_pair_14[:, 0],        # (B,3,3)
    #                 instances=inst,
    #                 valid_k=inst["valid"],                   # (B,K) でも OK（batch 側 valid と同じなら valid_k をそのまま渡す）
    #                 image_size=featL.shape[-2:],
    #             )
    #             depth_rend = render_out["depth_pred"]
    #             point_map_rend_cur = rot_utils.depth_to_pointmap_from_K(depth_rend, K_pair_14[:, 0])
            
    #         # --- ここから全iterで Δpose を推定して合成 ---
    #         mask_14 = mask_logits
    #         extra   = stuff["hidden0"]
    #         if self.use_so3_log_xyz:
    #             R_log = rot_utils.so3_log_map(current_rot_map)            # (B,3,H,W)
    #         else:
    #             R_log = None
            
    #         if self.use_so3_log_ratio_normal:
    #             Zc = point_map_cur[:, 2:3].clamp_min(1e-6)
    #             Zr = point_map_rend_cur[:, 2:3].clamp_min(1e-6)
    #             dz_ratio = torch.log(Zc / Zr)                    # (B,1,H,W)

    #             xz_c = point_map_cur[:, 0:1] / Zc
    #             yz_c = point_map_cur[:, 1:2] / Zc
    #             xz_r = point_map_rend_cur[:, 0:1] / Zr
    #             yz_r = point_map_rend_cur[:, 1:2] / Zr
    #             dn_x = xz_c - xz_r
    #             dn_y = yz_c - yz_r
    #             add_feats = [dz_ratio, dn_x, dn_y]   # 必要に応じて posenc も
    #         else:
    #             add_feats = None
    #         delta_pose_map = self.delta_pose_updater(
    #             point_map_cur, point_map_rend_cur, R_cur_log=R_log,
    #             context=context0, mask=mask_14, extra_feats=extra,
    #             add_feats=add_feats
    #         )
    #         # # SE(3)指数写像 → 合成（列ベクトル系）
    #         R_delta, t_delta = rot_utils.se3_from_delta_map(delta_pose_map)     # (B,3,3,H/4,W/4), (B,3,H/4,W/4)
    #         # # レンダ側/現在地図を更新
    #         current_rot_map, current_pos_map = rot_utils.update_pose_maps(
    #             current_rot_map,              # (B,3,3,H/4,W/4)
    #             current_pos_map,              # (B,3,H/4,W/4)   ← これは「現在の並進マップ t=(tx,ty,tz)」を想定
    #             R_delta,                      # (B,3,3,H/4,W/4)
    #             d_map=t_delta,                # (B,3,H/4,W/4)   ← ここでは t_delta を (dx,dy,dz) として解釈
    #             weight=100.0,
    #             depth_transform="exp",
    #             detach_depth_for_xy=True,     # 不安定なら True 推奨
    #             eps=1e-6
    #         )
    #         rot_maps.append(current_rot_map)
    #         pos_maps.append(current_pos_map)
            
                        
    #         # ==== ここから各 iter で「レンダ側 point map」を“インスタンス単位”に更新 ====
    #         if use_gt_peaks:
    #             Wk = gt_Wk_1_4
    #             # fg_w = gt_mask_1_4
    #             fg_w = torch.ones_like(gt_mask_1_4)
    #         else:
    #             # 1) インスタンス重み（Wk）と前景重みを準備
    #             Wk = inst["weight_map"]        # (B,K,1,H/4,W/4)  ※extract_instancesの戻りを保持している想定
    #             fg_w = torch.sigmoid(mask_logits).clamp(0, 1) # (B,1,H/4,W/4)    ※他でも可。学習安定優先なら detach

    #         # 2) 現在の rot/pos マップから (R_k, t_k) を抽出
    #         Rk, tk, is_valid, _, _ = rot_utils.pose_from_maps_auto(
    #             rot_map=current_rot_map,       # (B,3,3,H/4,W/4)
    #             pos_map=current_pos_map,       # (B,3,H/4,W/4)
    #             Wk_1_4=Wk,                     # (B,K,1,H/4,W/4)
    #             wfg=fg_w > 0.5,
    #             min_px=10,
    #             min_wsum=1e-6,
    #         )
            
    #         # 3) レンダリングベースで再描画（各 iter）
    #             # (a) インスタンスごとの SE(3) を T に組み立てる
    #             #     compose_T_from_Rt(R, t, valid) の挙動に合わせて valid をそのまま渡す
    #         T_upd = rot_utils.compose_T_from_Rt(Rk, tk, is_valid)   # (B,K,4,4)

    #         # (b) 更新ポーズで再レンダ
    #         render_out_iter = self.render(
    #             meshes_flat=meshes_flat,             # flatten済み Meshes（B×K_valid の並び想定）
    #             T_cam_obj=T_upd,                     # (B,K,4,4)
    #             K_left=K_pair_14[:, 0],              # (B,3,3)   ※1/4 K
    #             valid_k=is_valid,                    # (B,K)     ※無効は内部でスキップ
    #             image_size=featL.shape[-2:],         # (H/4, W/4) と同じ空間解像度でOK
    #         )

    #         # (c) レンダ深度 → point-map（1/4解像度）
    #         depth_rend_iter = render_out_iter["depth"]              # (B,1,H/4,W/4)
    #         point_map_rend_cur = rot_utils.depth_to_pointmap_from_K(
    #             depth_rend_iter, K_pair_14[:, 0]
    #         )  # (B,3,H/4,W/4)

    #         # 3) inst_id_map がある場合はハード割当てで射影ワープ＋スプラット
    #         # if "inst_id_map" in render_out.keys():
    #         #     point_map_rend_cur = _splat_per_instance_projective(
    #         #         point_map_rend=point_map_rend_cur,   # (B,3,H/4,W/4) 前反復のレンダ側地図
    #         #         inst_id_map=gt_Wk_1_4,             # (B,1,H/4,W/4)
    #         #         Rk=Rk, tk=tk,                        # (B,K,3,3), (B,K,3)
    #         #         K14=K_pair_14[:, 0],                 # (B,3,3)
    #         #         tau=0.02, eps=1e-6,
    #         #     )
    #         # else:
    #         #     # inst_id_map が無い場合のフォールバック（ソフト割当て版）
    #         #     # soft responsibility: resp = softmax(log(Wk) - Z/τ) など
    #         #     resp = (Wk[:, :, 0] * fg_w[:, 0]).clamp_min(0)                         # (B,K,H,W)
    #         #     resp = resp / resp.sum(dim=1, keepdim=True).clamp_min(1e-6)            # Σk=1
    #         #     # 画素ごとに X→RkX+t を合成（簡易）
    #         #     X = point_map_rend_cur.unsqueeze(1).expand(-1, Rk.size(1), -1, -1, -1) # (B,K,3,H,W)
    #         #     Xp = torch.einsum('bkij,bkjhw->bkihw', Rk, X) + tk.unsqueeze(-1).unsqueeze(-1)  # (B,K,3,H,W)
    #         #     point_map_rend_cur = (resp.unsqueeze(2) * Xp).sum(dim=1)               # (B,3,H,W)
            
    #         # ---- shape-constraint disparity をつくって、次 iter の初期値に使う ----
    #         # 1) レンダリング側 point-map から disp_14 を再計算
    #         disp_rend_14 = _disparity_from_pointmap_14(
    #             point_map_14=point_map_rend_cur,          # (B,3,H/4,W/4)
    #             K_pair_14=K_pair_14,                # (B,3,3) 1/4 K
    #             baseline_mm=baseline_mm,
    #             eps=1e-6,
    #         )  # (B,1,H/4,W/4)

    #         # 2) mask（前景）で内側をレンダdispに、外側を既存dispに
    #         #    - デフォルトはソフトブレンド（ハードにしたければ >thr の2値化でもOK）
    #         fg_w = torch.sigmoid(mask_logits).clamp(0, 1)  # (B,1,H/4,W/4)

    #         # （オプション）EMA で過度な振れを抑える
            
    #         Z_mm = point_map_rend_cur[:, 2:3]         # (B,1,H/4,W/4)
    #         valid_rend = (Z_mm > 1.0)                 # 1mm より浅い/0は無効
    #         Z_mm_list.append(Z_mm.clone())

    #         # 2) disp_rend_14 の非有限値は視差に置換（数値安定）
    #         disp_rend_14 = torch.where(
    #             valid_rend,
    #             disp_rend_14,
    #             disp
    #         )

    #         # 3) 物体外/レンダ無効ピクセルは必ずdispを採用
    #         #    物体内かつレンダ有効のところだけレンダ視差を使う
    #         use_rend = valid_rend & (fg_w > 0.5)      # 閾値はお好みで。連続重みで使うなら後述の連続版へ
    #         disp_mix = torch.where(use_rend, disp_rend_14, disp)

    #         # 4) EMAで過度な揺れを抑制
    #         disp_sc = self.shape_constraint_ema * disp + (1.0 - self.shape_constraint_ema) * disp_mix

    #         # 3) 次 iter のスタート視差として採用（安定化のため detach）
    #         if with_shape_constraint:
    #             sc_disp = disp_sc.to(dtype=fp_dtype)
    #         else:
    #             sc_disp = disp

    #     # 返却
    #     dst: Dict[str, torch.Tensor] = {
    #         **stuff,
    #         "corr_pyr": corr_pyr,
    #         "disp_preds": disp_preds,  # (B,1,H/4,W/4)
    #         # iter0 のヘッド出力（単発）
    #         "mask_logits":  mask_logits,        # (B,1,H/4,W/4)
    #         "center_logits":center_logits,      # (B,1,H/4,W/4)
    #         "pos_mu":       pos_mu,             # (B,3,H/4,W/4)
    #         "pos_logvar":   pos_logvar,         # (B,3,H/4,W/4)
    #         "rot_mat":      rot_mat,            # (B,3,3,H/4,W/4)
    #         "rot_maps": rot_maps,
    #         "pos_maps": pos_maps,
    #         "rot_logvar_theta": rot_logvar_theta, # (B,1,H/4,W/4)
    #         "cls_logits":   cls_logits,         # (B,C,H/4,W/4)
    #         "depth_1_4":    depth_1_4,          # 便宜上
    #         "left_mask_1_4": left_mask_1_4,     # lossのbootstrap用
    #         "point_map_rend_last": point_map_rend_cur,    # 最終反復のレンダ側ポイントマップ
    #         "instances": inst,
    #         "Z_mm_list": Z_mm_list,
    #     }
    #     return dst
    

def _render_t0_outputs(
    renderer,                       # SilhouetteDepthRenderer
    meshes_flat,                    # Meshes(len = sum(valid_k))
    K_left_14: torch.Tensor,        # (B,3,3) 左K [px]
    instances: Dict[str, torch.Tensor],   # extract_instances_from_head_vec の戻り
    valid_k: torch.Tensor,          # (B,K) bool/0-1
    image_size: Tuple[int,int],     # (H,W)
) -> Dict[str, torch.Tensor]:
    """
    t=0 用：予測ポーズのみをレンダリングして (B,1,H,W) の silhouette / depth を返す。
    ※ meshes_flat の並び順は valid_k=True の順でフラットになっている前提。
    """
    if renderer is None or meshes_flat is None:
        return {}

    # 予測ポーズ（列ベクトル系 4x4）
    # R:(B,K,3,3), t:(B,K,3), valid:(B,K)
    T_pred = rot_utils.compose_T_from_Rt(instances["R"], instances["t"], instances["valid"])  # (B,K,4,4)

    # レンダリング（PyTorch3D は基本 f32 安定）
    pred_r = renderer(
        meshes_flat=meshes_flat,
        T_cam_obj=T_pred.to(torch.float32),
        K_left=K_left_14.to(torch.float32),
        valid_k=valid_k,
        image_size=image_size,
    )

    out: Dict[str, torch.Tensor] = {
        "silhouette_pred": pred_r["silhouette"],  # (B,1,H,W)
        "depth_pred":      pred_r["depth"],       # (B,1,H,W)
    }
    # 拡張レンダラ対応（持ってるときだけ返す）
    for k in ("inst_alpha", "inst_visible_masks", "inst_id_map"):
        if k in pred_r: out[f"{k}_pred"] = pred_r[k]
    return out


# ===================== utils: 射影ワープ + 双一次スプラット（soft z 合成） =====================
def _splat_per_instance_projective(
    point_map_rend: torch.Tensor,   # (B,3,H,W)  現在の "レンダ側" point map(X,Y,Z)
    inst_id_map: torch.Tensor,      # (B,1,H,W)  {-1,0..K-1}
    Rk: torch.Tensor,               # (B,K,3,3)
    tk: torch.Tensor,               # (B,K,3)
    K14: torch.Tensor,              # (B,3,3)  1/4解像度のK
    tau: float = 0.02,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    各インスタンスに属する画素だけを取り出して SE(3) で動かし、画像平面に投影。
    双一次スプラット + soft z 重みで合成。
    戻り値: (B,3,H,W)
    """
    B, _, H, W = point_map_rend.shape
    device = point_map_rend.device
    dtype  = point_map_rend.dtype

    num = torch.zeros(B, 3, H, W, device=device, dtype=dtype)
    den = torch.zeros(B, 1, H, W, device=device, dtype=dtype)

    K = Rk.size(1)
    for k in range(K):
        Mk = (inst_id_map[:, 0] == k)  # (B,H,W) bool
        if not Mk.any():
            continue

        b_idx, y_idx, x_idx = torch.where(Mk)  # (N,)
        X = point_map_rend[b_idx, :, y_idx, x_idx]    # (N,3)

        R = Rk[b_idx, k]                               # (N,3,3)
        t = tk[b_idx, k]                               # (N,3)
        Xp = (R @ X.unsqueeze(-1)).squeeze(-1) + t     # (N,3)

        z = Xp[:, 2]
        valid_z = (z > 1e-6)
        if not valid_z.any():
            continue
        b_idx = b_idx[valid_z]; y_idx = y_idx[valid_z]; x_idx = x_idx[valid_z]
        Xp = Xp[valid_z]; z = z[valid_z]

        fx = K14[b_idx, 0, 0]; fy = K14[b_idx, 1, 1]
        cx = K14[b_idx, 0, 2]; cy = K14[b_idx, 1, 2]
        uf = fx * (Xp[:, 0] / z) + cx
        vf = fy * (Xp[:, 1] / z) + cy

        in_img = (uf >= 0) & (uf <= (W-1)) & (vf >= 0) & (vf <= (H-1))
        if not in_img.any():
            continue

        b_idx = b_idx[in_img]; uf = uf[in_img]; vf = vf[in_img]
        Xp = Xp[in_img]; z = z[in_img]

        x0 = torch.floor(uf).to(torch.long); x1 = (x0 + 1).clamp_max(W-1)
        y0 = torch.floor(vf).to(torch.long); y1 = (y0 + 1).clamp_max(H-1)
        wx = (uf - x0.float()); wy = (vf - y0.float())
        nbrs = [
            (y0, x0, (1 - wx) * (1 - wy)),
            (y0, x1, (wx)     * (1 - wy)),
            (y1, x0, (1 - wx) * (wy)),
            (y1, x1, (wx)     * (wy)),
        ]

        logwz = -(z / tau)  # soft z（Zバッファ近似）

        # フラット化して index_add でスプラット
        num_flat = num.view(B, 3, H * W)
        den_flat = den.view(B, 1, H * W)

        for yy, xx, wb in nbrs:
            w = (wb * torch.exp(logwz)).unsqueeze(1)          # (N,1)
            lin = (b_idx * H + yy) * W + xx                    # (N,)
            num_flat[b_idx, :, lin] += (w * Xp).transpose(0, 1)
            den_flat[b_idx, :, lin] += w.transpose(0, 1)

    out = num / den.clamp_min(eps)
    # den==0 の穴は入力をそのまま残すなどの埋め戻し推奨
    mask_hole = (den <= eps)
    if mask_hole.any():
        out = torch.where(mask_hole.expand_as(out), point_map_rend, out)
    return out


def _disparity_from_pointmap_14(
    point_map_14: torch.Tensor,      # (B,3,H/4,W/4), Z は [mm]
    K_pair_14: torch.Tensor,                 # (B,2,3,3), full-res intrinsics [Left, Right]
    baseline_mm: Union[float, torch.Tensor],
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    1/4解像度の point_map(mm) から 1/4 disparity を計算。
    disp_14 = fx_14 * baseline_mm / Z_mm
    ただし Z<=0.1mm の画素は視差0にする。
    """
    B, _, H14, W14 = point_map_14.shape
    device = point_map_14.device
    dtype  = point_map_14.dtype

    Z_raw = point_map_14[:, 2:3]                # (B,1,H/4,W/4) [mm]
    valid_z = (Z_raw > 0.1)                     # 0.1 mm より浅い所は無効
    Z = Z_raw.clamp_min(eps)                    # 0割防止

    fx_14 = K_pair_14[:, 0, 0, 0].view(-1, 1, 1, 1).to(dtype)  # (B,1,1,1), pixels
    disparity_shift = (K_pair_14[:, 1, 0, 2] - K_pair_14[:, 0, 0, 2]).view(B,1,1,1).to(dtype)  # cx_right - cx_left

    if torch.is_tensor(baseline_mm):
        Bmm = baseline_mm.to(device=device, dtype=dtype)
        if Bmm.ndim == 0:
            Bmm = Bmm.view(1, 1, 1, 1)
        elif Bmm.ndim == 1:                     # (B,) -> (B,1,1,1)
            Bmm = Bmm.view(-1, 1, 1, 1)
    else:
        Bmm = torch.tensor(baseline_mm, device=device, dtype=dtype).view(1, 1, 1, 1)
    
    disp_1_4 = fx_14 * Bmm / Z - disparity_shift

    return disp_1_4 * valid_z.to(dtype)       # Z<=1mm を 0 に
