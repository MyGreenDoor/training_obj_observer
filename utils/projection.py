# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
from typing import Tuple
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    MeshRasterizer, RasterizationSettings,
    SoftSilhouetteShader, BlendParams,
    PerspectiveCameras
)

def cameras_from_opencv_projection(
    R: torch.Tensor,
    tvec: torch.Tensor,
    focal_length: torch.Tensor,
    principal_point: torch.Tensor,
    image_size: torch.Tensor,
    device,
) -> PerspectiveCameras:

    # Retype the image_size correctly and flip to width, height.
    image_size_wh = image_size.to(R).flip(dims=(1,))

    # Screen to NDC conversion:
    # For non square images, we scale the points such that smallest side
    # has range [-1, 1] and the largest side has range [-u, u], with u > 1.
    # This convention is consistent with the PyTorch3D renderer, as well as
    # the transformation function `get_ndc_to_screen_transform`.
    scale = (image_size_wh.to(R).min(dim=1, keepdim=True)[0] - 1) / 2.0
    scale = scale.expand(-1, 2)
    c0 = (image_size_wh - 1) / 2.0

    # Get the PyTorch3D focal length and principal point.
    focal_pytorch3d = focal_length / scale
    p0_pytorch3d = -(principal_point - c0) / scale

    # For R, T we flip x, y axes (opencv screen space has an opposite
    # orientation of screen axes).
    # We also transpose R (opencv multiplies points from the opposite=left side).
    # The original code clone the R and T, which will cause R and T requires_grad = False
    # TODO figure out why clone the R and T
    R_pytorch3d = R.clone().permute(0, 2, 1)
    T_pytorch3d = tvec.clone()
    R_pytorch3d[:, :, :2] = R_pytorch3d[:, :, :2] * -1
    T_pytorch3d[:, :2] *= -1

    return PerspectiveCameras(
        R=R_pytorch3d,
        T=T_pytorch3d,
        focal_length=focal_pytorch3d,
        principal_point=p0_pytorch3d,
        image_size=image_size,
        device=device
    )
    
    
# class SilhouetteDepthRenderer(torch.nn.Module):
#     def __init__(self, faces_per_pixel: int = 8, blur_sigma: float = 1e-4, blur_gamma: float = 1e-4):
#         super().__init__()
#         self.faces_per_pixel = faces_per_pixel
#         self.blend_params = BlendParams(sigma=blur_sigma, gamma=blur_gamma)

#     @torch.no_grad()
#     def _K_to_fx_fy_cx_cy(self, K_left_b: torch.Tensor):
#         fx = K_left_b[0, 0]; fy = K_left_b[1, 1]
#         cx = K_left_b[0, 2]; cy = K_left_b[1, 2]
#         return fx, fy, cx, cy

#     def forward(
#         self,
#         meshes_flat: Meshes,          # len = sum(valid_k)
#         T_cam_obj: torch.Tensor,      # (B,K,4,4)  列ベクトル系
#         K_left: torch.Tensor,         # (B,3,3)
#         valid_k: torch.Tensor,        # (B,K)  bool/0-1
#         image_size: Tuple[int, int],  # (H,W)
#     ):
#         device = K_left.device
#         dtype  = K_left.dtype
#         B, K = valid_k.shape
#         H, W = image_size

#         sil_out = []
#         dep_out = []
#         # 追加: インスタンスごとの出力（グローバルKに合わせた形）
#         inst_alpha_all   = torch.zeros(B, K, H, W, device=device, dtype=torch.float32)
#         inst_vis_all     = torch.zeros(B, K, H, W, device=device, dtype=torch.float32)
#         inst_id_map_all  = torch.full((B, 1, H, W), -1, device=device, dtype=torch.int32)

#         cursor = 0
#         K_left_f32    = K_left.to(torch.float32)
#         T_cam_obj_f32 = T_cam_obj.to(torch.float32)

#         for b in range(B):
#             vmask_b = (valid_k[b] > 0)                 # (K,)
#             Kb = int(vmask_b.sum().item())
#             if Kb == 0:
#                 sil_out.append(torch.zeros(1, 1, H, W, device=device, dtype=torch.float32))
#                 dep_out.append(torch.zeros(1, 1, H, W, device=device, dtype=torch.float32))
#                 continue

#             # ローカルindex(kb)→グローバルindex(k)
#             global_idx = torch.nonzero(vmask_b, as_tuple=False).squeeze(1)  # (Kb,)

#             T_b   = T_cam_obj_f32[b, vmask_b]       # (Kb,4,4)
#             R_col = T_b[:, :3, :3].contiguous()
#             t_col = T_b[:, :3,  3].contiguous()

#             fx, fy, cx, cy = self._K_to_fx_fy_cx_cy(K_left_f32[b])
#             focal_length    = torch.stack([fx.expand(Kb), fy.expand(Kb)], dim=1)  # (Kb,2)
#             principal_point = torch.stack([cx.expand(Kb), cy.expand(Kb)], dim=1)  # (Kb,2)
#             imgsz = torch.tensor([[H, W]], dtype=torch.float32, device=device).expand(Kb, -1).contiguous()

#             cameras = cameras_from_opencv_projection(
#                 R_col, t_col, focal_length, principal_point, imgsz, device
#             )

#             meshes_b = meshes_flat[cursor:cursor+Kb]
#             cursor  += Kb

#             rast_settings = RasterizationSettings(
#                 image_size=(H, W),
#                 blur_radius=0.0,
#                 faces_per_pixel=self.faces_per_pixel,
#                 cull_backfaces=True,
#                 bin_size=0
#             )
#             rasterizer = MeshRasterizer(cameras=cameras, raster_settings=rast_settings)
#             fragments  = rasterizer(meshes_b)  # (Kb,H,W,F)

#             shader = SoftSilhouetteShader(blend_params=self.blend_params)
#             rgba   = shader(fragments, meshes_b, cameras=cameras)    # (Kb,H,W,4)
#             alpha  = rgba[..., 3].clamp_(0.0, 1.0)                   # (Kb,H,W)

#             pix2face = fragments.pix_to_face                         # (Kb,H,W,F)  -1: none
#             valid_pix = (pix2face >= 0)
#             zbuf = fragments.zbuf                                    # (Kb,H,W,F)
#             zbuf_masked = torch.where(valid_pix, zbuf, torch.full_like(zbuf, float('inf')))
#             depth_min, _ = zbuf_masked.min(dim=-1)                   # (Kb,H,W)
#             depth_min = torch.where(torch.isfinite(depth_min), depth_min, torch.zeros_like(depth_min))

#             # --- 画像ごとの union（既存動作） ---
#             sil_b = 1.0 - torch.prod(1.0 - alpha, dim=0, keepdim=True)   # (1,H,W)
#             dep_b, _ = torch.min(
#                 torch.where(alpha > 0, depth_min, torch.full_like(depth_min, float('inf'))),
#                 dim=0, keepdim=True
#             )
#             dep_b = torch.where(torch.isfinite(dep_b), dep_b, torch.zeros_like(dep_b))
#             dep_b = dep_b * (sil_b > 0).to(dep_b.dtype)

#             sil_out.append(sil_b.unsqueeze(0))   # (1,1,H,W)
#             dep_out.append(dep_b.unsqueeze(0))   # (1,1,H,W)

#             # ============ 追加: インスタンスセグメンテーション系の生成 ============
#             # 1) ソフト・シルエット（オクルージョン非考慮）をグローバルKに配置
#             inst_alpha_all[b, global_idx] = alpha  # (Kb,H,W) を (K,H,W) の相応スロットに

#             # 2) 可視インスタンス（最前面）IDマップと可視マスク
#             # depth_min を使って最前面インスタンスを選ぶ。ただし背景（全部inf）を除外
#             depth_vis = torch.where(alpha > 0, depth_min, torch.full_like(depth_min, float('inf')))  # (Kb,H,W)
#             min_depth, owner_kb = depth_vis.min(dim=0)  # (H,W), (H,W)  kb ∈ [0,Kb)
#             has_fg = torch.isfinite(min_depth)          # (H,W)

#             # owner のグローバルIDに変換
#             owner_global = torch.full((H, W), -1, device=device, dtype=torch.int32)
#             owner_global[has_fg] = global_idx[owner_kb[has_fg]].to(torch.int32)

#             inst_id_map_all[b, 0] = owner_global  # (1,H,W)

#             # 可視マスク: one-hot(owner_global==k)
#             # まずローカル one-hot（Kb,H,W）
#             vis_local = torch.zeros(Kb, H, W, device=device, dtype=torch.float32)
#             vis_local.scatter_(0, owner_kb.unsqueeze(0), 1.0)
#             vis_local = vis_local * has_fg.to(vis_local.dtype)  # 背景を0に

#             # グローバルKに配置
#             inst_vis_all[b, global_idx] = vis_local

#         silhouette = torch.cat(sil_out, dim=0)  # (B,1,H,W)
#         depth      = torch.cat(dep_out, dim=0)  # (B,1,H,W)

#         return {
#             "silhouette": silhouette.to(dtype),          # (B,1,H,W)
#             "depth":      depth.to(dtype),               # (B,1,H,W)
#             # 追加分:
#             "inst_alpha":         inst_alpha_all.to(dtype),  # (B,K,H,W)  ソフトα（重なり無視）
#             "inst_visible_masks": inst_vis_all.to(dtype),    # (B,K,H,W)  可視（最前面）1/0
#             "inst_id_map":        inst_id_map_all,           # (B,1,H,W)  int32, 背景=-1
#         }


class SilhouetteDepthRenderer(torch.nn.Module):
    """
    目的：
      - 完全なバイナリ物体マスク（外形で塗りつぶし）
      - 最前面（手前）深度
      - 可視IDマップ（前景画素の所有インスタンス）
      - インスタンスごとの(非オクルージョン考慮の)バイナリマスク / 可視マスク

    重要ポイント：
      * rasterizer.fragments.pix_to_face[..., 0] >= 0 を「ヒット判定」として使用
      * depth は zbuf[..., 0] を各インスタンスで取り、min を画像方向に取る
      * Shader は使わない（ソフトシルエット由来の“網目表示”を排除）
      * faces_per_pixel は 1 を強制（トップヒットのみ）
    """
    def __init__(self, cull_backfaces: bool = False):
        super().__init__()
        # cull_backfaces=True だと非閉メッシュや法線反転で“穴”が出やすい
        self.cull_backfaces = cull_backfaces

    @torch.no_grad()
    def _K_to_fx_fy_cx_cy(self, K_left_b: torch.Tensor):
        fx = K_left_b[0, 0]; fy = K_left_b[1, 1]
        cx = K_left_b[0, 2]; cy = K_left_b[1, 2]
        return fx, fy, cx, cy

    def forward(
        self,
        meshes_flat: Meshes,          # len = sum(valid_k)
        T_cam_obj: torch.Tensor,      # (B,K,4,4)  列ベクトル系
        K_left: torch.Tensor,         # (B,3,3)
        valid_k: torch.Tensor,        # (B,K)  bool/0-1
        image_size: Tuple[int, int],  # (H,W)
    ):
        device = K_left.device
        dtype  = K_left.dtype
        B, K = valid_k.shape
        H, W = image_size

        sil_out = []
        dep_out = []

        # 追加: インスタンスごとの出力（グローバルKに合わせた形）
        inst_alpha_all   = torch.zeros(B, K, H, W, device=device, dtype=torch.float32)   # 非オクルージョン考慮のバイナリ
        inst_vis_all     = torch.zeros(B, K, H, W, device=device, dtype=torch.float32)   # 最前面可視のみ1
        inst_id_map_all  = torch.full((B, 1, H, W), -1, device=device, dtype=torch.int32)

        cursor = 0
        K_left_f32    = K_left.to(torch.float32)
        T_cam_obj_f32 = T_cam_obj.to(torch.float32)

        # ラスタライザ設定：トップヒットのみ、AAなし
        # faces_per_pixel=1 を強制してトップヒットだけを使う
        # blur_radius=0（アンチエイリアス無し）
        def make_rasterizer(cameras):
            rast_settings = RasterizationSettings(
                image_size=(H, W),
                blur_radius=0.0,
                faces_per_pixel=1,
                cull_backfaces=self.cull_backfaces,
                bin_size=0,             # 0=自動（大画像で速いケース多い）
            )
            return MeshRasterizer(cameras=cameras, raster_settings=rast_settings)

        inf_val = float('inf')

        for b in range(B):
            vmask_b = (valid_k[b] > 0)                 # (K,)
            Kb = int(vmask_b.sum().item())
            if Kb == 0:
                sil_out.append(torch.zeros(1, 1, H, W, device=device, dtype=torch.float32))
                dep_out.append(torch.zeros(1, 1, H, W, device=device, dtype=torch.float32))
                continue

            # ローカルindex(kb)→グローバルindex(k)
            global_idx = torch.nonzero(vmask_b, as_tuple=False).squeeze(1)  # (Kb,)

            T_b   = T_cam_obj_f32[b, vmask_b]       # (Kb,4,4)
            R_col = T_b[:, :3, :3].contiguous()
            t_col = T_b[:, :3,  3].contiguous()

            fx, fy, cx, cy = self._K_to_fx_fy_cx_cy(K_left_f32[b])
            focal_length    = torch.stack([fx.expand(Kb), fy.expand(Kb)], dim=1)  # (Kb,2)
            principal_point = torch.stack([cx.expand(Kb), cy.expand(Kb)], dim=1)  # (Kb,2)
            imgsz = torch.tensor([[H, W]], dtype=torch.float32, device=device).expand(Kb, -1).contiguous()

            cameras = cameras_from_opencv_projection(
                R_col, t_col, focal_length, principal_point, imgsz, device
            )

            meshes_b = meshes_flat[cursor:cursor+Kb]
            cursor  += Kb

            rasterizer = make_rasterizer(cameras)
            fragments  = rasterizer(meshes_b)  # (Kb,H,W,1)

            # トップヒットのみで完全二値のインスタンスマスク
            pix2face_top = fragments.pix_to_face[..., 0]               # (Kb,H,W)
            hit0 = (pix2face_top >= 0)                                 # (Kb,H,W)
            alpha_bin = hit0.float()                                   # (Kb,H,W)

            # 各インスタンスの（他インスタンスのオクルージョンを無視した）バイナリマスク
            inst_alpha_all[b, global_idx] = alpha_bin

            # 各インスタンスのトップヒット深度
            z0 = fragments.zbuf[..., 0]                                # (Kb,H,W)
            z0 = torch.where(hit0, z0, torch.full_like(z0, inf_val))   # ヒットなしは inf

            # 画像全体のシルエット（Kb 方向の OR）
            sil_b = (alpha_bin.any(dim=0, keepdim=True)).float()       # (1,H,W)

            # 画像全体の最表面深度（Kb 方向の min）
            dep_b, owner_kb = z0.min(dim=0, keepdim=True)              # (1,H,W), (1,H,W)※owner_kb shapeはブロードキャスト対応
            dep_b = torch.where(torch.isfinite(dep_b), dep_b, torch.zeros_like(dep_b))
            dep_b = dep_b * sil_b

            sil_out.append(sil_b.unsqueeze(0))   # (1,1,H,W)
            dep_out.append(dep_b.unsqueeze(0))   # (1,1,H,W)

            # ===== 可視インスタンスIDマップと可視マスク（最前面） =====
            # min の index を (H,W) に整形
            owner_kb_hw = owner_kb[0]                               # (H,W)
            min_depth_hw = dep_b[0]                                 # (H,W) すでに背景は0
            has_fg = (sil_b[0] > 0)                                 # (H,W)

            # グローバルIDに変換
            owner_global = torch.full((H, W), -1, device=device, dtype=torch.int32)
            if has_fg.any():
                owner_global[has_fg] = global_idx[owner_kb_hw[has_fg]].to(torch.int32)
            inst_id_map_all[b, 0] = owner_global

            # 可視 one-hot（最前面だけ 1）
            if has_fg.any():
                vis_local = torch.zeros(Kb, H, W, device=device, dtype=torch.float32)
                # scatter_ するため index 形状を合わせる
                vis_local.scatter_(0, owner_kb_hw.unsqueeze(0), 1.0)
                vis_local = vis_local * has_fg.to(vis_local.dtype)
                inst_vis_all[b, global_idx] = vis_local

        silhouette = torch.cat(sil_out, dim=0)  # (B,1,1,H,W) → 実際は (B,1,1,H,W)? 上で(1,1,H,W)入れているので (B,1,1,H,W)
        depth      = torch.cat(dep_out, dim=0)  # (B,1,1,H,W)
        return {
            "silhouette": silhouette.to(dtype),          # (B,1,H,W)  完全二値（OR）
            "depth":      depth.to(dtype),               # (B,1,H,W)  最表面深度
            "inst_alpha":         inst_alpha_all.to(dtype),  # (B,K,H,W)  非オクルージョンの各インスタンス二値マスク
            "inst_visible_masks": inst_vis_all.to(dtype),    # (B,K,H,W)  可視（最前面）のみ 1
            "inst_id_map":        inst_id_map_all,           # (B,1,H,W)  int32, 背景=-1
        }