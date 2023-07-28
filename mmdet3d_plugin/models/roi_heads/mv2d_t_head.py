# ------------------------------------------------------------------------
# Modified from PETR (https://github.com/megvii-research/PETR)
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.core import bbox2roi
from mmdet.models.builder import HEADS
from .mv2d_s_head import MV2DSHead


@HEADS.register_module()
class MV2DTHead(MV2DSHead):
    def __init__(self,
                 num_views=6,
                 **kwargs):
        super(MV2DTHead, self).__init__(**kwargs)
        self.num_views = num_views

    def _bbox_forward_denoise(self, x, proposal_list, img_metas):
        # avoid empty 2D detection
        if sum([len(p) for p in proposal_list]) == 0:
            proposal = torch.tensor([[0, 50, 50, 100, 100, 0]], dtype=proposal_list[0].dtype,
                                    device=proposal_list[0].device)
            proposal_list = [proposal] + proposal_list[1:]

        rois = bbox2roi(proposal_list)
        intrinsics, extrinsics = self.get_box_params(proposal_list,
                                                     [img_meta['intrinsics'] for img_meta in img_metas],
                                                     [img_meta['extrinsics'] for img_meta in img_metas])
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)

        # 3dpe was concatenated to fpn feature
        c = bbox_feats.size(1)
        bbox_feats, _ = bbox_feats.split([c // 2, c // 2], dim=1)

        # intrinsics as extra input feature
        extra_feats = dict(
            intrinsic=self.process_intrins_feat(rois, intrinsics)
        )

        # query generator
        reference_points, return_feats = self.query_generator(bbox_feats, intrinsics, extrinsics, extra_feats)
        reference_points[..., 0:1] = (reference_points[..., 0:1] - self.pc_range[0]) / (
                self.pc_range[3] - self.pc_range[0])
        reference_points[..., 1:2] = (reference_points[..., 1:2] - self.pc_range[1]) / (
                self.pc_range[4] - self.pc_range[1])
        reference_points[..., 2:3] = (reference_points[..., 2:3] - self.pc_range[2]) / (
                self.pc_range[5] - self.pc_range[2])
        reference_points.clamp(min=0, max=1)

        # split image features and 3dpe
        feat, pe = x[self.feat_lvl].split([c // 2, c // 2], dim=1)  # [num_views, c, h, w]
        stride = self.strides[self.feat_lvl]

        # box correlation
        num_rois_per_img = [len(p) for p in proposal_list]
        feat_for_rois = self.box_corr_module.gen_box_correlation(rois, num_rois_per_img, img_metas, feat, stride)

        # generate image padding mask
        num_views, c, h, w = feat.shape
        mask = torch.zeros_like(feat[:, 0]).bool()  # [num_views, h, w]
        input_img_h, input_img_w, _ = img_metas[0]['pad_shape']
        mask_outside = feat.new_ones((1, num_views, input_img_h, input_img_w))
        for img_id in range(num_views):
            img_h, img_w, _ = img_metas[img_id]['img_shape']
            mask_outside[0, img_id, :img_h, :img_w] = 0
        mask_outside = F.interpolate(mask_outside, size=feat.shape[-2:]).to(torch.bool)[0]
        mask[mask_outside] = 1

        # generate cross attention mask
        cross_attn_mask = ~feat_for_rois
        if self.training:
            invalid_rois = cross_attn_mask.view(cross_attn_mask.size(0), -1).all(1)
            cross_attn_mask[invalid_rois, 0, 0, 0] = 0

        roi_mask = (~cross_attn_mask).any(dim=0)  # [num_views, h, w], 1 for valid
        feat = feat.permute(0, 2, 3, 1)[roi_mask][..., None, None]  # [num_valid, c, 1, 1]
        pe = pe.permute(0, 2, 3, 1)[roi_mask][..., None, None]
        mask = mask[roi_mask][..., None, None]
        cross_attn_mask = cross_attn_mask[:, roi_mask][..., None, None]  # [num_rois, num_valid, 1, 1]

        # denoise training
        if self.use_denoise:
            num_ori_reference = len(reference_points)
            reference_points, attn_mask, mask_dict = self.prepare_for_dn(
                1, reference_points, img_metas[0:1], num_ori_reference)
            reference_points = reference_points[0]
            num_pad_reference = len(reference_points) - num_ori_reference
            pad_cross_attn_mask = cross_attn_mask.all(dim=0)[None].repeat(num_pad_reference, 1, 1, 1)
            cross_attn_mask = torch.cat([pad_cross_attn_mask, cross_attn_mask], dim=0)
        else:
            attn_mask = None
            mask_dict = None

        all_cls_scores, all_bbox_preds = self.bbox_head(reference_points[None],
                                                        feat[None],
                                                        mask[None],
                                                        pe[None],
                                                        attn_mask=attn_mask,
                                                        cross_attn_mask=cross_attn_mask,
                                                        force_fp32=self.force_fp32, )

        if mask_dict and mask_dict['pad_size'] > 0:
            output_known_class = all_cls_scores[:, :, :mask_dict['pad_size'], :]
            output_known_coord = all_bbox_preds[:, :, :mask_dict['pad_size'], :]
            mask_dict['output_known_lbs_bboxes'] = (output_known_class, output_known_coord)
            all_cls_scores = all_cls_scores[:, :, mask_dict['pad_size']:, :]
            all_bbox_preds = all_bbox_preds[:, :, mask_dict['pad_size']:, :]

        cls_scores, bbox_preds = [], []
        for c, b in zip(all_cls_scores, all_bbox_preds):
            cls_scores.append(c.flatten(0, 1))
            bbox_preds.append(b.flatten(0, 1))

        bbox_results = dict(
            cls_scores=cls_scores, bbox_preds=bbox_preds, bbox_feats=bbox_feats, return_feats=return_feats,
            intrinsics=intrinsics, extrinsics=extrinsics, rois=rois, dn_mask_dict=mask_dict,
        )

        return bbox_results

    def _bbox_forward(self, x, proposal_list, img_metas):
        time_stamp = np.array([img_meta['timestamp'] for img_meta in img_metas])
        mean_time_stamp = time_stamp[self.num_views:].mean() - time_stamp[:self.num_views].mean()

        bbox_results = self._bbox_forward_denoise(x, proposal_list, img_metas)

        if len(img_metas) > self.num_views:
            bbox_preds = bbox_results['bbox_preds']
            bbox_preds_with_time = [
                torch.cat([pred[..., :8], pred[..., 8:] / mean_time_stamp], dim=-1) for pred in bbox_preds]
            bbox_results['bbox_preds'] = bbox_preds_with_time

        return bbox_results
