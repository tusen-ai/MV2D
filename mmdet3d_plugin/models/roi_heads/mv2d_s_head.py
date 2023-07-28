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
from .mv2d_head import MV2DHead


@HEADS.register_module()
class MV2DSHead(MV2DHead):
    def __init__(self,
                 # denoise setting
                 use_denoise=False,
                 neg_bbox_loss=False,
                 denoise_scalar=10,
                 denoise_noise_scale=1.0,
                 denoise_noise_trans=0.0,
                 denoise_weight=1.0,
                 denoise_split=0.75,
                 **kwargs):
        super(MV2DSHead, self).__init__(**kwargs)
        self.use_denoise = use_denoise
        self.neg_bbox_loss = neg_bbox_loss
        self.denoise_scalar = denoise_scalar
        self.denoise_noise_scale = denoise_noise_scale
        self.denoise_noise_trans = denoise_noise_trans
        self.denoise_weight = denoise_weight
        self.denoise_split = denoise_split

    def prepare_for_dn(self, batch_size, reference_points, img_metas, ref_num, eps=1e-4):
        if self.training:
            targets = [
                torch.cat((img_meta['gt_bboxes_3d'].gravity_center, img_meta['gt_bboxes_3d'].tensor[:, 3:]),
                          dim=1) for img_meta in img_metas]
            labels = [img_meta['gt_labels_3d'] for img_meta in img_metas]
            known = [(torch.ones_like(t)).cuda() for t in labels]
            know_idx = known
            unmask_bbox = unmask_label = torch.cat(known)
            known_num = [t.size(0) for t in targets]
            labels = torch.cat([t for t in labels])
            boxes = torch.cat([t for t in targets])
            batch_idx = torch.cat([torch.full((t.size(0),), i) for i, t in enumerate(targets)])

            known_indice = torch.nonzero(unmask_label + unmask_bbox)
            known_indice = known_indice.view(-1)
            # add noise
            known_indice = known_indice.repeat(self.denoise_scalar, 1).view(-1)
            known_labels = labels.repeat(self.denoise_scalar, 1).view(-1).long().to(reference_points.device)
            known_bid = batch_idx.repeat(self.denoise_scalar, 1).view(-1)
            known_bboxs = boxes.repeat(self.denoise_scalar, 1).to(reference_points.device)
            known_bbox_center = known_bboxs[:, :3].clone()
            known_bbox_scale = known_bboxs[:, 3:6].clone()

            if self.denoise_noise_scale > 0:
                diff = known_bbox_scale / 2 + self.denoise_noise_trans
                rand_prob = torch.rand_like(known_bbox_center) * 2 - 1.0
                known_bbox_center += torch.mul(rand_prob,
                                               diff) * self.denoise_noise_scale
                known_bbox_center[..., 0:1] = (known_bbox_center[..., 0:1] - self.pc_range[0]) / (
                        self.pc_range[3] - self.pc_range[0])
                known_bbox_center[..., 1:2] = (known_bbox_center[..., 1:2] - self.pc_range[1]) / (
                        self.pc_range[4] - self.pc_range[1])
                known_bbox_center[..., 2:3] = (known_bbox_center[..., 2:3] - self.pc_range[2]) / (
                        self.pc_range[5] - self.pc_range[2])
                known_bbox_center = known_bbox_center.clamp(min=0.0 + eps, max=1.0 - eps)
                mask = torch.norm(rand_prob, 2, 1) > self.denoise_split
                known_labels[mask] = self.num_classes

            single_pad = int(max(known_num))
            pad_size = int(single_pad * self.denoise_scalar)
            padding_bbox = torch.zeros(pad_size, 3).to(reference_points.device)
            padded_reference_points = torch.cat([padding_bbox, reference_points], dim=0).unsqueeze(0).repeat(batch_size,
                                                                                                             1, 1)

            if len(known_num):
                map_known_indice = torch.cat([torch.tensor(range(num)) for num in known_num])  # [1,2, 1,2,3]
                map_known_indice = torch.cat(
                    [map_known_indice + single_pad * i for i in range(self.denoise_scalar)]).long()
            if len(known_bid):
                padded_reference_points[(known_bid.long(), map_known_indice)] = known_bbox_center.to(
                    reference_points.device)

            tgt_size = pad_size + ref_num
            attn_mask = torch.ones(tgt_size, tgt_size).to(reference_points.device) < 0
            # match query cannot see the reconstruct
            attn_mask[pad_size:, :pad_size] = True
            # reconstruct cannot see each other
            for i in range(self.denoise_scalar):
                if i == 0:
                    attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
                if i == self.denoise_scalar - 1:
                    attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
                else:
                    attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
                    attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True

            mask_dict = {
                'known_indice': torch.as_tensor(known_indice).long(),
                'batch_idx': torch.as_tensor(batch_idx).long(),
                'map_known_indice': torch.as_tensor(map_known_indice).long(),
                'known_lbs_bboxes': (known_labels, known_bboxs),
                'know_idx': know_idx,
                'pad_size': pad_size
            }

        else:
            padded_reference_points = reference_points.unsqueeze(0).repeat(batch_size, 1, 1)
            attn_mask = None
            mask_dict = None

        return padded_reference_points, attn_mask, mask_dict

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
        bbox_feats, pe = bbox_feats.split([c // 2, c // 2], dim=1)

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

        # generate box correlation
        corr, mask = self.box_corr_module.gen_box_roi_correlation(rois, [len(p) for p in proposal_list], img_metas)

        if self.use_denoise and self.training:
            # bbox_feats: [num_rois, c, h, w]
            n_rois, c, h, w = bbox_feats.shape
            cross_attn_mask = bbox_feats.new_ones((n_rois, n_rois + 1)).bool()
            corr[~mask] = n_rois  # [num_rois, max_corr]
            cross_attn_mask = torch.scatter(cross_attn_mask, 1, corr, 0)
            cross_attn_mask = cross_attn_mask[:, :n_rois, None, None].expand(n_rois, n_rois, h, w)

            reference_points_ori = reference_points
            reference_points, attn_mask, mask_dict = self.prepare_for_dn(1, reference_points, img_metas[0:1],
                                                                         len(reference_points))
            reference_points = reference_points[0]
            cross_attn_mask_pad = cross_attn_mask.new_zeros(
                (len(reference_points) - len(reference_points_ori), n_rois, h, w))
            cross_attn_mask = torch.cat([cross_attn_mask_pad, cross_attn_mask])

            all_cls_scores, all_bbox_preds = self.bbox_head(reference_points[None],
                                                            bbox_feats[None],
                                                            torch.zeros_like(bbox_feats[None, :, 0]).bool(),
                                                            pe[None],
                                                            attn_mask=attn_mask,
                                                            cross_attn_mask=cross_attn_mask,
                                                            force_fp32=self.force_fp32, )
        else:
            mask_dict = None

            corr_feats = bbox_feats[corr]  # [num_rois, num_corrs, c, h, w]
            corr_pe = pe[corr]
            all_cls_scores, all_bbox_preds = self.bbox_head(reference_points[:, None],
                                                            corr_feats,
                                                            ~mask[..., None, None].expand_as(corr_feats[:, :, 0]),
                                                            corr_pe,
                                                            attn_mask=None,
                                                            cross_attn_mask=None,
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
        bbox_results = self._bbox_forward_denoise(x, proposal_list, img_metas)
        return bbox_results

    def prepare_for_dn_loss(self, mask_dict):
        """
        prepare dn components to calculate loss
        Args:
            mask_dict: a dict that contains dn information
        """
        output_known_class, output_known_coord = mask_dict['output_known_lbs_bboxes']
        known_labels, known_bboxs = mask_dict['known_lbs_bboxes']
        map_known_indice = mask_dict['map_known_indice'].long()
        known_indice = mask_dict['known_indice'].long()
        batch_idx = mask_dict['batch_idx'].long()
        bid = batch_idx[known_indice]
        if len(output_known_class) > 0:
            output_known_class = output_known_class.permute(1, 2, 0, 3)[(bid, map_known_indice)].permute(1, 0, 2)
            output_known_coord = output_known_coord.permute(1, 2, 0, 3)[(bid, map_known_indice)].permute(1, 0, 2)
        num_tgt = known_indice.numel()
        return known_labels, known_bboxs, output_known_class, output_known_coord, num_tgt

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_3d,
                      gt_labels_3d,
                      ori_gt_bboxes_3d,
                      ori_gt_labels_3d,
                      attr_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      **kwargs):
        assert len(img_metas) // img_metas[0]['num_views'] == 1

        num_imgs = len(img_metas)

        proposal_boxes = []
        proposal_scores = []
        proposal_classes = []
        for i in range(num_imgs):
            proposal_boxes.append(proposal_list[i][:, :6])
            proposal_scores.append(proposal_list[i][:, 4])
            proposal_classes.append(proposal_list[i][:, 5])

        # position encoding
        pos_enc = self.position_encoding(x, img_metas)
        x = [torch.cat([feat, pe], dim=1) for feat, pe in zip(x, pos_enc)]

        losses = dict()

        if self.use_denoise:
            img_metas[0]['gt_bboxes_3d'] = ori_gt_bboxes_3d[0]
            img_metas[0]['gt_labels_3d'] = ori_gt_labels_3d[0]

        results_from_last = self._bbox_forward_train(x, proposal_boxes, img_metas)
        preds = results_from_last['pred']

        cls_scores = preds['cls_scores']
        bbox_preds = preds['bbox_preds']
        loss_weights = copy.deepcopy(self.stage_loss_weights)

        # use the matching results from last stage for loss calculation
        loss_stage = []
        num_layers = len(cls_scores)
        for layer in range(num_layers):
            loss_bbox = self.bbox_head.loss(
                ori_gt_bboxes_3d, ori_gt_labels_3d, {'cls_scores': [cls_scores[num_layers - 1 - layer]],
                                                     'bbox_preds': [bbox_preds[num_layers - 1 - layer]]},
            )
            loss_stage.insert(0, loss_bbox)

        if results_from_last.get('dn_mask_dict', None) is not None:
            dn_mask_dict = results_from_last['dn_mask_dict']
            known_labels, known_bboxs, output_known_class, output_known_coord, num_tgt = self.prepare_for_dn_loss(
                dn_mask_dict)
            for i in range(len(output_known_class)):
                dn_loss_cls, dn_loss_bbox = self.bbox_head.dn_loss_single(
                    output_known_class[i], output_known_coord[i], known_bboxs, known_labels, num_tgt,
                    self.pc_range, self.denoise_split, neg_bbox_loss=self.neg_bbox_loss
                )
                losses[f'l{i}.dn_loss_cls'] = dn_loss_cls * self.denoise_weight * loss_weights[i]
                losses[f'l{i}.dn_loss_bbox'] = dn_loss_bbox * self.denoise_weight * loss_weights[i]

        for layer in range(num_layers):
            lw = loss_weights[layer]
            for k, v in loss_stage[layer].items():
                losses[f'l{layer}.{k}'] = v * lw if 'loss' in k else v

        return losses
