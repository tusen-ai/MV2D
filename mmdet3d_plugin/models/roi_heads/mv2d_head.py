# Copyright (c) OpenMMLab. All rights reserved.
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from mmdet.models.roi_heads.base_roi_head import BaseRoIHead
from mmdet.models.roi_heads.test_mixins import BBoxTestMixin, MaskTestMixin
from mmdet.core import bbox2roi
from mmdet.models.builder import HEADS, build_head, build_roi_extractor
from mmdet3d_plugin.models.utils.pe import PE
from mmcv.cnn import ConvModule
from .utils.box_correlation import BoxCorrelation
from .utils.query_generator import QueryGenerator


@HEADS.register_module()
class MV2DHead(BaseRoIHead, BBoxTestMixin, MaskTestMixin):
    def __init__(self,
                 bbox_roi_extractor,
                 bbox_head,
                 query_generator,
                 pe,
                 box_correlation,
                 pc_range,
                 intrins_feat_scale=0.1,
                 feat_lvl=0,
                 force_fp32=False,
                 train_cfg=None,
                 test_cfg=None,
                 **kwargs):
        super(MV2DHead, self).__init__(bbox_roi_extractor=bbox_roi_extractor, bbox_head=bbox_head,
                                       train_cfg=train_cfg, test_cfg=test_cfg, **kwargs)
        self.roi_size = bbox_roi_extractor['roi_layer']['output_size']
        if isinstance(self.roi_size, int):
            self.roi_size = [self.roi_size, self.roi_size]

        query_generator.update(dict(loss_cls=self.bbox_head.loss_cls))
        self.query_generator = QueryGenerator(**query_generator)
        self.position_encoding = PE(**pe)
        self.box_corr_module = BoxCorrelation(**box_correlation)

        self.pc_range = pc_range
        self.intrins_feat_scale = intrins_feat_scale
        self.feat_lvl = feat_lvl

        self.stage_loss_weights = train_cfg.get('stage_loss_weights') if train_cfg else None
        self.force_fp32 = force_fp32

    @torch.no_grad()
    def get_box_params(self, bboxes, intrinsics, extrinsics):
        # TODO: check grad flow from boxes to intrinsic
        intrinsic_list = []
        extrinsic_list = []
        for img_id, (bbox, intrinsic, extrinsic) in enumerate(zip(bboxes, intrinsics, extrinsics)):
            # bbox: [n, (x, y, x, y)], rois_i: [n, c, h, w], intrinsic: [4, 4], extrinsic: [4, 4]
            intrinsic = torch.from_numpy(intrinsic).to(bbox.device).double()
            extrinsic = torch.from_numpy(extrinsic).to(bbox.device).double()
            intrinsic = intrinsic.repeat(bbox.shape[0], 1, 1)
            extrinsic = extrinsic.repeat(bbox.shape[0], 1, 1)
            # consider corners
            wh_bbox = bbox[:, 2:4] - bbox[:, :2]
            wh_roi = wh_bbox.new_tensor(self.roi_size)
            scale = wh_roi[None] / wh_bbox
            intrinsic[:, :2, 2] = intrinsic[:, :2, 2] - bbox[:, :2] - 0.5 / scale
            intrinsic[:, :2] = intrinsic[:, :2] * scale[..., None]
            intrinsic_list.append(intrinsic)
            extrinsic_list.append(extrinsic)
        intrinsic_list = torch.cat(intrinsic_list, 0)
        extrinsic_list = torch.cat(extrinsic_list, 0)
        return intrinsic_list, extrinsic_list

    @property
    def strides(self):
        return self.position_encoding.strides

    @property
    def num_classes(self):
        return self.bbox_head.num_classes

    def init_assigner_sampler(self):
        self.bbox_assigner = None
        self.bbox_sampler = None

    def init_bbox_head(self, bbox_roi_extractor, bbox_head):
        """Initialize ``bbox_head``"""
        self.bbox_roi_extractor = build_roi_extractor(bbox_roi_extractor)
        bbox_head.update(dict(train_cfg=self.train_cfg, test_cfg=self.test_cfg))
        self.bbox_head = build_head(bbox_head)

    def init_mask_head(self, mask_roi_extractor, mask_head):
        raise NotImplementedError

    def process_intrins_feat(self, rois, intrinsics, min_size=4):
        intrinsics = intrinsics.view(intrinsics.shape[0], 16).clone().float()
        intrinsics = intrinsics * self.intrins_feat_scale
        wh_bbox = rois[:, 3:5] - rois[:, 1:3]
        invalid_bbox = (wh_bbox < min_size).any(1)
        intrinsics[invalid_bbox] = 0
        return intrinsics

    def _bbox_forward(self, x, proposal_list, img_metas):
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
        pe = pe.permute(0, 2, 3, 1)[roi_mask][..., None, None]      # [num_valid, c, 1, 1]
        mask = mask[roi_mask][..., None, None]                      # [num_valid, 1, 1]
        cross_attn_mask = cross_attn_mask[:, roi_mask][..., None, None]  # [num_rois, num_valid, 1, 1]

        all_cls_scores, all_bbox_preds = self.bbox_head(reference_points[None],
                                                        feat[None],
                                                        mask[None],
                                                        pe[None],
                                                        cross_attn_mask=cross_attn_mask,
                                                        pe=(self.position_encoding, x, img_metas),
                                                        force_fp32=self.force_fp32
                                                        )

        cls_scores, bbox_preds = [], []
        for c, b in zip(all_cls_scores, all_bbox_preds):
            cls_scores.append(c.flatten(0, 1))
            bbox_preds.append(b.flatten(0, 1))

        bbox_results = dict(
            cls_scores=cls_scores, bbox_preds=bbox_preds, bbox_feats=bbox_feats, return_feats=return_feats,
            intrinsics=intrinsics, extrinsics=extrinsics, rois=rois,
        )

        return bbox_results

    def _bbox_forward_train(self, x, proposal_list, img_metas):
        """Run forward function and calculate loss for box head in training."""

        bbox_results = self._bbox_forward(x, proposal_list, img_metas)
        bbox_results.update(pred={'cls_scores': bbox_results['cls_scores'], 'bbox_preds': bbox_results['bbox_preds']})

        return bbox_results

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

        num_views = len(img_metas)

        proposal_boxes = []
        proposal_scores = []
        proposal_classes = []
        for i in range(num_views):
            proposal_boxes.append(proposal_list[i][:, :6])
            proposal_scores.append(proposal_list[i][:, 4])
            proposal_classes.append(proposal_list[i][:, 5])

        # position encoding
        pos_enc = self.position_encoding(x, img_metas)
        x = [torch.cat([feat, pe], dim=1) for feat, pe in zip(x, pos_enc)]

        losses = dict()
        results_from_last = self._bbox_forward_train(x, proposal_boxes, img_metas)

        cls_scores = results_from_last['pred']['cls_scores']
        bbox_preds = results_from_last['pred']['bbox_preds']

        # use the matching results from last stage for loss calculation
        loss_stage = []
        num_layers = len(cls_scores)
        for layer in range(num_layers):
            loss_bbox = self.bbox_head.loss(
                ori_gt_bboxes_3d, ori_gt_labels_3d, {'cls_scores': [cls_scores[num_layers - 1 - layer]],
                                                     'bbox_preds': [bbox_preds[num_layers - 1 - layer]]},
            )
            loss_stage.insert(0, loss_bbox)

        for layer in range(num_layers):
            lw = self.stage_loss_weights[layer]
            for k, v in loss_stage[layer].items():
                losses[f'l{layer}.{k}'] = v * lw if 'loss' in k else v

        return losses

    def simple_test(self, x, proposal_list, img_metas, rescale=False):
        assert self.with_bbox, 'Bbox head must be implemented.'
        assert len(img_metas) // img_metas[0]['num_views'] == 1

        # position encoding
        pos_enc = self.position_encoding(x, img_metas)
        x = [torch.cat([feat, pe], dim=1) for feat, pe in zip(x, pos_enc)]

        results_from_last = dict()

        results_from_last['batch_size'] = len(img_metas) // img_metas[0]['num_views']
        results_from_last = self._bbox_forward(x, proposal_list, img_metas)

        cls_scores = results_from_last['cls_scores'][-1]
        bbox_preds = results_from_last['bbox_preds'][-1]

        bbox_list = self.bbox_head.get_bboxes({'cls_scores': [cls_scores], 'bbox_preds': [bbox_preds]}, img_metas,)

        return bbox_list
