# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from os import path as osp

import mmcv
import numpy as np
import torch
import os
import cv2

from mmdet3d.core import (bbox3d2result, box3d_multiclass_nms)
from mmdet3d.models.builder import DETECTORS, build_detector, build_head, build_neck
from .mv2d import MV2D


@DETECTORS.register_module()
class MV2DT(MV2D):
    def __init__(self,
                 num_views=6,
                 grad_all=True,
                 **kwargs,
                 ):
        super(MV2DT, self).__init__(**kwargs)
        self.num_views = num_views
        self.grad_all = grad_all

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes_2d,
                      gt_labels_2d,
                      gt_bboxes_2d_to_3d,
                      gt_bboxes_3d,
                      gt_labels_3d,
                      attr_labels=None,
                      gt_bboxes_ignore=None):

        losses = dict()

        batch_size, num_views, c, h, w = img.shape
        img = img.view(batch_size * num_views, *img.shape[2:])
        assert batch_size == 1, 'only support batch_size 1 now'

        if self.use_grid_mask:
            img = self.grid_mask(img)

        # get pseudo monocular input
        # gt_bboxes, gt_labels, gt_bboxes_3d, gt_labels_3d, gt_bboxes_ignore, img_metas:
        #   independent GT for each view
        # ori_gt_bboxes_3d, ori_gt_labels_3d:
        #   original GT for all the views
        ori_img_metas, ori_gt_bboxes_3d, ori_gt_labels_3d, ori_gt_bboxes_ignore = img_metas, gt_bboxes_3d, gt_labels_3d, gt_bboxes_ignore
        gt_bboxes, gt_labels, gt_bboxes_3d, gt_labels_3d, gt_bboxes_ignore, img_metas = [], [], [], [], [], []
        for i in range(batch_size):
            img_metas_views = ori_img_metas[i]
            for j in range(num_views):
                img_meta = dict(num_views=num_views)
                for k, v in img_metas_views.items():
                    if isinstance(v, list):
                        img_meta[k] = v[j]
                    elif k == 'ori_shape':
                        img_meta[k] = v[:3]
                    else:
                        img_meta[k] = v
                img_metas.append(img_meta)

            gt_labels_3d_views = ori_gt_labels_3d[i]
            gt_bboxes_3d_views = ori_gt_bboxes_3d[i].to(gt_labels_3d_views.device)
            for j in range(self.num_views):
                gt_ids = (gt_bboxes_2d_to_3d[i][j]).unique()
                select = gt_ids[gt_ids > -1].long()
                gt_bboxes_3d.append(gt_bboxes_3d_views[select])
                gt_labels_3d.append(gt_labels_3d_views[select])
            # no GT in previous frames
            for j in range(self.num_views, num_views):
                box_type = gt_bboxes_3d[0].__class__
                box_dim = gt_bboxes_3d[0].tensor.size(-1)
                gt_bboxes_3d.append(box_type(img.new_zeros((0, box_dim)), box_dim=box_dim))
                gt_labels_3d.append(img.new_zeros(0, dtype=torch.long))

            gt_bboxes.extend(gt_bboxes_2d[i])
            gt_labels.extend(gt_labels_2d[i])
            gt_bboxes_ignore.extend(ori_gt_bboxes_ignore[i])

        # calculate losses for base detector
        if not self.grad_all:
            detector_feat_current = self.extract_feat(img[:self.num_views])
            with torch.no_grad():
                detector_feat_history = self.extract_feat(img[self.num_views:])
            detector_feat = [torch.cat([x1, x2]) for x1, x2 in zip(detector_feat_current, detector_feat_history)]
        else:
            detector_feat = self.extract_feat(img)
            detector_feat_current = [x[:self.num_views] for x in detector_feat]
            detector_feat_history = [x[self.num_views:] for x in detector_feat]

        # only the current frame is used in 2D detection loss
        img_current = img[:self.num_views]
        img_metas_current = img_metas[:self.num_views]
        losses_detector = self.base_detector.forward_train_w_feat(
            detector_feat_current,
            img_current,
            img_metas_current,
            gt_bboxes,
            gt_labels,
            gt_bboxes_ignore)
        for k, v in losses_detector.items():
            losses['det_' + k] = v

        # generate 2D detection
        with torch.no_grad():
            self.base_detector.set_detection_cfg(self.train_cfg.get('detection_proposal'))
            results = self.base_detector.simple_test_w_feat(detector_feat, img_metas)
            detections = self.process_2d_detections(results, img.device)

        if self.train_cfg.get('complement_2d_gt', -1) > 0:
            detections_gt = self.process_2d_gt(gt_bboxes, gt_labels, img.device)
            detections_gt = detections_gt + [img.new_zeros((0, 6))] * (num_views - self.num_views)
            detections = [self.complement_2d_gt(det, det_gt, thr=self.train_cfg.get('complement_2d_gt'))
                          for det, det_gt in zip(detections, detections_gt)]

        # calculate losses for 3d detector
        if not self.grad_all:
            feat_current = self.process_detector_feat(detector_feat_current)
            with torch.no_grad():
                feat_history = self.process_detector_feat(detector_feat_history)
            feat = [torch.cat([x1, x2]) for x1, x2 in zip(feat_current, feat_history)]
        else:
            feat = self.process_detector_feat(detector_feat)

        roi_losses = self.roi_head.forward_train(feat, img_metas, detections,                           # num_views
                                                 gt_bboxes, gt_labels, gt_bboxes_3d, gt_labels_3d,      # self.num_views
                                                 ori_gt_bboxes_3d, ori_gt_labels_3d,
                                                 attr_labels, None)
        losses.update(roi_losses)
        return losses

