import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import cv2
import mmcv
from mmcv.runner import auto_fp16, force_fp32


class BoxCorrelation(nn.Module):
    def __init__(self, sample_size=4, num_depth=8, depth_start=0.5, depth_end=70,
                 correlation_mode=None, LID=True, expand_stride=0, force_cpu=False):
        super(BoxCorrelation, self).__init__()
        self.sample_size = sample_size
        self.num_depth = num_depth
        self.depth_start = depth_start
        self.depth_end = depth_end
        self.correlation_mode = correlation_mode
        self.LID = LID
        self.expand_stride = expand_stride
        self.force_cpu = force_cpu

    @torch.no_grad()
    def epipolar_in_group(self, rois, image_shape, trans_mats, num_proposals_per_img,
                          feat_coords, stride, feat_in_groups, box2group, num_groups_per_img):
        num_views = trans_mats.size(0)

        # feat_coords: [h, w, 2], feat_in_groups: list(num_views)->[num_groups_img, h, w],
        #   box2group: list(num_views)->[num_rois_img]
        feat_in_groups_cat = torch.cat(feat_in_groups, dim=0)       # [num_groups, h, w]
        num_points_per_img = [x.sum() for x in feat_in_groups]
        num_points_per_group = feat_in_groups_cat.sum([-2, -1]).tolist()
        points_in_groups = feat_coords[None].expand(feat_in_groups_cat.size(0), *feat_coords.shape)[feat_in_groups_cat]
        points_in_groups_imgid = torch.cat([torch.full([n], i, dtype=points_in_groups.dtype, device=rois.device) for i, n in enumerate(num_points_per_img)])
        points_in_groups = torch.cat([points_in_groups_imgid[:, None], points_in_groups], dim=-1)

        # points: [num_points, 3->(view_id, x, y)]
        points = points_in_groups
        num_points = points.size(0)

        # transformed_points: [num_points, num_views, num_depths, 2], valid_mask: [num_points, num_views, num_depths]
        transformed_points, t_points_mask = self.gen_epipolar_in_each_view(points, image_shape, trans_mats)

        # rois_pad: [num_views, max_rois, 4]
        rois_pad = pad_sequence(rois.split(num_proposals_per_img, 0), batch_first=True, padding_value=-100)[..., 1:]

        transformed_points = transformed_points[:, :, None]     # [num_points, num_views, 1, num_depths, 2]
        t_points_mask = t_points_mask[:, :, None]         # [num_points, num_views, 1, num_depths]
        rois_pad = rois_pad[None, :, :, None]      # [1, num_views, max_rois, 1, 4]
        # t_points_in_rois: [num_points, num_views, max_rois, num_depths]
        t_points_in_rois = ((transformed_points > rois_pad[..., 0:2]) & (transformed_points < rois_pad[..., 2:4])).all(-1)
        t_points_in_rois = t_points_in_rois & t_points_mask
        t_points_in_rois = t_points_in_rois.any(-1)     # [num_points, num_views, max_rois]

        # convert to absolute box id and group id
        box2group_pad = pad_sequence(box2group, batch_first=True, padding_value=-1).clone()     # [num_views, max_rois]
        box2group_mask = box2group_pad > -1

        start = 0
        for i, n in enumerate(num_groups_per_img):
            box2group_pad[i] += start
            start += n

        box2group_pad[~box2group_mask] = -1

        # matched_group_ids: [num_points, num_views, max_rois]
        matched_group_ids = box2group_pad[None].expand(num_points, *box2group_pad.shape).clone()
        matched_group_ids[~t_points_in_rois] = -1

        # t_points_valid: [num_points]
        t_points_valid = t_points_in_rois.any(-1).any(-1)
        group_valid = pad_sequence(t_points_valid.split(num_points_per_group, 0), batch_first=True)
        group_valid = group_valid.any(-1)

        group_matched_ids = []
        points_slice = torch.tensor(num_points_per_group, dtype=torch.int64, device=rois.device)
        points_slice = torch.cat([points_slice.new_tensor([0]), points_slice.cumsum(0)])
        for group_id in group_valid.nonzero()[:, 0]:
            matched_ids = matched_group_ids[points_slice[group_id]:points_slice[group_id+1]].unique()
            matched_ids = matched_ids[matched_ids > -1]
            group_matched_ids.append(matched_ids)

        if group_matched_ids:
            group_matched_ids = pad_sequence(group_matched_ids, batch_first=True, padding_value=-1)
            max_matched = group_matched_ids.size(1)
            group_corr = torch.full([feat_in_groups_cat.size(0), max_matched], -1, dtype=torch.int64, device=rois.device)
            group_corr[group_valid] = group_matched_ids
        else:
            group_corr = torch.zeros([feat_in_groups_cat.size(0), 0], dtype=torch.int64, device=rois.device)

        return group_corr

    @torch.no_grad()
    def gen_box_correlation(self, rois, num_proposals_per_img, img_metas, feat, stride):
        image_shape = img_metas[0]['pad_shape']
        num_views = img_metas[0]['num_views']
        batch_size = len(img_metas) // num_views
        assert batch_size == 1, 'only support batch_size 1 now'

        _, _, h, w = feat.shape
        ys = (torch.arange(h, dtype=feat.dtype, device=feat.device) + 0.5) * stride - 0.5
        xs = (torch.arange(w, dtype=feat.dtype, device=feat.device) + 0.5) * stride - 0.5
        feat_coords = torch.stack(torch.meshgrid(ys, xs)[::-1], dim=-1)     # [h, w, 2]

        rois_b = rois.clone()

        # generate in-roi mask
        coords = feat_coords.clone()[None, :, :]    # [1, h, w, 2(xy)]
        box_bxyxy = rois_b[:, None, None]            # [num_rois_b, 1, 1, 5(bxyxy)]
        in_bbox = (coords[..., 0:2] + 0.5 * stride + self.expand_stride * stride >= box_bxyxy[..., 1:3]) & \
                  (coords[..., 0:2] - 0.5 * stride - self.expand_stride * stride <= box_bxyxy[..., 3:5])  # [num_rois_b, h, w, 2]
        in_bbox = in_bbox.all(-1)  # [num_rois_b, h, w]
        feat_in_rois = torch.zeros((rois_b.size(0), num_views, h, w), dtype=torch.bool, device=rois_b.device)
        feat_in_rois[torch.arange(rois_b.size(0)), rois_b[:, 0].long()] = in_bbox   # [num_rois_b, num_imgs, h, w]

        # lidar2img: [num_views, 4, 4]
        lidar2img = torch.stack([torch.from_numpy(x['lidar2img']).to(rois.device) for x in img_metas],
                                dim=0).double()
        img2lidar = torch.inverse(lidar2img)
        # trans_mats: [num_views, num_views, 4, 4]
        trans_mats = torch.matmul(lidar2img[None], img2lidar[:, None])
        # find matched rois in other views
        matched_roi_ids_epipolar, valid_mask_epipolar = \
            self.epipolar_in_box(rois_b, image_shape, trans_mats, num_proposals_per_img, img_metas)

        # matched rois in all the views
        matched_roi_ids = torch.arange(len(rois_b), dtype=torch.int64, device=rois.device)[:, None]
        valid_mask = torch.ones_like(matched_roi_ids).bool()
        matched_roi_ids = torch.cat([matched_roi_ids, matched_roi_ids_epipolar], dim=1)
        valid_mask = torch.cat([valid_mask, valid_mask_epipolar], dim=1)

        num_valid_per_roi = valid_mask.sum(-1).tolist()
        matched_roi_ids = pad_sequence(matched_roi_ids[valid_mask].split(num_valid_per_roi, 0), batch_first=True)  # [num_rois_b, max_valid]
        valid_mask = pad_sequence(valid_mask[valid_mask].split(num_valid_per_roi, 0), batch_first=True)  # [num_rois_b, max_valid]

        # generate in-relevant-rois mask
        feat_in_corr_rois = torch.zeros_like(feat_in_rois)

        # save gpu memory
        if self.force_cpu:
            feat_in_rois = feat_in_rois.cpu()
            matched_roi_ids = matched_roi_ids.cpu()
            valid_mask = valid_mask.cpu()
            feat_in_corr_rois = feat_in_corr_rois.cpu()

        valid_num = valid_mask.sum(-1)
        valid_sec = [0, 16, 32, 64, 1e3]

        for i in range(len(valid_sec) - 1):
            valid_i = (valid_sec[i] < valid_num) & (valid_num <= valid_sec[i+1])
            valid_max = valid_num[valid_i].max() if valid_i.sum() > 0 else 0
            if valid_max > 0:
                feat_in_corr_rois_i = feat_in_rois[matched_roi_ids[valid_i, :valid_max]]   # [num_valid, max_valid, num_imgs, h, w]
                feat_in_corr_rois_i = feat_in_corr_rois_i & valid_mask[valid_i, :valid_max, None, None, None]
                feat_in_corr_rois_i = feat_in_corr_rois_i.any(1)        # [num_valid, num_imgs, h, w]
                feat_in_corr_rois[valid_i] = feat_in_corr_rois_i

        if self.force_cpu:
            feat_in_corr_rois = feat_in_corr_rois.to(rois.device)

        return feat_in_corr_rois

    @torch.no_grad()
    def gen_box_roi_correlation(self, rois, num_proposals_per_img, img_metas,):
        if rois.numel() == 0:
            corr = rois.new_zeros((0, 0), dtype=torch.int64)
            mask = rois.new_zeros((0, 0), dtype=torch.bool)
            return corr, mask

        image_shape = img_metas[0]['pad_shape']

        # lidar2img: [num_views, 4, 4]
        lidar2img = torch.stack([torch.from_numpy(x['lidar2img']).to(rois.device) for x in img_metas],
                                dim=0).double()
        img2lidar = torch.inverse(lidar2img)
        # trans_mats: [num_views, num_views, 4, 4]
        trans_mats = torch.matmul(lidar2img[None], img2lidar[:, None])
        # find matched rois in other views
        matched_roi_ids_epipolar, valid_mask_epipolar = \
            self.epipolar_in_box(rois, image_shape, trans_mats, num_proposals_per_img, img_metas)

        # matched rois in all the views
        matched_roi_ids = torch.arange(len(rois), dtype=torch.int64, device=rois.device)[:, None]
        valid_mask = torch.ones_like(matched_roi_ids).bool()
        matched_roi_ids = torch.cat([matched_roi_ids, matched_roi_ids_epipolar], dim=1)
        valid_mask = torch.cat([valid_mask, valid_mask_epipolar], dim=1)

        num_valid_per_roi = valid_mask.sum(-1).tolist()
        matched_roi_ids = pad_sequence(matched_roi_ids[valid_mask].split(num_valid_per_roi, 0), batch_first=True)  # [num_rois_b, max_valid]
        valid_mask = pad_sequence(valid_mask[valid_mask].split(num_valid_per_roi, 0), batch_first=True)  # [num_rois_b, max_valid]

        return matched_roi_ids, valid_mask

    @torch.no_grad()
    def gen_sample_points_in_rois(self, rois):
        # rois: [num_rois, 5->(view_id, xmin, ymin, xmax, ymax)]
        xs = ys = torch.linspace(0, 1, self.sample_size, device=rois.device)
        grid_y, grid_x = torch.meshgrid(ys, xs)
        coords_roi = torch.stack([grid_x, grid_y], dim=-1)  # [size, size, 2]
        # convert coords from roi frame to image frame
        wh_bbox = rois[:, 3:5] - rois[:, 1:3]  # [num_rois, 2]
        coords_img = rois[:, None, None, 1:3] + wh_bbox[:, None, None] * coords_roi[None]
        num_rois = coords_img.size(0)
        sample_points = coords_img.reshape(num_rois, self.sample_size * self.sample_size, 2)    # [n_rois, n_points, 2]
        batch_ids = rois[:, None, 0:1].expand_as(sample_points[..., 0:1])
        # roi_points: # [n_rois, n_points, 3->(view_id, x, y)]
        roi_points = torch.cat([batch_ids, sample_points], dim=-1)
        return roi_points

    @torch.no_grad()
    def gen_epipolar_in_each_view(self, points, image_shape, trans_mats, ):
        # points: [num_points, 3->(view_id, x, y)]
        # depth_range: [num_points, 2->(min_depth, max_depth)]
        # num_depths: int
        # image_shape: (input_img_h, input_img_w)
        # trans_mats: [num_views, num_views, 4, 4], transformation matrices for each pair of view
        num_depths = self.num_depth
        num_points = points.size(0)

        if self.LID:
            index = torch.arange(start=0, end=num_depths, step=1, device=points.device).float()
            index_1 = index + 1
            bin_size = (self.depth_end - self.depth_start) / (num_depths * (1 + num_depths))
            depth_interv = self.depth_start + bin_size * index * index_1
        else:
            depth_interv = torch.linspace(self.depth_start, self.depth_end, num_depths, device=points.device)
        # depth_values: [num_points, num_depths]
        depth_values = depth_interv[None].expand(num_points, num_depths)
        # points_2d: [num_points, num_depths, 3]
        points_2d = torch.cat([points[:, None, 1:3].expand(num_points, num_depths, 2), depth_values[..., None]], dim=-1).to(trans_mats.dtype)
        # points_cam_hom: [num_points, num_depths, 4]
        points_cam_hom = torch.cat([points_2d[..., :2] * points_2d[..., 2:3], points_2d[..., 2:3], points_2d.new_ones((num_points, num_depths, 1))], dim=-1)
        points_cam_hom = points_cam_hom

        # convert points_cam to other view
        view_ids = points[:, 0].long()
        # trans_mats: [num_points, num_views, 4, 4]
        # import ipdb; ipdb.set_trace()
        trans_mats = trans_mats[view_ids]
        transformed_points_cam = torch.matmul(trans_mats[:, :, None], points_cam_hom[:, None, ..., None])[..., :3, 0]
        # transformed_points: [num_points, num_views, num_depths, 2]
        transformed_points = transformed_points_cam[..., :2] / transformed_points_cam[..., 2:3].clamp_min(1e-2)
        # valid_mask: [num_points, num_views, num_depths]
        valid_mask = torch.ones_like(transformed_points[..., 0], dtype=torch.bool)
        # only keep the points with (depth > 0)
        valid_mask[transformed_points_cam[..., 2] < self.depth_start] = 0

        points_in_img_x = (0 <= transformed_points[..., 0]) & (transformed_points[..., 0] <= image_shape[1] - 1)
        points_in_img_y = (0 <= transformed_points[..., 1]) & (transformed_points[..., 1] <= image_shape[0] - 1)
        points_in_img = points_in_img_x & points_in_img_y
        valid_mask = valid_mask & points_in_img

        # exclude points in the original view
        valid_mask[torch.arange(num_points), view_ids] = 0

        return transformed_points.float(), valid_mask

    @torch.no_grad()
    def epipolar_in_box(self, rois, image_shape, trans_mats, num_proposals_per_img, img_metas, **kwargs):
        if rois.numel() == 0:
            corr = rois.new_zeros((0, 0), dtype=torch.int64)
            mask = rois.new_zeros((0, 0), dtype=torch.bool)
            return corr, mask

        num_views = trans_mats.size(0)
        # points: [num_rois, num_sample_points, 3->(view_id, x, y)]
        points = self.gen_sample_points_in_rois(rois)
        num_rois, np = points.shape[:2]

        # points: [num_points, 3->(view_id, x, y)]
        points = points.reshape(num_rois * np, 3)
        num_points = points.size(0)

        # transformed_points: [num_points, num_views, num_depths, 2], valid_mask: [num_points, num_views, num_depths]
        transformed_points, valid_mask = self.gen_epipolar_in_each_view(points, image_shape, trans_mats)

        num_depth = self.num_depth
        # transformed_points: [num_rois, num_views, num_sample_points, 2], valid_mask: [num_rois, num_views, num_sample_points]
        transformed_points = transformed_points.view(num_rois, np, num_views, num_depth, 2)\
            .permute(0, 2, 1, 3, 4).reshape(num_rois, num_views, np * num_depth, 2)
        valid_mask = valid_mask.view(num_rois, np, num_views, num_depth).permute(0, 2, 1, 3)\
            .reshape(num_rois, num_views, np * num_depth)
        num_sample_points = valid_mask.size(2)

        rois_per_view = rois.split(num_proposals_per_img, 0)
        rois_ids = torch.arange(num_rois, dtype=torch.int64, device=rois.device)
        rois_ids = rois_ids.split(num_proposals_per_img, 0)

        # rois_pad: [num_views, max_rois, 5]
        rois_pad = pad_sequence(rois_per_view, batch_first=True)
        # rois_ids_pad: [num_views, max_rois], num_rois_per_view: [num_views, max_rois]
        rois_ids_pad = pad_sequence(rois_ids, batch_first=True, padding_value=-1)

        # epipolar_in_rois: [num_rois, num_views, max_rois]
        transformed_points = transformed_points[:, :, None]     # [num_rois, num_views, 1, num_sample_points, 2]
        valid_mask = valid_mask[:, :, None]                     # [num_rois, num_views, 1, num_sample_points]
        rois_pad = rois_pad[None, :, :, None]                   # [1, num_views, max_rois, 1, 5]
        points_in_rois = (rois_pad[..., 1] <= transformed_points[..., 0]) & (transformed_points[..., 0] <= rois_pad[..., 3]) \
                         & (rois_pad[..., 2] <= transformed_points[..., 1]) & (transformed_points[..., 1] <= rois_pad[..., 4])
        points_in_rois = points_in_rois & valid_mask            # [num_rois, num_views, max_rois, num_sample_points]
        epipolar_in_rois = points_in_rois.any(-1)
        epipolar_in_rois = epipolar_in_rois & (rois_ids_pad > -1)[None]

        if self.correlation_mode == 'all_matched':
            epipolar_in_view = epipolar_in_rois.any(-1)                     # [num_rois, num_views]

            t_points = transformed_points[:, :, 0][epipolar_in_view]        # [num_valid, num_sample_points, 2]
            t_points_mask = valid_mask[:, :, 0][epipolar_in_view]           # [num_valid, num_sample_points]
            t_points_xymax = t_points.clone()
            t_points_xymax[~t_points_mask] = -1e4
            t_points_xymin = t_points.clone()
            t_points_xymin[~t_points_mask] = 1e4
            t_rois_xymax = t_points_xymax.max(1)[0]
            t_rois_xymin = t_points_xymin.min(1)[0]
            t_rois = torch.cat([t_rois_xymin, t_rois_xymax], dim=1)         # [num_valid, 4]

            nonzero_ids = epipolar_in_view.nonzero()
            view_id = nonzero_ids[:, 1]

            rois_view = rois_pad[0, :, :, 0][view_id]                       # [num_valid, max_rois, 5]
            rois_ids_view = rois_ids_pad[view_id]                           # [num_valid, max_rois]
            rois_mask_view = rois_ids_view > -1                             # [num_valid, max_rois]
            if rois_view.numel() == 0:
                iou = torch.zeros_like(rois_view[..., 0])
            else:
                iou = self.box_iou(t_rois[:, None], rois_view[..., 1:])[:, 0]   # [num_valid, max_rois]
            iou[~rois_mask_view] = 0

            all_roi_id = rois_ids_view  # [num_valid, max_rois]
            all_mask = iou > 0
            num_corr_view_per_roi = epipolar_in_view.sum(-1).tolist()

            corr = pad_sequence(all_roi_id.split(num_corr_view_per_roi, 0), batch_first=True)      # [num_rois, max_valid, max_rois]
            mask = pad_sequence(all_mask.split(num_corr_view_per_roi, 0), batch_first=True)        # [num_rois, max_valid, max_rois]
            corr = corr.flatten(-2, -1)
            mask = mask.flatten(-2, -1)
            return corr, mask
        elif self.correlation_mode.startswith('topk_matched'):
            info = self.correlation_mode.split(':')
            topk = int(info[1])
            iou_thr = float(info[2])
            ratio = float(info[3])

            epipolar_in_view = epipolar_in_rois.any(-1)                     # [num_rois, num_views]

            t_points = transformed_points[:, :, 0][epipolar_in_view]        # [num_valid, num_sample_points, 2]
            t_points_mask = valid_mask[:, :, 0][epipolar_in_view]           # [num_valid, num_sample_points]
            t_points_xymax = t_points.clone()
            t_points_xymax[~t_points_mask] = -1e4
            t_points_xymin = t_points.clone()
            t_points_xymin[~t_points_mask] = 1e4
            t_rois_xymax = t_points_xymax.max(1)[0]
            t_rois_xymin = t_points_xymin.min(1)[0]
            t_rois = torch.cat([t_rois_xymin, t_rois_xymax], dim=1)         # [num_valid, 4]

            nonzero_ids = epipolar_in_view.nonzero()
            roi_id = nonzero_ids[:, 0]
            view_id = nonzero_ids[:, 1]

            rois_view = rois_pad[0, :, :, 0][view_id]                       # [num_valid, max_rois, 5]
            rois_ids_view = rois_ids_pad[view_id]                           # [num_valid, max_rois]
            rois_mask_view = rois_ids_view > -1                             # [num_valid, max_rois]
            if rois_view.numel() == 0:
                iou = torch.zeros_like(rois_view[..., 0])
            else:
                iou = self.box_iou(t_rois[:, None], rois_view[..., 1:])[:, 0]   # [num_valid, max_rois]
            iou[~rois_mask_view] = 0

            topk_iou_index = iou.argsort(-1, descending=True)[:, :topk]
            topk_roi_id = torch.gather(rois_ids_view, -1, topk_iou_index)   # [num_valid, topk]
            topk_iou = torch.gather(iou, -1, topk_iou_index)
            topk_iou_max = topk_iou.max(-1, keepdim=True)[0]
            topk_mask = ((topk_iou > ratio * topk_iou_max) | (topk_iou > iou_thr)) & (topk_iou > 0)   # [num_valid, topk]

            num_corrs_per_roi = epipolar_in_view.sum(-1).tolist()
            corr = pad_sequence(topk_roi_id.split(num_corrs_per_roi, 0), batch_first=True)      # [num_rois, max_valid, topk]
            mask = pad_sequence(topk_mask.split(num_corrs_per_roi, 0), batch_first=True)        # [num_rois, max_valid, topk]
            corr = corr.flatten(-2, -1)
            mask = mask.flatten(-2, -1)

            return corr, mask

    @staticmethod
    def box_iou(rois_a, rois_b, eps=1e-4):
        rois_a = rois_a[..., None, :]                # [*, n, 1, 4]
        rois_b = rois_b[..., None, :, :]             # [*, 1, m, 4]
        xy_start = torch.maximum(rois_a[..., 0:2], rois_b[..., 0:2])
        xy_end = torch.minimum(rois_a[..., 2:4], rois_b[..., 2:4])
        wh = torch.maximum(xy_end - xy_start, rois_a.new_tensor(0))     # [*, n, m, 2]
        intersect = wh.prod(-1)                                         # [*, n, m]
        wh_a = rois_a[..., 2:4] - rois_a[..., 0:2]      # [*, m, 1, 2]
        wh_b = rois_b[..., 2:4] - rois_b[..., 0:2]      # [*, 1, n, 2]
        area_a = wh_a.prod(-1)
        area_b = wh_b.prod(-1)
        union = area_a + area_b - intersect
        iou = intersect / (union + eps)
        return iou

