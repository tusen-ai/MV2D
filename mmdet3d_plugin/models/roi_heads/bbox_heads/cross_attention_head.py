# ------------------------------------------------------------------------
# Modified from PETR (https://github.com/megvii-research/PETR)
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------

import torch
import torch.nn as nn
import math
import copy
from mmdet.models.utils.builder import TRANSFORMER
from mmcv.runner import BaseModule, auto_fp16, force_fp32
from mmcv.cnn import Conv2d, Linear, build_activation_layer, bias_init_with_prob
from mmdet.core import build_bbox_coder, build_assigner, build_sampler, multi_apply, reduce_mean
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.utils import build_transformer
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet3d_plugin.core.bbox.util import normalize_bbox
from mmdet3d_plugin.models.utils.pe import pos2posemb3d
from mmdet3d_plugin.models.utils import PETRTransformer


@TRANSFORMER.register_module()
class MV2DTransformer(PETRTransformer):
    def forward(self, x, mask, query_embed, pos_embed,
                attn_mask=None, cross_attn_mask=None, **kwargs):
        # x: [bs, n, c, h, w], mask: [bs, n, h, w], query_embed: [bs, n_query, c]
        bs, n, c, h, w = x.shape
        memory = x.permute(1, 3, 4, 0, 2).reshape(n * h * w, bs, c) # [bs, n, c, h, w] -> [n*h*w, bs, c]
        mask = mask.view(bs, n * h * w)  # [bs, n, h, w] -> [bs, n*h*w]
        query_embed = query_embed.permute(1, 0, 2)
        pos_embed = pos_embed.permute(1, 3, 4, 0, 2).reshape(n * h * w, bs, c) # [bs, n, c, h, w] -> [n*h*w, bs, c]
        target = torch.zeros_like(query_embed)
        if cross_attn_mask is not None:
            cross_attn_mask = cross_attn_mask.flatten(1, 3)   # [n_query, n, h, w] -> [n_query, n * h * w]

        # out_dec: [num_layers, num_query, bs, dim]
        out_dec = self.decoder(
            query=target,
            key=memory,
            value=memory,
            key_pos=pos_embed,
            query_pos=query_embed,
            key_padding_mask=mask,
            attn_masks=[attn_mask, cross_attn_mask],
            **kwargs,
            )
        out_dec = out_dec.transpose(1, 2)
        memory = memory.reshape(n, h, w, bs, c).permute(3, 0, 4, 1, 2)
        return out_dec, memory


class RegLayer(nn.Module):
    def __init__(self,  embed_dims=256,
                        shared_reg_fcs=2,
                        group_reg_dims=(2, 1, 3, 2, 2),  # xy, z, size, rot, velo
                        act_layer=nn.ReLU,
                        drop=0.0):
        super().__init__()

        reg_branch = []
        for _ in range(shared_reg_fcs):
            reg_branch.append(Linear(embed_dims, embed_dims))
            reg_branch.append(act_layer())
            reg_branch.append(nn.Dropout(drop))
        self.reg_branch = nn.Sequential(*reg_branch)

        self.task_heads = nn.ModuleList()
        for reg_dim in group_reg_dims:
            task_head = nn.Sequential(
                Linear(embed_dims, embed_dims),
                act_layer(),
                Linear(embed_dims, reg_dim)
            )
            self.task_heads.append(task_head)

    def forward(self, x):
        reg_feat = self.reg_branch(x)
        outs = []
        for task_head in self.task_heads:
            out = task_head(reg_feat.clone())
            outs.append(out)
        outs = torch.cat(outs, -1)
        return outs


@HEADS.register_module()
class CrossAttentionBoxHead(BaseModule):
    def __init__(self, num_classes, transformer, pc_range, embed_dims=256, num_reg_fcs=2,
                 group_reg_dims=(2, 2, 1, 1, 2, 2), use_reg_layer=False, pre_embed=False,
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 loss_bbox=dict(
                     type='SmoothL1Loss', beta=1.0, loss_weight=1.0),
                 bbox_coder=dict(
                     type='NMSFreeCoder',
                     # type='NMSFreeClsCoder',
                     post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
                     pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                     max_num=100,
                     num_classes=10),
                 sync_cls_avg_factor=False,
                 train_cfg=None,
                 test_cfg=None,
                 **kwargs
                 ):
        super(CrossAttentionBoxHead, self).__init__()

        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.transformer = build_transformer(transformer)

        self.pc_range = pc_range
        self.embed_dims = embed_dims
        self.pre_embed = pre_embed
        if not self.pre_embed:
            self.query_embedding = nn.Sequential(
                nn.Linear(self.embed_dims*3//2, self.embed_dims),
                nn.ReLU(),
                nn.Linear(self.embed_dims, self.embed_dims),
            )

        self.num_pred = transformer['decoder']['num_layers']
        self.num_classes = num_classes
        self.cls_out_channels = num_classes
        cls_branch = []
        for _ in range(num_reg_fcs):
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
        fc_cls = nn.Sequential(*cls_branch)
        self.cls_branches = nn.ModuleList(
            [copy.deepcopy(fc_cls) for _ in range(self.num_pred)])
        if not use_reg_layer:
            reg_branch = []
            for _ in range(num_reg_fcs):
                reg_branch.append(Linear(self.embed_dims, self.embed_dims))
                reg_branch.append(nn.ReLU())
            reg_branch.append(Linear(self.embed_dims, sum(group_reg_dims)))
            reg_branch = nn.Sequential(*reg_branch)
        else:
            reg_branch = RegLayer(self.embed_dims, num_reg_fcs, group_reg_dims)
        self.reg_branches = nn.ModuleList(
            [copy.deepcopy(reg_branch) for _ in range(self.num_pred)])

        # follow PETR
        self.bbox_coder = build_bbox_coder(bbox_coder)
        if train_cfg is not None:
            self.assigner = build_assigner(train_cfg.get('assigner'))
            self.sampler = build_sampler(train_cfg.get('sampler_cfg', dict(type='PseudoSampler')))
        self.bg_cls_weight = 0
        class_weight = loss_cls.get('class_weight', None)
        if class_weight is not None:
            assert isinstance(class_weight, float), 'Expected ' \
                'class_weight to have type float. Found ' \
                f'{type(class_weight)}.'
            # NOTE following the official DETR rep0, bg_cls_weight means
            # relative classification weight of the no-object class.
            bg_cls_weight = loss_cls.get('bg_cls_weight', class_weight)
            assert isinstance(bg_cls_weight, float), 'Expected ' \
                'bg_cls_weight to have type float. Found ' \
                f'{type(bg_cls_weight)}.'
            class_weight = torch.ones(num_classes + 1) * class_weight
            # set background class as the last indice
            class_weight[num_classes] = bg_cls_weight
            loss_cls.update({'class_weight': class_weight})
            if 'bg_cls_weight' in loss_cls:
                loss_cls.pop('bg_cls_weight')
            if 'class_weight' in loss_cls:
                loss_cls.pop('class_weight')
            self.bg_cls_weight = bg_cls_weight
        self.sync_cls_avg_factor = sync_cls_avg_factor
        if 'code_size' in kwargs:
            self.code_size = kwargs['code_size']
        else:
            self.code_size = 10
        if 'code_weights' in kwargs:
            self.code_weights = kwargs['code_weights']
        else:
            self.code_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]
        self.code_weights = self.code_weights[:self.code_size]
        self.code_weights = nn.Parameter(torch.tensor(
            self.code_weights, requires_grad=False), requires_grad=False)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.fp16_enabled = False

    def init_weights(self):
        """Initialize the transformer weights."""
        self.transformer.init_weights()
        bias_init = bias_init_with_prob(0.01)
        for m in self.cls_branches:
            nn.init.constant_(m[-1].bias, bias_init)

    def position_embedding(self, query_pos):
        return self.query_embedding(pos2posemb3d(query_pos, num_pos_feats=self.embed_dims//2))

    def forward(self, reference_points, x, masks, pos_embed,
                attn_mask=None, cross_attn_mask=None, force_fp32=False, query_embeds=None,
                return_query_feats=False, **kwargs):
        if not self.pre_embed:
            query_embeds = self.position_embedding(reference_points)

        if force_fp32:
            with torch.autocast('cuda', enabled=False):
                outs_dec, _ = self.transformer(x.float(), masks, query_embeds.float(), pos_embed.float(),
                                               attn_mask=attn_mask, cross_attn_mask=cross_attn_mask, **kwargs)
        else:
            outs_dec, _ = self.transformer(x, masks, query_embeds, pos_embed,
                                           attn_mask=attn_mask, cross_attn_mask=cross_attn_mask, **kwargs)

        outputs_classes = []
        outputs_coords = []
        for lvl in range(outs_dec.shape[0]):
            reference = inverse_sigmoid(reference_points.clone())
            assert reference.shape[-1] == 3
            outputs_class = self.cls_branches[lvl](outs_dec[lvl])
            tmp = self.reg_branches[lvl](outs_dec[lvl])

            tmp[..., 0:2] += reference[..., 0:2]
            tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
            tmp[..., 4:5] += reference[..., 2:3]
            tmp[..., 4:5] = tmp[..., 4:5].sigmoid()

            outputs_coord = tmp
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        all_cls_scores = torch.stack(outputs_classes)
        all_bbox_preds = torch.stack(outputs_coords)

        all_bbox_preds[..., 0:1] = (all_bbox_preds[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0])
        all_bbox_preds[..., 1:2] = (all_bbox_preds[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1])
        all_bbox_preds[..., 4:5] = (all_bbox_preds[..., 4:5] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2])

        if return_query_feats:
            return all_cls_scores, all_bbox_preds, outs_dec[-1]
        return all_cls_scores, all_bbox_preds

    def _get_target_single(self,
                           cls_score,
                           bbox_pred,
                           gt_labels,
                           gt_bboxes,
                           gt_bboxes_ignore=None):
        """"Compute regression and classification targets for one image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_query, 4].
            gt_bboxes (Tensor): Ground truth bboxes for one image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth class indices for one image
                with shape (num_gts, ).
            gt_bboxes_ignore (Tensor, optional): Bounding boxes
                which can be ignored. Default None.
        Returns:
            tuple[Tensor]: a tuple containing the following for one image.
                - labels (Tensor): Labels of each image.
                - label_weights (Tensor]): Label weights of each image.
                - bbox_targets (Tensor): BBox targets of each image.
                - bbox_weights (Tensor): BBox weights of each image.
                - pos_inds (Tensor): Sampled positive indices for each image.
                - neg_inds (Tensor): Sampled negative indices for each image.
        """

        num_bboxes = bbox_pred.size(0)
        # assigner and sampler
        assign_result = self.assigner.assign(bbox_pred, cls_score, gt_bboxes,
                                             gt_labels, gt_bboxes_ignore)
        sampling_result = self.sampler.sample(assign_result, bbox_pred,
                                              gt_bboxes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label targets
        labels = gt_bboxes.new_full((num_bboxes,),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        code_size = gt_bboxes.size(1)
        bbox_targets = torch.zeros_like(bbox_pred)[..., :code_size]
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0
        # DETR
        if sampling_result.pos_gt_bboxes.shape[1] == 4:
            bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes.reshape(sampling_result.pos_gt_bboxes.shape[0],
                                                                           self.code_size - 1)
        else:
            bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes

        return (labels, label_weights, bbox_targets, bbox_weights,
                pos_inds, neg_inds)

    def get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_bboxes_ignore_list=None):
        """"Compute regression and classification targets for a batch image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image with shape [num_query,
                cls_out_channels].
            bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx, cy, w, h) and shape [num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            tuple: a tuple containing the following targets.
                - labels_list (list[Tensor]): Labels for all images.
                - label_weights_list (list[Tensor]): Label weights for all \
                    images.
                - bbox_targets_list (list[Tensor]): BBox targets for all \
                    images.
                - bbox_weights_list (list[Tensor]): BBox weights for all \
                    images.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
        """
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        (labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, pos_inds_list, neg_inds_list) = multi_apply(
            self._get_target_single, cls_scores_list, bbox_preds_list,
            gt_labels_list, gt_bboxes_list, gt_bboxes_ignore_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, num_total_pos, num_total_neg)

    @force_fp32(apply_to=('preds_dicts'))
    def get_bboxes(self, preds_dicts, img_metas, rescale=False):
        """Generate bboxes from bbox head predictions.
        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.
        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        preds_dicts = self.bbox_coder.decode(preds_dicts)
        num_samples = len(preds_dicts)

        ret_list = []
        for i in range(num_samples):
            preds = preds_dicts[i]
            bboxes = preds['bboxes']
            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
            bboxes = img_metas[i]['box_type_3d'](bboxes, bboxes.size(-1))
            scores = preds['scores']
            labels = preds['labels']
            ret_list.append([bboxes, scores, labels])
        return ret_list

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'gt_bboxes_list', 'gt_labels_list'))
    def loss_single(self,
                    cls_scores,
                    bbox_preds,
                    gt_bboxes_list,
                    gt_labels_list,
                    cls_reg_targets=None,
                    gt_bboxes_ignore_list=None):

        num_imgs = len(cls_scores)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        if cls_reg_targets is None:
            cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                               gt_bboxes_list, gt_labels_list,
                                               gt_bboxes_ignore_list)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        cls_scores, bbox_preds = torch.cat(cls_scores), torch.cat(bbox_preds)
        # classification loss
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(cls_scores.new_tensor([cls_avg_factor]))

        cls_avg_factor = max(cls_avg_factor, 1)

        # loss_cls = self.loss_cls(cls_scores, labels, label_weights, avg_factor=cls_avg_factor)
        if len(cls_scores) == 0:
            loss_cls = cls_scores.sum() * cls_avg_factor
        else:
            loss_cls = self.loss_cls(cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # regression L1 loss
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        normalized_bbox_targets = normalize_bbox(bbox_targets, None)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.code_weights

        loss_bbox = self.loss_bbox(
            bbox_preds[isnotnan, :10], normalized_bbox_targets[isnotnan, :10], bbox_weights[isnotnan, :10],
            avg_factor=num_total_pos)

        loss_cls = torch.nan_to_num(loss_cls)
        loss_bbox = torch.nan_to_num(loss_bbox)
        return loss_cls, loss_bbox, cls_reg_targets

    def loss(self,
             gt_bboxes_3d_list,
             gt_labels_3d_list,
             preds_dicts,
             cls_reg_targets=None,
             gt_bboxes_ignore=None):
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'

        cls_scores = preds_dicts['cls_scores']
        bbox_preds = preds_dicts['bbox_preds']

        device = gt_labels_3d_list[0].device
        gt_bboxes_3d_list = [torch.cat(
            (gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]),
            dim=1).to(device) for gt_bboxes in gt_bboxes_3d_list]

        losses_cls, losses_bbox, cls_reg_targets = self.loss_single(
            cls_scores, bbox_preds, gt_bboxes_3d_list, gt_labels_3d_list,
            cls_reg_targets=cls_reg_targets, gt_bboxes_ignore_list=gt_bboxes_ignore)

        loss_dict = dict()
        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls
        loss_dict['loss_bbox'] = losses_bbox
        # loss_dict['cls_reg_targets'] = cls_reg_targets
        return loss_dict

    def loss_batch(self, *args):
        loss_dict = dict()
        losses_cls, losses_bbox, cls_reg_targets = multi_apply(
            self.loss,
            *args,
        )
        loss_dict['loss_cls'] = losses_cls
        loss_dict['loss_bbox'] = losses_bbox
        return loss_dict

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'known_bboxs'))
    def dn_loss_single(self,
                       cls_scores,
                       bbox_preds,
                       known_bboxs,
                       known_labels,
                       num_total_pos,
                       pc_range,
                       split,
                       neg_bbox_loss=False):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.
        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        # import ipdb; ipdb.set_trace()
        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 3.14159 / 6 * split * split * split  ### positive rate
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        bbox_weights = torch.ones_like(bbox_preds)
        label_weights = torch.ones_like(known_labels)
        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(
            cls_scores, known_labels.long(), label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        if not neg_bbox_loss:
            neg_samples = (known_labels == self.num_classes)
            known_bboxs[neg_samples] = 0

        # regression L1 loss
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        normalized_bbox_targets = normalize_bbox(known_bboxs, pc_range)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.code_weights
        bbox_weights[:, 6:8] = 0  ###dn alaways reduce the mAOE, which is useless when training for a long time.
        loss_bbox = self.loss_bbox(
            bbox_preds[isnotnan, :10], normalized_bbox_targets[isnotnan, :10], bbox_weights[isnotnan, :10],
            avg_factor=num_total_pos)

        loss_cls = torch.nan_to_num(loss_cls)
        loss_bbox = torch.nan_to_num(loss_bbox)

        return loss_cls, loss_bbox