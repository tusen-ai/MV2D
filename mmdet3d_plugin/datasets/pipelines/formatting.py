import mmcv
import numpy as np
from mmcv.parallel import DataContainer as DC
from mmdet.datasets.pipelines import to_tensor
from mmdet3d.datasets import PIPELINES
from mmdet3d.datasets.pipelines.formating import DefaultFormatBundle3D, Collect3D
from mmdet.core.visualization.image import imshow_det_bboxes, imshow_gt_det_bboxes
from mmdet3d.core.visualizer import show_multi_modality_result
from mmdet3d.core.visualizer.image_vis import draw_lidar_bbox3d_on_img


@PIPELINES.register_module()
class DefaultFormatBundleMono3D(DefaultFormatBundle3D):

    def __call__(self, results):
        results = super(DefaultFormatBundleMono3D, self).__call__(results)
        for key in [
                'gt_bboxes_2d', 'gt_labels_2d', 'gt_bboxes_2d_to_3d',
        ]:
            if key not in results:
                continue
            if isinstance(results[key], list):
                results[key] = DC([to_tensor(res) for res in results[key]])
            else:
                results[key] = DC(to_tensor(results[key]))
        return results


@PIPELINES.register_module()
class CollectMono3D(Collect3D):
    def __init__(
        self,
        keys,
        meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img',
                   'depth2img', 'cam2img', 'pad_shape', 'scale_factor', 'flip',
                   'pcd_horizontal_flip', 'pcd_vertical_flip', 'box_mode_3d',
                   'box_type_3d', 'img_norm_cfg', 'pcd_trans', 'sample_idx',
                   'pcd_scale_factor', 'pcd_rotation', 'pcd_rotation_angle',
                   'pts_filename', 'transformation_3d_flow', 'trans_mat',
                   'affine_aug', 'intrinsics', 'extrinsics', 'timestamp'),
        debug=False,
        classes=('car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle', 'motorcycle',
                 'pedestrian', 'traffic_cone', 'barrier', 'ignore'),
    ):
        super(CollectMono3D, self).__init__(keys, meta_keys)
        self.debug = debug
        self.classes = classes

    @staticmethod
    def parse_img_metas(img_metas):
        if isinstance(img_metas, DC):
            img_metas = img_metas.data
        num_views = len(img_metas['img_shape'])
        img_metas_views = img_metas
        img_metas = []
        for j in range(num_views):
            img_meta = dict()
            for k, v in img_metas_views.items():
                if isinstance(v, list):
                    img_meta[k] = v[j]
                elif k == 'ori_shape':
                    img_meta[k] = v[:3]
                else:
                    img_meta[k] = v
            img_metas.append(img_meta)
        return img_metas

    @staticmethod
    def denormalize(img, img_norm_config):
        img = img.permute(1, 2, 0).numpy()
        img = mmcv.imdenormalize(img, img_norm_config['mean'], img_norm_config['std'], img_norm_config['to_rgb'])
        return img

    @staticmethod
    def get_box_params(bboxes, intrinsics, extrinsics, roi_size):
        import torch
        intrinsic_list = []
        extrinsic_list = []
        for img_id, (bbox, intrinsic, extrinsic) in enumerate(zip(bboxes, intrinsics, extrinsics)):
            # bbox: [n, (x, y, x, y)], rois_i: [n, c, h, w], intrinsic: [4, 4], extrinsic: [4, 4]
            intrinsic = torch.from_numpy(intrinsic).to(bbox.device).type(bbox.dtype)
            extrinsic = torch.from_numpy(extrinsic).to(bbox.device).type(bbox.dtype)
            intrinsic = intrinsic.repeat(bbox.shape[0], 1, 1)
            extrinsic = extrinsic.repeat(bbox.shape[0], 1, 1)
            wh_bbox = bbox[:, 2:4] - bbox[:, :2]
            wh_roi = wh_bbox.new_tensor(roi_size)
            scale = wh_roi[None] / wh_bbox
            intrinsic[:, :2, 2] = intrinsic[:, :2, 2] - bbox[:, :2] - 0.5 / scale
            intrinsic[:, :2] = intrinsic[:, :2] * scale[..., None]
            intrinsic_list.append(intrinsic)
            extrinsic_list.append(extrinsic)
        intrinsic_list = torch.cat(intrinsic_list, 0)
        extrinsic_list = torch.cat(extrinsic_list, 0)
        return intrinsic_list, extrinsic_list

    def __call__(self, results):
        results = super(CollectMono3D, self).__call__(results)
        if self.debug:
            vis_2d = True
            vis_3d = True
            vis_bbox = True
            img_metas = self.parse_img_metas(results['img_metas'])
            for img_id, (img, img_meta) in enumerate(zip(results['img'].data, img_metas)):
                img = self.denormalize(img, img_meta['img_norm_cfg'])
                img_3d, img_bbox = img.copy(), img.copy()
                file_name = 'debug/' + '/'.join(img_meta['filename'].split('/')[-2:])
                file_name_3d = file_name.replace('.jpg', '_gt3d.jpg')
                prefix_bbox = 'debug/bbox/' + '/'.join(img_meta['filename'].split('/')[-2:]).replace('.jpg', '')

                # problem img: CAM_BACK/n008-2018-09-18-14-35-12-0400__CAM_BACK__1537295917937558.jpg
                if vis_2d:
                    bboxes_2d = results['gt_bboxes_2d'].data[img_id].numpy()
                    labels_2d = results['gt_labels_2d'].data[img_id].numpy()
                    img = imshow_det_bboxes(
                        img,
                        bboxes_2d,
                        labels_2d,
                        class_names=self.classes,
                        bbox_color='green',
                        text_color='green',
                        show=False,
                        out_file=None)
                    bboxes_ignore = results['gt_bboxes_ignore'].data[img_id].numpy()
                    labels_ignore = np.zeros(len(bboxes_ignore), dtype=np.int32) + len(self.classes) - 1
                    img = imshow_det_bboxes(
                        img,
                        bboxes_ignore,
                        labels_ignore,
                        class_names=self.classes,
                        bbox_color='red',
                        text_color='red',
                        show=False,
                        out_file=None)
                    mmcv.imwrite(img, file_name)

                if vis_3d:
                    gt_ids = results['gt_bboxes_2d_to_3d'].data[img_id].unique()
                    gt_ids = gt_ids[gt_ids > -1].long()
                    bboxes_3d = results['gt_bboxes_3d'].data[gt_ids]
                    # lidar2img = img_meta['intrinsics'] @ img_meta['extrinsics'].T
                    lidar2img = img_meta['lidar2img']
                    img_3d = draw_lidar_bbox3d_on_img(bboxes_3d, img_3d, lidar2img, None)
                    mmcv.imwrite(img_3d, file_name_3d)

                if vis_bbox:
                    bboxes_2d = results['gt_bboxes_2d'].data[img_id]
                    gt_ids = results['gt_bboxes_2d_to_3d'].data[img_id].long()
                    bboxes_2d = bboxes_2d[gt_ids > -1]
                    gt_ids = gt_ids[gt_ids > -1]
                    bboxes_3d = results['gt_bboxes_3d'].data[gt_ids]

                    roi_size = (40, 40)
                    intrinsics, extrinsics = self.get_box_params(
                        [bboxes_2d.int().float()], [img_meta['intrinsics']], [img_meta['extrinsics']], roi_size)

                    for i in range(len(bboxes_2d)):
                        b2d = bboxes_2d[i]
                        b3d = bboxes_3d[i:i+1]
                        intrins = intrinsics[i]
                        extrins = extrinsics[i]

                        crop = b2d.int().numpy()
                        img_crop = img_bbox[crop[1]:crop[3], crop[0]:crop[2]].copy()
                        img_crop = mmcv.imresize(img_crop, roi_size)
                        lidar2img = (intrins @ extrins.T).numpy()
                        img_crop = draw_lidar_bbox3d_on_img(b3d, img_crop, lidar2img, None)
                        mmcv.imwrite(img_crop, prefix_bbox + '%03d.jpg' % i)

            import ipdb; ipdb.set_trace()

        return results

