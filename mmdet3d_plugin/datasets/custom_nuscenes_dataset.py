# ------------------------------------------------------------------------
# Modified from PETR (https://github.com/megvii-research/PETR)
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
import json
import tempfile
from os import path as osp
import numpy as np
import pyquaternion
from nuscenes.utils.data_classes import Box as NuScenesBox
from nuscenes.eval.common.data_classes import EvalBoxes
import mmcv
from mmdet.datasets.api_wrappers import COCO
from mmdet.datasets import DATASETS
from mmdet3d.datasets import NuScenesMonoDataset, NuScenesDataset
import os
import copy
from mmdet3d.core.bbox import CameraInstance3DBoxes, LiDARInstance3DBoxes, get_box_type, Box3DMode


@DATASETS.register_module()
class CustomNuScenesDataset(NuScenesDataset):
    r"""NuScenesMono Dataset.
    This dataset add camera intrinsics and extrinsics and 2d bbox to the results.
    """
    def __init__(self, ann_file_2d, load_separate=False, **kwargs):
        self.load_separate = load_separate
        self.ann_file_2d = ann_file_2d
        super(CustomNuScenesDataset, self).__init__(**kwargs)
        self.load_annotations_2d(ann_file_2d)

    def __len__(self):
        return super(CustomNuScenesDataset, self).__len__()

    def load_annotations(self, ann_file):
        data = mmcv.load(ann_file, file_format='pkl')
        data_infos_ori = data_infos = list(sorted(data['infos'], key=lambda e: e['timestamp']))
        data_infos = data_infos[::self.load_interval]
        self.metadata = data['metadata']
        self.version = self.metadata['version']

        if self.load_separate:
            data_infos_path = []
            out_dir = self.ann_file.split('.')[0]
            for i in mmcv.track_iter_progress(range(len(data_infos_ori))):
                out_file = osp.join(out_dir, '%07d.pkl' % i)
                data_infos_path.append(out_file)
                if not osp.exists(out_file):
                    mmcv.dump(data_infos_ori[i], out_file, file_format='pkl')
            data_infos_path = data_infos_path[::self.load_interval]
            return data_infos_path

        return data_infos

    def pre_pipeline(self, results):
        results['img_fields'] = []
        results['bbox3d_fields'] = []
        results['bbox2d_fields'] = []
        results['pts_mask_fields'] = []
        results['pts_seg_fields'] = []
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []
        results['box_type_3d'] = self.box_type_3d
        results['box_mode_3d'] = self.box_mode_3d

    def load_annotations_2d(self, ann_file):
        self.coco = COCO(ann_file)
        self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.impath_to_imgid = {}
        self.imgid_to_dataid = {}
        data_infos = []
        total_ann_ids = []
        for i in self.coco.get_img_ids():
            info = self.coco.load_imgs([i])[0]
            info['filename'] = info['file_name']
            self.impath_to_imgid['./data/nuscenes/' + info['file_name']] = i
            self.imgid_to_dataid[i] = len(data_infos)
            data_infos.append(info)
            ann_ids = self.coco.get_ann_ids(img_ids=[i])
            total_ann_ids.extend(ann_ids)
        assert len(set(total_ann_ids)) == len(
            total_ann_ids), f"Annotation ids in '{ann_file}' are not unique!"
        self.data_infos_2d = data_infos

    def impath_to_ann2d(self, impath):
        img_id = self.impath_to_imgid[impath]
        data_id = self.imgid_to_dataid[img_id]
        ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
        ann_info = self.coco.load_anns(ann_ids)
        return self.get_ann_info_2d(self.data_infos_2d[data_id], ann_info)

    def get_data_info(self, index):
        """Get data info according to the given index.
        Args:
            index (int): Index of the sample data to get.
        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations \
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        if not self.load_separate:
            info = self.data_infos[index]
        else:
            info = mmcv.load(self.data_infos[index], file_format='pkl')
        # standard protocal modified from SECOND.Pytorch
        input_dict = dict(
            sample_idx=info['token'],
            pts_filename=info['lidar_path'],
            sweeps=info['sweeps'],
            timestamp=info['timestamp'] / 1e6,
        )

        image_paths = []
        lidar2img_rts = []
        intrinsics = []
        extrinsics = []
        img_timestamp = []
        for cam_type, cam_info in info['cams'].items():
            img_timestamp.append(cam_info['timestamp'] / 1e6)
            image_paths.append(cam_info['data_path'])
            # obtain lidar to image transformation matrix
            lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
            lidar2cam_t = cam_info[
                              'sensor2lidar_translation'] @ lidar2cam_r.T
            lidar2cam_rt = np.eye(4)
            lidar2cam_rt[:3, :3] = lidar2cam_r.T
            lidar2cam_rt[3, :3] = -lidar2cam_t
            intrinsic = cam_info['cam_intrinsic']
            viewpad = np.eye(4)
            viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
            lidar2img_rt = (viewpad @ lidar2cam_rt.T)
            intrinsics.append(viewpad)
            extrinsics.append(
                lidar2cam_rt)  ###The extrinsics mean the tranformation from lidar to camera. If anyone want to use the extrinsics as sensor to lidar, please use np.linalg.inv(lidar2cam_rt.T) and modify the ResizeCropFlipImage and LoadMultiViewImageFromMultiSweepsFiles.
            lidar2img_rts.append(lidar2img_rt)

        input_dict.update(
            dict(
                img_timestamp=img_timestamp,
                img_filename=image_paths,
                lidar2img=lidar2img_rts,
                intrinsics=intrinsics,
                extrinsics=extrinsics
            ))

        input_dict['img_info'] = info
        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos

            gt_bboxes_3d = annos['gt_bboxes_3d']  # 3d bboxes
            gt_labels_3d = annos['gt_labels_3d']
            gt_bboxes_2d = []  # per-view 2d bboxes
            gt_bboxes_ignore = []  # per-view 2d bboxes
            gt_bboxes_2d_to_3d = []  # mapping from per-view 2d bboxes to 3d bboxes
            gt_labels_2d = []  # mapping from per-view 2d bboxes to 3d bboxes

            for cam_i in range(len(image_paths)):
                ann_2d = self.impath_to_ann2d(image_paths[cam_i])
                labels_2d = ann_2d['labels']
                bboxes_2d = ann_2d['bboxes_2d']
                bboxes_ignore = ann_2d['gt_bboxes_ignore']
                bboxes_cam = ann_2d['bboxes_cam']
                lidar2cam = extrinsics[cam_i].T

                centers_lidar = gt_bboxes_3d.gravity_center.numpy()
                centers_lidar_hom = np.concatenate([centers_lidar, np.ones((len(centers_lidar), 1))], axis=1)
                centers_cam = (centers_lidar_hom @ lidar2cam.T)[:, :3]
                match = self.center_match(bboxes_cam, centers_cam)
                assert (labels_2d[match > -1] == gt_labels_3d[match[match > -1]]).all()

                gt_bboxes_2d.append(bboxes_2d)
                gt_bboxes_2d_to_3d.append(match)
                gt_labels_2d.append(labels_2d)
                gt_bboxes_ignore.append(bboxes_ignore)

            annos['gt_bboxes_2d'] = gt_bboxes_2d
            annos['gt_labels_2d'] = gt_labels_2d
            annos['gt_bboxes_2d_to_3d'] = gt_bboxes_2d_to_3d
            annos['gt_bboxes_ignore'] = gt_bboxes_ignore
        return input_dict

    def center_match(self, bboxes_a, bboxes_b):
        cts_a, cts_b = bboxes_a[:, :3], bboxes_b[:, :3]
        if len(cts_a) == 0:
            return np.zeros(len(cts_a), dtype=np.int32) - 1
        if len(cts_b) == 0:
            return np.zeros(len(cts_a), dtype=np.int32) - 1
        dist = np.abs(cts_a[:, None] - cts_b[None]).sum(-1)
        match = dist.argmin(1)
        match[dist.min(1) > 1e-3] = -1
        return match

    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: Annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`):
                    3D ground truth bboxes
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - gt_names (list[str]): Class names of ground truths.
        """
        if not self.load_separate:
            info = self.data_infos[index]
        else:
            info = mmcv.load(self.data_infos[index], file_format='pkl')
        # filter out bbox containing no points
        if self.use_valid_flag:
            mask = info['valid_flag']
        else:
            mask = info['num_lidar_pts'] > 0
        gt_bboxes_3d = info['gt_boxes'][mask]
        gt_names_3d = info['gt_names'][mask]
        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d)

        if self.with_velocity:
            gt_velocity = info['gt_velocity'][mask]
            nan_mask = np.isnan(gt_velocity[:, 0])
            gt_velocity[nan_mask] = [0.0, 0.0]
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)

        # the nuscenes box center is [0.5, 0.5, 0.5], we change it to be
        # the same as KITTI (0.5, 0.5, 0)
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d)
        return anns_results

    def get_ann_info_2d(self, img_info_2d, ann_info_2d):
        """Parse bbox annotation.

        Args:
            img_info (list[dict]): Image info.
            ann_info (list[dict]): Annotation info of an image.

        Returns:
            dict: A dict containing the following keys: bboxes, labels,
                gt_bboxes_3d, gt_labels_3d, attr_labels, centers2d,
                depths, bboxes_ignore, masks, seg_map
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_bboxes_cam3d = []
        for i, ann in enumerate(ann_info_2d):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info_2d['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info_2d['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                bbox_cam3d = np.array(ann['bbox_cam3d']).reshape(1, -1)
                bbox_cam3d = np.concatenate([bbox_cam3d], axis=-1)
                gt_bboxes_cam3d.append(bbox_cam3d.squeeze())

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_cam3d:
            gt_bboxes_cam3d = np.array(gt_bboxes_cam3d, dtype=np.float32)
        else:
            gt_bboxes_cam3d = np.zeros((0, 6), dtype=np.float32)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        ann = dict(
            bboxes_cam=gt_bboxes_cam3d,
            bboxes_2d=gt_bboxes,
            gt_bboxes_ignore=gt_bboxes_ignore,
            labels=gt_labels, )
        return ann

    def format_results(self, results, jsonfile_prefix=None):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: Returns (result_files, tmp_dir), where `result_files` is a
                dict containing the json filepaths, `tmp_dir` is the temporal
                directory created for saving json files when
                `jsonfile_prefix` is not specified.
        """
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None

        # currently the output prediction results could be in two formats
        # 1. list of dict('boxes_3d': ..., 'scores_3d': ..., 'labels_3d': ...)
        # 2. list of dict('pts_bbox' or 'img_bbox':
        #     dict('boxes_3d': ..., 'scores_3d': ..., 'labels_3d': ...))
        # this is a workaround to enable evaluation of both formats on nuScenes
        # refer to https://github.com/open-mmlab/mmdetection3d/issues/449
        if not ('pts_bbox' in results[0] or 'img_bbox' in results[0]):
            result_files = self._format_bbox(results, jsonfile_prefix)
        else:
            # should take the inner dict out of 'pts_bbox' or 'img_bbox' dict
            result_files = dict()
            for name in results[0]:
                if name in ['pts_bbox', 'img_bbox']:
                    print(f'\nFormating bboxes of {name}')
                    results_ = [out[name] for out in results]
                    tmp_file_ = osp.join(jsonfile_prefix, name)
                    result_files.update(
                        {name: self._format_bbox(results_, tmp_file_)})
        return result_files, tmp_dir

    def _evaluate_single(self,
                         result_path,
                         logger=None,
                         metric='bbox',
                         result_name='pts_bbox',
                         ):
        """Evaluation for a single model in nuScenes protocol.

        Args:
            result_path (str): Path of the result file.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Default: None.
            metric (str, optional): Metric name used for evaluation.
                Default: 'bbox'.
            result_name (str, optional): Result name in the metric prefix.
                Default: 'pts_bbox'.

        Returns:
            dict: Dictionary of evaluation details.
        """
        from nuscenes import NuScenes
        from nuscenes.eval.detection.evaluate import NuScenesEval

        output_dir = osp.join(*osp.split(result_path)[:-1])
        nusc = NuScenes(
            version=self.version, dataroot=self.data_root, verbose=False)
        eval_set_map = {
            'v1.0-mini': 'mini_val',
            'v1.0-trainval': 'val',
        }
        nusc_eval = NuScenesEval(
            nusc,
            config=self.eval_detection_configs,
            result_path=result_path,
            eval_set=eval_set_map[self.version],
            output_dir=output_dir,
            verbose=False)

        nusc_eval.main(render_curves=False)

        # record metrics
        metrics = mmcv.load(osp.join(output_dir, 'metrics_summary.json'))
        detail = dict()
        metric_prefix = f'{result_name}_NuScenes'
        for name in self.CLASSES:
            for k, v in metrics['label_aps'][name].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}_AP_dist_{}'.format(metric_prefix, name, k)] = val
            for k, v in metrics['label_tp_errors'][name].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}_{}'.format(metric_prefix, name, k)] = val
            for k, v in metrics['tp_errors'].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}'.format(metric_prefix,
                                      self.ErrNameMapping[k])] = val

        detail['{}/NDS'.format(metric_prefix)] = metrics['nd_score']
        detail['{}/mAP'.format(metric_prefix)] = metrics['mean_ap']
        return detail

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 result_names=['pts_bbox'],
                 show=False,
                 out_dir=None,
                 pipeline=None,):

        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)

        if isinstance(result_files, dict):
            results_dict = dict()
            for name in result_names:
                print('Evaluating bboxes of {}'.format(name))
                ret_dict = self._evaluate_single(result_files[name])
            results_dict.update(ret_dict)
        elif isinstance(result_files, str):
            results_dict = self._evaluate_single(result_files)

        if tmp_dir is not None and not isinstance(tmp_dir, str):
            tmp_dir.cleanup()

        if show or out_dir:
            self.show(results, out_dir, show=show, pipeline=pipeline)
        return results_dict
