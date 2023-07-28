_base_ = [
    '../../_base_/schedules/mmdet_schedule_1x.py', '../../_base_/default_runtime.py'
]

class_names = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
]

plugin_dir = 'mmdet3d_plugin/'
dataset_type = 'CustomNuScenesDataset'
data_root = 'data/nuscenes/'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)
ida_aug_conf = {
    "resize_lim": (0.8, 1.0),
    "final_dim": (512, 1408),
    "bot_pct_lim": (0.0, 0.0),
    "rot_lim": (0.0, 0.0),
    "H": 900,
    "W": 1600,
    "rand_flip": True,
}

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='LoadAnnotationsMono3D', with_bbox_3d=True, with_label_3d=True, with_bbox_2d=True, with_attr_label=False),
    dict(type='ObjectRangeFilterMono', point_cloud_range=point_cloud_range, with_bbox_2d=True),
    dict(type='ObjectNameFilterMono', classes=class_names, with_bbox_2d=True),
    dict(type='ResizeCropFlipImageMono', data_aug_conf=ida_aug_conf, with_bbox_2d=True, training=True),
    dict(type='GlobalRotScaleTransImage',
         rot_range=[-0.3925, 0.3925],
         translation_std=[0, 0, 0],
         scale_ratio_range=[0.95, 1.05],
         reverse_angle=True,
         training=True
         ),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundleMono3D', class_names=class_names),
    dict(type='CollectMono3D',
         debug=False,
         keys=['gt_bboxes_3d', 'gt_labels_3d', 'gt_bboxes_2d', 'gt_labels_2d', 'gt_bboxes_2d_to_3d', 'gt_bboxes_ignore',
               'img'])
]
test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='LoadAnnotationsMono3D', with_bbox_3d=False, with_label_3d=False, with_bbox_2d=False, with_attr_label=False),
    dict(type='ResizeCropFlipImageMono', data_aug_conf=ida_aug_conf, with_bbox_2d=False, training=False),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundleMono3D',
                class_names=class_names,
                with_label=False),
            dict(type='CollectMono3D', debug=False,
                 keys=['img'])
        ])
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=5,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_train.pkl',
        ann_file_2d=data_root + 'nuscenes_infos_train_mono3d.coco.json',
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        box_type_3d='LiDAR'
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_val.pkl',
        ann_file_2d=data_root + 'nuscenes_infos_val_mono3d.coco.json',
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR',
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_val.pkl',
        ann_file_2d=data_root + 'nuscenes_infos_val_mono3d.coco.json',
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR',
    ))
