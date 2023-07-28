_base_ = [
    './maskrcnn_r50.py'
]

model = dict(
    base_detector=dict(
        init_cfg=dict(type='Pretrained', checkpoint='./weights/mask_rcnn_r101_fpn_1x_nuim_20201024_134803-65c7623a.pth'),
        backbone=dict(
            depth=101,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            frozen_stages=1,
            norm_cfg=dict(type='BN', requires_grad=False),
            norm_eval=True,
            style='pytorch',
            init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet101')),
    )
)
