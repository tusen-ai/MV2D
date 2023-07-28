# MV2D
This repo is the official PyTorch implementation for paper:   
[Object as Query: Lifting any 2D Object Detector to 3D Detection](https://arxiv.org/abs/2301.02364). Accepted by ICCV 2023.

We design Multi-View 2D Objects guided 3D Object Detector (MV2D), which can lift any 2D object detector to multi-view 3D object detection. Since 2D detections can provide valuable priors for object existence, MV2D exploits 2D detectors to generate object queries conditioned on the rich image semantics. These dynamically generated queries help MV2D to recall objects in the field of view and show a strong capability of localizing 3D objects. For the generated queries, we design a sparse cross attention module to force them to focus on the features of specific objects, which suppresses interference from noises. 

## Preparation
This implementation is built upon [PETR](https://github.com/megvii-research/PETR/tree/main), and can be constructed as the [install.md](https://github.com/megvii-research/PETR/blob/main/install.md).

* Environments  
  Linux, Python == 3.8.10, CUDA == 11.3, pytorch == 1.11.0, mmdet == 2.25.1, mmdet3d == 0.28.0   

* Detection Data   
Follow the mmdet3d to process the nuScenes dataset (https://github.com/open-mmlab/mmdetection3d/blob/master/docs/en/data_preparation.md).

* Pretrained weights   
We use nuImages pretrained weights from [mmdetection3d](https://github.com/open-mmlab/mmdetection3d/tree/main/configs/nuimages). Download the pretrained weights and put them into `weights/` directory. 

* After preparation, you will be able to see the following directory structure:  
  ```
  MV2D
  ├── mmdetection3d
  ├── configs
  ├── mmdet3d_plugin
  ├── tools
  ├── data
  │   ├── nuscenes
  │     ├── ...
  ├── weights
  ├── README.md
  ```

## Train & Inference
<!-- ```bash
git clone https://github.com/tusen-ai/MV2D.git
``` -->
```bash
cd MV2D
```
You can train the model following:
```bash
bash tools/dist_train.sh configs/mv2d/exp/mv2d_r50_frcnn_two_frames_1408x512_ep24.py 8 
```
You can evaluate the model following:
```bash
bash tools/dist_test.sh configs/mv2d/exp/mv2d_r50_frcnn_two_frames_1408x512_ep24.py work_dirs/mv2d_r50_frcnn_two_frames_1408x512_ep24/latest.pth 8 --eval bbox
```

## Main Results
|                                             config                                              |  mAP  |  NDS  |  checkpoint  |
|:-----------------------------------------------------------------------------------------------:|:-----:|:-----:|:------------:|
|    [MV2D-T_R50_1408x512_ep72](./configs/mv2d/exp/mv2d_r50_frcnn_two_frames_1408x512_ep72.py)    | 0.453 | 0.543 | [download](https://drive.google.com/file/d/10zwn2UWb2IzIWqJK1a2y466ZSWoLkD-e/view?usp=drive_link) |  


## Acknowledgement
Many thanks to the authors of [mmdetection3d](https://github.com/open-mmlab/mmdetection3d) and [petr](https://github.com/megvii-research/PETR/tree/main).

# Citation
If you find this repo useful for your research, please cite
```
@article{wang2023object,
  title={Object as query: Equipping any 2d object detector with 3d detection ability},
  author={Wang, Zitian and Huang, Zehao and Fu, Jiahui and Wang, Naiyan and Liu, Si},
  journal={arXiv preprint arXiv:2301.02364},
  year={2023}
}
```
# Contact
For questions about our paper or code, please contact **Zitian Wang**(wangzt.kghl@gmail.com).