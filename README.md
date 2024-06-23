# intensity_detection

An intensity-based defense method for detecting fake objects injected by laser-based attackers in LiDAR point clouds.


All these files must be located in OpenPCDet/tools after installing OpenPCDet via Docker.

KITTI Dataset 사용
(https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d)


**save_box_fake.py** : Stage 1 (Threshold-based Classification)

**pytorch_kitti/main.py** : Stage2 (DGCNN-based Classification)

rotate.py : By using object files (bin) as input, augmented data with random yaw direction rotation can be obtained.

**sn2_final.py** : Run sn2_1stage.py, sn2_2stage.py, and sn2_filter.py sequentially to measure the performance for Scenario 2



The code is developed based on [OpenPCDet](https://github.com/open-mmlab/OpenPCDet).

This project is inspired by and builds upon the methods described in the [WangYueFt/dgcnn repository](https://github.com/WangYueFt/dgcnn).

명령어 / 파일 설명

Scenario 1: A situation where there is only one object on the road

Scenario 2: A situation where there are multiple normal objects and one fake object is injected

주석 한글 -> 영어

Code to create fake objects

attack_class.py: An optimized attack on a pillar-based object detection algorithm.
Modify OpenPCDet/pcdet/models/dense_heads/anchor_head_single.py and then execute.
Based on [Robust3DOD](https://github.com/Eaphan/Robust3DOD)

https://github.com/Eaphan/Robust3DOD

