# intensity_detection

An intensity-based defense method for detecting fake objects injected by laser-based attackers in LiDAR point clouds.


All these files must be located in OpenPCDet/tools after installing OpenPCDet via Docker.

https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d


**save_box_fake.py** : Stage 1 (Threshold-based Classification)

**pytorch_kitti/main.py** : Stage2 (DGCNN-based Classification)

attack_class.py : pillar 기반 객체탐지 알고리즘에 대한 최적화 공격

rotate.py : By using object files (bin) as input, augmented data with random yaw direction rotation can be obtained.

**sn2_final.py** : Run sn2_1stage.py, sn2_2stage.py, and sn2_filter.py sequentially to measure the performance for Scenario 2


The code is developed based on [OpenPCDet](https://github.com/open-mmlab/OpenPCDet).

This project is inspired by and builds upon the methods described in the [WangYueFt/dgcnn repository](https://github.com/WangYueFt/dgcnn).

명령어 / 파일 설명

시나리오 1, 2 설명

주석 한글 -> 영어
