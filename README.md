# intensity_detection

An intensity-based defense method for detecting fake objects injected by laser-based attackers in LiDAR point clouds.


All these files must be located in OpenPCDet/tools after installing OpenPCDet via Docker.


**save_box_fake.py** : Stage 1 (Threshold-based Classification)

**pytorch_kitti/main.py** : Stage2 (DGCNN-based Classification)

**sn2_final.py** : Run sn2_1stage.py, sn2_2stage.py, and sn2_filter.py sequentially to measure the performance for Scenario 2 


The code is developed based on [OpenPCDet](https://github.com/open-mmlab/OpenPCDet).

This project is inspired by and builds upon the methods described in the [WangYueFt/dgcnn repository](https://github.com/WangYueFt/dgcnn).

