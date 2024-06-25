# intensity_detection

An intensity-based defense method for detecting fake objects injected by laser-based attackers in LiDAR point clouds.

The code is developed based on [OpenPCDet](https://github.com/open-mmlab/OpenPCDet).

All these files must be located in OpenPCDet/tools after installing OpenPCDet via Docker.

Download the [KITTI Dataset](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d)


**Scenario Description**

- *Normal Scenario* : A situation where no attacks are performed (You can input the bin file as it is)

- *Attack Scenario 1* : A situation where there is only one object on the road

- *Attack Scenario 2* : A situation where there are multiple normal objects and one fake object is injected

**Explanation of Attack Code**

**(1) Normal Scenario, Attack Scenario 1**

- *save_box_fake.py* : Stage 1 (Threshold-based Classification)

Configure save_box_fake.py to receive the folder containing boxes classified as attack in stage2 as input.

- *pytorch_kitti/main.py* : Stage2 (DGCNN-based Classification)

**(2) Attack Scenario 2**

- *sn2_final.py* : Run sn2_1stage.py, sn2_2stage.py, and sn2_filter.py sequentially to measure the detection performance → Attack Scenario 2

**Code to create fake objects**

- *save_obj.py* : Cut and save the points within the bounding boxes that are within a specific distance(20m).

- *modify_intensity.py* : Unify the intensity of the saved objects to a single value or randomly change it within a specific range.

- *attack_class.py*: An optimized attack on a pillar-based object detection algorithm.
Modify OpenPCDet/pcdet/models/dense_heads/anchor_head_single.py and then execute.
Based on [Robust3DOD](https://github.com/Eaphan/Robust3DOD)

- *rotate.py* : By using object files (bin) as input, augmented data with random yaw direction rotation can be obtained.

