import numpy as np
import torch
import torch.nn as nn

from .anchor_head_template import AnchorHeadTemplate


class AnchorHeadSingle(AnchorHeadTemplate):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range,
                 predict_boxes_when_training=True, **kwargs):
        super().__init__(
            model_cfg=model_cfg, num_class=num_class, class_names=class_names, grid_size=grid_size, point_cloud_range=point_cloud_range,
            predict_boxes_when_training=predict_boxes_when_training
        )

        self.num_anchors_per_location = sum(self.num_anchors_per_location)

        self.conv_cls = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.num_class,
            kernel_size=1
        )
        self.conv_box = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.box_coder.code_size,
            kernel_size=1
        )

        if self.model_cfg.get('USE_DIRECTION_CLASSIFIER', None) is not None:
            self.conv_dir_cls = nn.Conv2d(
                input_channels,
                self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS,
                kernel_size=1
            )
        else:
            self.conv_dir_cls = None
        self.init_weights()

    def init_weights(self):
        pi = 0.01
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)

    def forward(self, data_dict):
        spatial_features_2d = data_dict['spatial_features_2d']

        cls_preds = self.conv_cls(spatial_features_2d)
        box_preds = self.conv_box(spatial_features_2d)

        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C], Information on the predicted classes for all objects that may exist within the image
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]

        #print(f"cls_preds.shape:{cls_preds.shape}")
        #print(f"cls_preds:{cls_preds}")

        self.forward_ret_dict['cls_preds'] = cls_preds
        self.forward_ret_dict['box_preds'] = box_preds

        if self.conv_dir_cls is not None:
            dir_cls_preds = self.conv_dir_cls(spatial_features_2d)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            self.forward_ret_dict['dir_cls_preds'] = dir_cls_preds

            #print(f"dir_cls_preds:{dir_cls_preds.shape}") ##
        else:
            dir_cls_preds = None

        if self.training:
            #print("**TRAIN MODE**")
            #print(f"data_dict :{list(data_dict.keys())}")

            #if 'gt_boxes' not in data_dict:
                # Dummy gt_boxes format: [x, y, z, dx, dy, dz, heading, class_index]
                #dummy_gt_boxes = torch.zeros((1, 1, 8), device='cuda')  # Batch size 1, number of objects 1, attributes 8
                # Set the class index to meaningful values or adjust appropriately according to the model
                #dummy_gt_boxes[:, :, 7] = -1  # For example, set the class index to -1
                #data_dict['gt_boxes'] = dummy_gt_boxes

            targets_dict = self.assign_targets(
                gt_boxes=data_dict['gt_boxes']
            )
            #print("gt_boxes:", gt_boxes)
            self.forward_ret_dict.update(targets_dict)

            ##

            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=data_dict['batch_size'],
                cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
            )
            data_dict['batch_cls_preds'] = batch_cls_preds
            data_dict['batch_box_preds'] = batch_box_preds
            data_dict['cls_preds_normalized'] = False

            ##

        if not self.training or self.predict_boxes_when_training:
        #if self.training or self.predict_boxes_when_training:
            #print("**TEST MODE**")
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=data_dict['batch_size'],
                cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
            )
            data_dict['batch_cls_preds'] = batch_cls_preds
            data_dict['batch_box_preds'] = batch_box_preds
            data_dict['cls_preds_normalized'] = False

            # Print the newly created values
            #print("batch_cls_preds max value:", batch_cls_preds.max().item())
            #print("batch_cls_preds min value:", batch_cls_preds.min().item())
            #print("batch_box_preds max value:", batch_box_preds.max().item())
            #print("batch_box_preds min value:", batch_box_preds.min().item())
            #if dir_cls_preds is not None:
                #print("dir_cls_preds max value:", dir_cls_preds.max().item())
                #print("dir_cls_preds min value:", dir_cls_preds.min().item())

            data_dict['batch_cls_preds'] = batch_cls_preds
            data_dict['batch_box_preds'] = batch_box_preds

        #else :
            #print("**TRAIN MODE**")

        return data_dict
