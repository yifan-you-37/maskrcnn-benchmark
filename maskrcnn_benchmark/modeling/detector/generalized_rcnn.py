# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn

from maskrcnn_benchmark.structures.image_list import to_image_list

from ..backbone import build_backbone
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads


class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()
        self.onnx_wrapper = ONNXWrapper(cfg)
        self.backbone = build_backbone(cfg)
        self.rpn = build_rpn(cfg, self.backbone.out_channels)
        self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)

        self.cfg = cfg
        self.device = torch.device(cfg.MODEL.DEVICE)

    def forward(self, images, image_sizes, targets=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        # images = to_image_list(images)
        images = to_image_list(images, self.cfg.DATALOADER.SIZE_DIVISIBILITY)
        images = images.to(self.device)

        #convert image_sizes to 2d tensor
        image_sizes = torch.IntTensor(image_sizes).to(self.device)

        # print('backbone tensor input shape:', images.image_sizes)
        input_names = ['input_image', 'input_image_size']
        output_names = ['output']
        dummy_input = (images.tensors, image_sizes)
        # torch.onnx.export(self.onnx_wrapper, dummy_input, "rcnn.onnx", verbose=True, input_names=input_names, output_names=output_names)
        return self.onnx_wrapper(images.tensors, image_sizes)
        
class ONNXWrapper(nn.Module):
    def __init__(self, cfg):
        super(ONNXWrapper, self).__init__()

        self.backbone = build_backbone(cfg)
        self.rpn = build_rpn(cfg, self.backbone.out_channels)
        self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)

    def forward(self, images, image_sizes, targets=None):
        # features = self.backbone(images.tensors)
        # proposals, proposal_losses = self.rpn(images, image_sizes, features, targets)
        
        features = self.backbone(images)
        proposals, proposal_losses = self.rpn(images, image_sizes, features, targets)

        if self.roi_heads:
            x, result, detector_losses = self.roi_heads(features, proposals, targets)
        else:
            # RPN-only models don't have roi_heads
            x = features
            result = proposals
            detector_losses = {}
        
        # if self.training:
        #     losses = {}
        #     losses.update(detector_losses)
        #     losses.update(proposal_losses)
        #     return losses

        # result = [tmp.bbox for tmp in result]
        print('done')
        return result
