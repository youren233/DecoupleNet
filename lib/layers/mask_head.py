import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F
from lib.utils import comm
from lib.layers.wrappers import Conv2d, ConvTranspose2d, cat
from lib.layers.batch_norm import get_norm

def mask_rcnn_inference(pred_mask_logits, bo_mask_logits, bound_logits, bo_bound_logits, pred_instances):
    """
    Convert pred_mask_logits to estimated foreground probability masks while also
    extracting only the masks for the predicted classes in pred_instances. For each
    predicted box, the mask of the same class is attached to the instance by adding a
    new "pred_masks" field to pred_instances.

    Args:
        pred_mask_logits (Tensor): A tensor of shape (B, C, Hmask, Wmask) or (B, 1, Hmask, Wmask)
            for class-specific or class-agnostic, where B is the total number of predicted masks
            in all images, C is the number of foreground classes, and Hmask, Wmask are the height
            and width of the mask predictions. The values are logits.
        pred_instances (list[Instances]): A list of N Instances, where N is the number of images
            in the batch. Each Instances must have field "pred_classes".

    Returns:
        None. pred_instances will contain an extra "pred_masks" field storing a mask of size (Hmask,
            Wmask) for predicted class. Note that the masks are returned as a soft (non-quantized)
            masks the resolution predicted by the network; post-processing steps, such as resizing
            the predicted masks to the original image resolution and/or binarizing them, is left
            to the caller.
    """
    cls_agnostic_mask = pred_mask_logits.size(1) == 1
    #pred_mask_logits = pred_mask_logits[:,0:1]
    if cls_agnostic_mask:
        mask_probs_pred = pred_mask_logits.sigmoid()
        bound_probs_pred = bound_logits.sigmoid()
        bo_mask_probs_pred = bo_mask_logits.sigmoid()
        bo_bound_probs_pred = bo_bound_logits.sigmoid()
    else:
        # Select masks corresponding to the predicted classes
        num_masks = pred_mask_logits.shape[0]
        class_pred = cat([i.pred_classes for i in pred_instances])
        indices = torch.arange(num_masks, device=class_pred.device)
        mask_probs_pred = pred_mask_logits[indices, class_pred][:, None].sigmoid()
    # mask_probs_pred.shape: (B, 1, Hmask, Wmask)

    num_boxes_per_image = [len(i) for i in pred_instances]
    mask_probs_pred = mask_probs_pred.split(num_boxes_per_image, dim=0)
    bo_mask_probs_pred = bo_mask_probs_pred.split(num_boxes_per_image, dim=0)
    bo_bound_probs_pred = bo_bound_probs_pred.split(num_boxes_per_image, dim=0)
    bound_probs_pred = bound_probs_pred.split(num_boxes_per_image, dim=0)

    for prob, instances in zip(mask_probs_pred, pred_instances):
        instances.pred_masks = prob  # (1, Hmask, Wmask)
        instances.raw_masks = prob

    for bo_prob, instances in zip(bo_mask_probs_pred, pred_instances):
        instances.pred_masks_bo = bo_prob  # (1, Hmask, Wmask)

    for bo_bound_prob, instances in zip(bo_bound_probs_pred, pred_instances):
        instances.pred_bounds_bo = bo_bound_prob  # (1, Hmask, Wmask)

    for bound_prob, instances in zip(bound_probs_pred, pred_instances):
        instances.pred_bounds = bound_prob  # (1, Hmask, Wmask)


class MaskRCNNConvUpsampleHead(nn.Module):
    """
    A mask head with several conv layers, plus an upsample layer (with `ConvTranspose2d`).
    """

    def __init__(self, cfg, input_shape):
        """
        The following attributes are parsed from config:
            num_conv: the number of conv layers
            conv_dim: the dimension of the conv layers
            norm: normalization for the conv layers
        """
        super(MaskRCNNConvUpsampleHead, self).__init__()

        # fmt: off
        num_classes       = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        conv_dims         = cfg.MODEL.ROI_MASK_HEAD.CONV_DIM
        self.norm         = cfg.MODEL.ROI_MASK_HEAD.NORM
        num_conv          = cfg.MODEL.ROI_MASK_HEAD.NUM_CONV
        input_channels    = input_shape
        cls_agnostic_mask = cfg.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK
        # fmt: on

        self.conv_norm_relus = []

        for k in range(num_conv):
            conv = Conv2d(
                input_channels if k == 0 else conv_dims,
                conv_dims,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=not self.norm,
                norm=get_norm(self.norm, conv_dims),
                activation=F.relu,
            )
            self.add_module("mask_fcn{}".format(k + 1), conv)
            self.conv_norm_relus.append(conv)

        self.boundary_conv_norm_relus = []
        for k in range(num_conv):
            conv = Conv2d(
                input_channels if k == 0 else conv_dims,
                conv_dims,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=not self.norm,
                norm=get_norm(self.norm, conv_dims),
                activation=F.relu,
            )
            self.add_module("boundary_fcn{}".format(k + 1), conv)
            self.boundary_conv_norm_relus.append(conv)

        self.deconv = ConvTranspose2d(
            conv_dims if num_conv > 0 else input_channels,
            conv_dims,
            kernel_size=2,
            stride=2,
            padding=0,
        )

        self.boundary_deconv_bo = ConvTranspose2d(
            conv_dims if num_conv > 0 else input_channels,
            conv_dims,
            kernel_size=2,
            stride=2,
            padding=0,
        )

        self.boundary_deconv = ConvTranspose2d(
            conv_dims if num_conv > 0 else input_channels,
            conv_dims,
            kernel_size=2,
            stride=2,
            padding=0,
        )

        self.bo_deconv = ConvTranspose2d(
            conv_dims if num_conv > 0 else input_channels,
            conv_dims,
            kernel_size=2,
            stride=2,
            padding=0,
        )

        self.query_transform_bound_bo = Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.key_transform_bound_bo = Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.value_transform_bound_bo = Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.output_transform_bound_bo = Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0, bias=False)

        self.query_transform_bound = Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.key_transform_bound = Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.value_transform_bound = Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.output_transform_bound = Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0, bias=False)


        self.scale = 1.0 / (input_channels ** 0.5)
        self.blocker_bound_bo = nn.BatchNorm2d(input_channels, eps=1e-04) # should be zero initialized
        self.blocker_bound = nn.BatchNorm2d(input_channels, eps=1e-04) # should be zero initialized

        num_mask_classes = 1 if cls_agnostic_mask else num_classes
        self.predictor = Conv2d(conv_dims, num_mask_classes, kernel_size=1, stride=1, padding=0)
        self.predictor_bo = Conv2d(conv_dims, num_mask_classes, kernel_size=1, stride=1, padding=0)
        self.boundary_predictor_bo = Conv2d(conv_dims, num_mask_classes, kernel_size=1, stride=1, padding=0)
        self.boundary_predictor = Conv2d(conv_dims, num_mask_classes, kernel_size=1, stride=1, padding=0)


        for layer in self.conv_norm_relus + self.boundary_conv_norm_relus + [self.deconv, self.bo_deconv, self.boundary_deconv, self.boundary_deconv_bo, self.query_transform_bound_bo, self.key_transform_bound_bo, self.value_transform_bound_bo, self.output_transform_bound_bo, self.query_transform_bound, self.key_transform_bound, self.value_transform_bound, self.output_transform_bound]:
            weight_init.c2_msra_fill(layer)
        # use normal distribution initialization for mask prediction layer
        nn.init.normal_(self.predictor.weight, std=0.001)
        if self.predictor.bias is not None:
            nn.init.constant_(self.predictor.bias, 0)

        nn.init.normal_(self.predictor_bo.weight, std=0.001)
        if self.predictor_bo.bias is not None:
            nn.init.constant_(self.predictor_bo.bias, 0)

        nn.init.normal_(self.boundary_predictor_bo.weight, std=0.001)
        if self.boundary_predictor_bo.bias is not None:
            nn.init.constant_(self.boundary_predictor_bo.bias, 0)

        nn.init.normal_(self.boundary_predictor.weight, std=0.001)
        if self.boundary_predictor.bias is not None:
            nn.init.constant_(self.boundary_predictor.bias, 0)


    def forward(self, x):
        B, C, H, W = x.size()
        x_ori = x.clone()

        for cnt, layer in enumerate(self.boundary_conv_norm_relus):
            x = layer(x)

            if cnt == 1 and len(x) != 0:
                # x: B,C,H,W
                # x_query: B,C,HW
                #x_input = AddCoords()(x)
                x_query_bound_bo = self.query_transform_bound_bo(x).view(B, C, -1)
                # x_query: B,HW,C
                x_query_bound_bo = torch.transpose(x_query_bound_bo, 1, 2)
                # x_key: B,C,HW
                x_key_bound_bo = self.key_transform_bound_bo(x).view(B, C, -1)
                # x_value: B,C,HW
                x_value_bound_bo = self.value_transform_bound_bo(x).view(B, C, -1)
                # x_value: B,HW,C
                x_value_bound_bo = torch.transpose(x_value_bound_bo, 1, 2)
                # W = Q^T K: B,HW,HW
                x_w_bound_bo = torch.matmul(x_query_bound_bo, x_key_bound_bo) * self.scale
                x_w_bound_bo = F.softmax(x_w_bound_bo, dim=-1)
                # x_relation = WV: B,HW,C
                x_relation_bound_bo = torch.matmul(x_w_bound_bo, x_value_bound_bo)
                # x_relation = B,C,HW
                x_relation_bound_bo = torch.transpose(x_relation_bound_bo, 1, 2)
                # x_relation = B,C,H,W
                x_relation_bound_bo = x_relation_bound_bo.view(B,C,H,W)

                x_relation_bound_bo = self.output_transform_bound_bo(x_relation_bound_bo)
                x_relation_bound_bo = self.blocker_bound_bo(x_relation_bound_bo)

                x = x + x_relation_bound_bo

        x_bound_bo = x.clone()

        x_bo = x.clone()
        
        x = x_ori + x

        for cnt, layer in enumerate(self.conv_norm_relus):
            x = layer(x)
            if cnt == 1 and len(x) != 0:
                # x: B,C,H,W
                # x_query: B,C,HW
                #x_input = AddCoords()(x)
                x_query_bound = self.query_transform_bound(x).view(B, C, -1)
                # x_query: B,HW,C
                x_query_bound = torch.transpose(x_query_bound, 1, 2)
                # x_key: B,C,HW
                x_key_bound = self.key_transform_bound(x).view(B, C, -1)
                # x_value: B,C,HW
                x_value_bound = self.value_transform_bound(x).view(B, C, -1)
                # x_value: B,HW,C
                x_value_bound = torch.transpose(x_value_bound, 1, 2)
                # W = Q^T K: B,HW,HW
                x_w_bound = torch.matmul(x_query_bound, x_key_bound) * self.scale
                x_w_bound = F.softmax(x_w_bound, dim=-1)
                # x_relation = WV: B,HW,C
                x_relation_bound = torch.matmul(x_w_bound, x_value_bound)
                # x_relation = B,C,HW
                x_relation_bound = torch.transpose(x_relation_bound, 1, 2)
                # x_relation = B,C,H,W
                x_relation_bound = x_relation_bound.view(B,C,H,W)

                x_relation_bound = self.output_transform_bound(x_relation_bound)
                x_relation_bound = self.blocker_bound(x_relation_bound)

                x = x + x_relation_bound

        x_bound = x.clone()

        x = F.relu(self.deconv(x))
        mask = self.predictor(x) 

        x_bo = F.relu(self.bo_deconv(x_bo))
        mask_bo = self.predictor_bo(x_bo) 

        x_bound_bo = F.relu(self.boundary_deconv_bo(x_bound_bo))
        boundary_bo = self.boundary_predictor_bo(x_bound_bo) 

        x_bound = F.relu(self.boundary_deconv(x_bound))
        boundary = self.boundary_predictor(x_bound) 

        return mask, boundary, mask_bo, boundary_bo


def build_mask_head(cfg, input_shape):
    """
    Build a mask head defined by `cfg.MODEL.ROI_MASK_HEAD.NAME`.
    """
    return MaskRCNNConvUpsampleHead(cfg, input_shape)