import torch


import argparse
from lib.config.bcnet_config import get_cfg
from lib.layers.mask_head import build_mask_head

# constants
WINDOW_NAME = "COCO detections"


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.FCOS.INFERENCE_TH = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin models")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument("--input", nargs="+", help="A list of space separated input images")
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
             "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    cfg = setup_cfg(args)


    stage_channel_factor = 2 ** 3  # res5 is 8x res2
    out_channels = 256 # cfg.MODEL.RESNETS.RES2_OUT_CHANNELS * stage_channel_factor
    pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
    # shapeSpec = ShapeSpec(channels=out_channels, width=pooler_resolution, height=pooler_resolution) # [b_size, 2048, 14, 14]

    input_sp = {"channels": 0}
    # input_sp.channels = out_channels
    head = build_mask_head(cfg, out_channels)

    input = torch.rand([16, 256, 64, 48])
    out = head(input)

    print('over')