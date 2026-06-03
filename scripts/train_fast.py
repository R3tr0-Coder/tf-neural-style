"""
train_fast.py — Train a Fast Style Transfer feed-forward network.

Usage:
    python scripts/train_fast.py \\
        --style   assets/starry_night.jpg \\
        --dataset /path/to/coco/train2017/ \\
        --config  configs/default.yaml \\
        --epochs  2
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import (
    FastStyleTrainer,
    load_config,
    setup_logging,
    set_seed,
    get_gpu_info,
    configure_gpu_memory_growth,
    build_fast_style_net,
    print_model_summary,
)

logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Train a Fast Style Transfer network")
    p.add_argument("--style",   required=True, help="Path to the target style image")
    p.add_argument("--dataset", required=True, help="Path to folder of training images")
    p.add_argument("--config",  default="configs/default.yaml", help="YAML config file")
    p.add_argument("--epochs",  type=int,   default=None, help="Override number of epochs")
    p.add_argument("--lr",      type=float, default=None, help="Override learning rate")
    return p.parse_args()


def apply_overrides(cfg, args):
    if args.epochs is not None: cfg["fast_style"]["epochs"]        = args.epochs
    if args.lr     is not None: cfg["fast_style"]["learning_rate"] = args.lr
    return cfg


def main():
    args = parse_args()
    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args)

    ckpt_dir = Path(cfg["fast_style"]["checkpoint_dir"])
    setup_logging(ckpt_dir)
    configure_gpu_memory_growth()
    set_seed(cfg["experiment"].get("seed", 42))

    logger.info("=" * 60)
    logger.info("Fast Style Transfer — Training")
    logger.info(f"  Style image : {args.style}")
    logger.info(f"  Dataset     : {args.dataset}")
    logger.info(f"  Device      : {get_gpu_info()}")
    fs = cfg["fast_style"]
    logger.info(f"  Epochs      : {fs['epochs']}")
    logger.info(f"  Batch size  : {fs['batch_size']}")
    logger.info(f"  Image size  : {fs['image_size']}")
    logger.info(f"  LR          : {fs['learning_rate']:.2e}")
    logger.info("=" * 60)

    # Print model architecture summary
    model = build_fast_style_net()
    print_model_summary(model, "FastStyleNet")

    trainer = FastStyleTrainer(cfg)
    trainer.train(args.style, args.dataset)


if __name__ == "__main__":
    main()
