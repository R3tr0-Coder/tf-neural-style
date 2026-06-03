"""
stylize.py — Run optimization-based Neural Style Transfer.

Usage:
    # Basic
    python scripts/stylize.py --content assets/photo.jpg --style assets/painting.jpg

    # Custom settings
    python scripts/stylize.py \\
        --content  assets/photo.jpg \\
        --style    assets/starry_night.jpg \\
        --output   outputs/my_result.jpg \\
        --config   configs/default.yaml \\
        --iter     500 \\
        --size     640 \\
        --content-weight 1e4 \\
        --style-weight   1e-2 \\
        --init     content
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import (
    Stylizer,
    load_config,
    setup_logging,
    set_seed,
    get_gpu_info,
    configure_gpu_memory_growth,
)

logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Neural Style Transfer (optimization-based)")
    p.add_argument("--content", required=True, help="Path to content image")
    p.add_argument("--style",   required=True, help="Path to style image")
    p.add_argument("--output",  default="outputs/result.jpg", help="Output image path")
    p.add_argument("--config",  default="configs/default.yaml", help="YAML config file")

    # CLI overrides
    p.add_argument("--iter",           type=int,   default=None, help="Number of optimization steps")
    p.add_argument("--size",           type=int,   default=None, help="Max image dimension (px)")
    p.add_argument("--content-weight", type=float, default=None, help="Content loss weight")
    p.add_argument("--style-weight",   type=float, default=None, help="Style loss weight")
    p.add_argument("--tv-weight",      type=float, default=None, help="Total variation weight")
    p.add_argument("--lr",             type=float, default=None, help="Adam learning rate")
    p.add_argument("--init",           type=str,   default="content",
                   choices=["content", "noise"],   help="Initialize from content or random noise")
    return p.parse_args()


def apply_overrides(cfg: dict, args) -> dict:
    st = cfg["style_transfer"]
    if args.iter           is not None: st["iterations"]     = args.iter
    if args.size           is not None: st["image_size"]      = args.size
    if args.content_weight is not None: st["content_weight"]  = args.content_weight
    if args.style_weight   is not None: st["style_weight"]    = args.style_weight
    if args.tv_weight      is not None: st["tv_weight"]       = args.tv_weight
    if args.lr             is not None: st["learning_rate"]   = args.lr
    return cfg


def main():
    args = parse_args()
    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args)

    output_path = Path(args.output)
    setup_logging(output_path.parent)
    configure_gpu_memory_growth()
    set_seed(cfg["experiment"].get("seed", 42))

    logger.info("=" * 60)
    logger.info("Neural Style Transfer (optimization-based)")
    logger.info(f"  Content  : {args.content}")
    logger.info(f"  Style    : {args.style}")
    logger.info(f"  Output   : {args.output}")
    logger.info(f"  Device   : {get_gpu_info()}")
    st = cfg["style_transfer"]
    logger.info(f"  Iterations     : {st['iterations']}")
    logger.info(f"  Image size     : {st['image_size']}")
    logger.info(f"  Content weight : {st['content_weight']:.2e}")
    logger.info(f"  Style weight   : {st['style_weight']:.2e}")
    logger.info(f"  TV weight      : {st['tv_weight']}")
    logger.info(f"  Init           : {args.init}")
    logger.info("=" * 60)

    stylizer = Stylizer(cfg)
    stylizer.run(args.content, args.style, args.output, init=args.init)


if __name__ == "__main__":
    main()
