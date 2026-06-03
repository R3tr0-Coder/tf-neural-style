"""
apply_fast.py — Apply a trained Fast Style Network to images.

Usage:
    # Single image
    python scripts/apply_fast.py \\
        --checkpoint outputs/fast_checkpoints/fast_style_final \\
        --image      path/to/photo.jpg \\
        --output     outputs/styled.jpg

    # Batch (folder)
    python scripts/apply_fast.py \\
        --checkpoint outputs/fast_checkpoints/fast_style_final \\
        --folder     path/to/images/ \\
        --out-dir    outputs/batch_results/
"""

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import apply_fast_style, configure_gpu_memory_growth

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    p = argparse.ArgumentParser(description="Apply trained Fast Style Network")
    p.add_argument("--checkpoint", required=True, help="Path to saved Keras model")
    p.add_argument("--image",   default=None, help="Single input image")
    p.add_argument("--folder",  default=None, help="Folder of input images")
    p.add_argument("--output",  default="outputs/styled.jpg", help="Output path (single image)")
    p.add_argument("--out-dir", default="outputs/batch/",     help="Output dir (batch mode)")
    p.add_argument("--max-dim", type=int, default=1024, help="Max image dimension")
    return p.parse_args()


def main():
    args = parse_args()
    configure_gpu_memory_growth()

    if args.image:
        apply_fast_style(args.checkpoint, args.image, args.output, args.max_dim)

    elif args.folder:
        folder = Path(args.folder)
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        images = [p for p in folder.iterdir() if p.suffix.lower() in SUPPORTED]
        logger.info(f"Processing {len(images)} images from {folder}")

        for img_path in sorted(images):
            out_path = out_dir / img_path.name
            apply_fast_style(args.checkpoint, str(img_path), str(out_path), args.max_dim)

        logger.info(f"All done → {out_dir}")
    else:
        logger.error("Provide --image or --folder.")
        sys.exit(1)


if __name__ == "__main__":
    main()
