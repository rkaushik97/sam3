#!/usr/bin/env python
"""Run SAM 3 image inference on a few sample images and save visualizations.

This is the smoke test for the M1 / MPS support branch. It picks the best
available device (cuda > mps > cpu) automatically and writes one PNG per
(image, prompt) pair into the same folder as this script.

Usage:
    python m1_test/run_sam3_image_inference.py
"""

import os
import warnings
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from sam3.model.device_utils import get_default_device
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path(__file__).resolve().parent

# (image filename in assets/images, text prompt)
TEST_CASES = [
    ("truck.jpg", "truck"),
    ("groceries.jpg", "grocery bag"),
    ("test_image.jpg", "child"),
]


def overlay_masks(image_pil, masks, boxes, scores, prompt, save_path):
    """Render the input image with per-instance mask overlays + boxes + scores."""
    img = np.array(image_pil.convert("RGB"))
    H, W = img.shape[:2]

    fig, ax = plt.subplots(1, 1, figsize=(10, 10 * H / max(W, 1)))
    ax.imshow(img)

    if masks is None or masks.shape[0] == 0:
        ax.set_title(f'prompt="{prompt}" — no detections')
    else:
        # masks: (N, 1, H, W) or (N, H, W) float in {0,1}-ish
        m = masks.detach().to("cpu").float().numpy()
        if m.ndim == 4:
            m = m[:, 0]
        b = boxes.detach().to("cpu").float().numpy()
        s = scores.detach().to("cpu").float().numpy()

        rng = np.random.default_rng(0)
        for i in range(m.shape[0]):
            color = rng.random(3)
            mask_rgba = np.zeros((H, W, 4), dtype=np.float32)
            mask_rgba[..., :3] = color
            mask_rgba[..., 3] = (m[i] > 0.5) * 0.5
            ax.imshow(mask_rgba)
            # box is xyxy in pixel coords
            x0, y0, x1, y1 = b[i].tolist()
            ax.add_patch(
                mpatches.Rectangle(
                    (x0, y0),
                    x1 - x0,
                    y1 - y0,
                    fill=False,
                    edgecolor=color,
                    linewidth=2,
                )
            )
            ax.text(
                x0,
                max(y0 - 6, 0),
                f"{s[i]:.2f}",
                color="white",
                fontsize=10,
                bbox=dict(facecolor=color, alpha=0.8, pad=2, edgecolor="none"),
            )

        ax.set_title(f'prompt="{prompt}" — {m.shape[0]} detection(s)')

    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    device = get_default_device()
    print(f"using device: {device}")
    print(f"torch: {torch.__version__}, mps available: {torch.backends.mps.is_available()}, cuda available: {torch.cuda.is_available()}")

    print("building SAM 3 image model (this downloads the checkpoint on first run)...")
    model = build_sam3_image_model()
    model.eval()
    # Lower the confidence threshold a little — running fp32 on MPS can shift
    # scores ~1-2% vs the CUDA bf16 baseline, so a few borderline detections
    # otherwise just barely miss the default 0.5 cutoff.
    processor = Sam3Processor(model, confidence_threshold=0.3)
    print(f"model on {next(model.parameters()).device}")

    for img_name, prompt in TEST_CASES:
        img_path = REPO_ROOT / "assets" / "images" / img_name
        if not img_path.exists():
            print(f"  skipping {img_name}: not found at {img_path}")
            continue
        print(f"\n[{img_name}] prompt={prompt!r}")
        image = Image.open(img_path).convert("RGB")

        with torch.inference_mode():
            state = processor.set_image(image)
            output = processor.set_text_prompt(state=state, prompt=prompt)

        masks = output["masks"]
        boxes = output["boxes"]
        scores = output["scores"]
        n = 0 if masks is None else int(masks.shape[0])
        print(f"  -> {n} detection(s) (threshold={processor.confidence_threshold})")
        if n > 0:
            print(f"     scores: {[f'{s:.3f}' for s in scores.detach().cpu().tolist()]}")
        else:
            # Show top raw scores so we can see if everything was just below threshold
            # vs. the model genuinely seeing nothing.
            raw = output.get("scores_all")
            if raw is None:
                # processor doesn't return raw scores by default — re-run with thresh=0 just for debug
                processor.set_confidence_threshold(0.0)
                output_dbg = processor.set_text_prompt(state=state, prompt=prompt)
                raw = output_dbg["scores"].detach().cpu()
                topk = torch.topk(raw, k=min(5, raw.numel()))
                print(f"     top-5 raw scores at thresh=0: {[f'{s:.3f}' for s in topk.values.tolist()]}")
                processor.set_confidence_threshold(0.3)

        out_path = OUT_DIR / f"{Path(img_name).stem}__{prompt.replace(' ', '_')}.png"
        overlay_masks(image, masks, boxes, scores, prompt, out_path)
        print(f"  saved: {out_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
