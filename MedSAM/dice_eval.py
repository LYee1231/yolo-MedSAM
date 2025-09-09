import os
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import csv
from typing import Optional


def load_mask(path: Path) -> np.ndarray:
    """Load PNG mask as numpy array, keep it as 2D (H, W)."""
    img = Image.open(path)
    # Convert to grayscale to avoid confusion from RGB channels
    if img.mode != "L":
        img = img.convert("L")
    return np.array(img)

def to_foreground(mask: np.ndarray, binarize: bool, thr: int, class_id: Optional[int]) -> np.ndarray:
    """
    Convert raw mask into foreground boolean map:
    - If class_id is provided (>=0), foreground is mask == class_id
    - Otherwise: if binarize=True, foreground is mask >= thr; if binarize=False, foreground is mask > 0
    """
    if class_id is not None and class_id >= 0:
        fg = (mask == class_id)
    else:
        fg = (mask >= thr) if binarize else (mask > 0)
    return fg

def dice_score(gt: np.ndarray, pred: np.ndarray, eps: float = 1e-8) -> float:
    """
    Dice = 2TP / (2TP + FP + FN)
    Computed based on foreground pixels only (boolean arrays).
    """
    gt = gt.astype(bool)
    pred = pred.astype(bool)

    intersection = np.logical_and(gt, pred).sum(dtype=np.float64)
    gt_sum = gt.sum(dtype=np.float64)
    pred_sum = pred.sum(dtype=np.float64)

    denom = gt_sum + pred_sum
    if denom == 0:
        # Both GT and Pred have no foreground: define as perfect match
        return 1.0
    return (2.0 * intersection) / (denom + eps)

def pair_files(gt_dir: Path, pred_dir: Path, exts=(".png", ".jpg", ".jpeg", ".tif", ".tiff")):
    """
    Pair files by filename stem (without extension).
    """
    gt_map = {}
    for p in gt_dir.rglob("*"):
        if p.suffix.lower() in exts and p.is_file():
            gt_map[p.stem] = p

    pairs = []
    missing_pred = []
    for stem, gt_path in gt_map.items():
        # Look for prediction file with the same stem in pred_dir
        found = None
        for ext in exts:
            cand = pred_dir / f"{stem}{ext}"
            if cand.exists():
                found = cand
                break
        if found is None:
            # Allow subdirectory structures in pred_dir, search recursively
            for q in pred_dir.rglob(f"{stem}*"):
                if q.suffix.lower() in exts and q.is_file() and q.stem == stem:
                    found = q
                    break
        if found is not None:
            pairs.append((gt_path, found))
        else:
            missing_pred.append(stem)

    return pairs, missing_pred

def main():
    parser = argparse.ArgumentParser(description="Compute Dice score between GT masks and predicted masks (PNG).")
    parser.add_argument("--gt_dir", type=str, required=True, help="Ground truth mask directory")
    parser.add_argument("--pred_dir", type=str, required=True, help="Predicted mask directory")
    parser.add_argument("--class_id", type=int, default=None, help="Class ID to evaluate (e.g., 1). Default None means nonzero pixels are foreground.")
    parser.add_argument("--binarize", action="store_true", help="If set, binarize grayscale masks (effective only when class_id is None)")
    parser.add_argument("--thr", type=int, default=128, help="Binarization threshold (>=thr is foreground), default 128")
    parser.add_argument("--save_csv", type=str, default=None, help="Optional: save per-image Dice scores to CSV")
    args = parser.parse_args()

    gt_dir = Path(args.gt_dir)
    pred_dir = Path(args.pred_dir)
    assert gt_dir.exists() and gt_dir.is_dir(), f"GT directory does not exist: {gt_dir}"
    assert pred_dir.exists() and pred_dir.is_dir(), f"Pred directory does not exist: {pred_dir}"

    pairs, missing_pred = pair_files(gt_dir, pred_dir)

    if len(pairs) == 0:
        print("No matching files found. Please check that filenames match (without extensions).")
        return

    if missing_pred:
        print(f"Warning: {len(missing_pred)} GT files have no matching prediction (examples: {missing_pred[:5]} ...)")

    rows = []
    dices = []

    for gt_path, pred_path in pairs:
        gt = load_mask(gt_path)
        pred = load_mask(pred_path)

        if gt.shape != pred.shape:
            raise ValueError(f"Shape mismatch: {gt_path.name} {gt.shape} vs {pred_path.name} {pred.shape}")

        gt_fg = to_foreground(gt, args.binarize, args.thr, args.class_id)
        pred_fg = to_foreground(pred, args.binarize, args.thr, args.class_id)

        d = dice_score(gt_fg, pred_fg)
        dices.append(d)
        rows.append({"name": gt_path.stem, "dice": float(d)})

    mean_dice = float(np.mean(dices)) if dices else 0.0

    # Print summary
    print(f"Number of evaluated images: {len(dices)}")
    print(f"Mean Dice: {mean_dice:.4f}")
    # Show preview of first few results
    preview = ", ".join([f"{r['name']}:{r['dice']:.4f}" for r in rows[:5]])
    if preview:
        print(f"Examples (first 5): {preview}")

    # Save to CSV
    if args.save_csv:
        out_csv = Path(args.save_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["name", "dice"])
            writer.writeheader()
            writer.writerows(rows)
            writer.writerow({"name": "__mean__", "dice": mean_dice})
        print(f"Results saved to: {out_csv.resolve()}")

if __name__ == "__main__":
    main()
