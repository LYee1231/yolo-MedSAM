# -*- coding: utf-8 -*-
"""
Use single-line YOLO txt boxes to prompt MedSAM (aligned with MedSAM_Inference.py style).
"""

import argparse
import os
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from skimage import io, transform
from segment_anything import sam_model_registry

def read_single_yolo_txt(txt_path: Path):
    """
    Reading a single line of YOLO labels: cls x y w h [conf] (all normalised to [0,1])
    Returns (xc, yc, w, h, conf or None); invalid inputs return None
    """
    if not txt_path.is_file():
        return None
    line = txt_path.read_text(encoding="utf-8", errors="ignore").strip()
    if not line:
        return None
    parts = line.split()
    if len(parts) < 5:
        return None
    try:
        x = float(parts[1]); y = float(parts[2]); w = float(parts[3]); h = float(parts[4])
        conf = float(parts[5]) if len(parts) >= 6 else None
    except Exception:
        return None
    return (x, y, w, h, conf)

def yolo_norm_to_xyxy(norm_box, W, H):
    """(xc, yc, w, h) normalised -> pixels [x1, y1, x2, y2], and cropped to the image range"""
    xc, yc, w, h, _ = norm_box
    xc *= W; yc *= H; w *= W; h *= H
    x1 = max(0, min(W - 1, xc - w / 2))
    y1 = max(0, min(H - 1, yc - h / 2))
    x2 = max(0, min(W - 1, xc + w / 2))
    y2 = max(0, min(H - 1, yc + h / 2))
    return np.array([x1, y1, x2, y2], dtype=np.float32)

def load_image_rgb(p: Path):
    """Read image; convert grayscale to three-channel RGB, convert BGR to RGB"""
    im = io.imread(str(p))
    if im is None:
        raise FileNotFoundError(f"Failed to read image: {p}")
    if im.ndim == 2:
        im3 = np.repeat(im[:, :, None], 3, axis=-1)
    else:
        im3 = im
    return im3

def preprocess_1024(image_rgb):
    """
    Following the approach in MedSAM_Inference.py:
    - Resize to (1024, 1024)
    - Linearly normalise to [0,1]
    - Return img_1024 (previously np.uint8 in [0,255] → now returns float[0,1]), tensor(1,3,1024,1024)
    """
    H, W, _ = image_rgb.shape
    img_1024 = transform.resize(
        image_rgb, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True
    ).astype(np.uint8)
    img_1024 = (img_1024 - img_1024.min()) / max(1e-8, (img_1024.max() - img_1024.min()))
    img_1024_tensor = torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0)  # (1,3,1024,1024)
    return img_1024, img_1024_tensor, H, W

@torch.no_grad()
def medsam_from_box(medsam_model, img_embed, box_1024, H, W):
    """
    Call path within the reusable script: prompt_encoder → mask_decoder → upscaled to original image dimensions
    """
    box_t = torch.as_tensor(box_1024, dtype=torch.float, device=img_embed.device)
    if box_t.ndim == 2:
        box_t = box_t[:, None, :]  # (B,1,4)

    sparse_emb, dense_emb = medsam_model.prompt_encoder(points=None, boxes=box_t, masks=None)
    low_res_logits, _ = medsam_model.mask_decoder(
        image_embeddings=img_embed,
        image_pe=medsam_model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_emb,
        dense_prompt_embeddings=dense_emb,
        multimask_output=False,
    )
    low_res_pred = torch.sigmoid(low_res_logits)  # (1,1,256,256)
    low_res_pred = F.interpolate(low_res_pred, size=(H, W), mode="bilinear", align_corners=False)
    mask = (low_res_pred.squeeze().cpu().numpy() > 0.5).astype(np.uint8)
    return mask

def main():
    ap = argparse.ArgumentParser("YOLO txt -> MedSAM inference (aligned with your pipeline)")
    ap.add_argument("--images_dir", required=True, help="folder of test images")
    ap.add_argument("--labels_dir", required=True, help="folder of YOLO txt (single line per image)")
    ap.add_argument("--output_dir", required=True, help="folder to save masks")
    ap.add_argument("--checkpoint", required=True, help="MedSAM checkpoint (.pth)")
    ap.add_argument("--model_type", default="vit_b", help="vit_b / vit_l / vit_h ...")
    ap.add_argument("--device", default="", help="'0' -> cuda:0, or 'cpu'")
    ap.add_argument("--save_overlay", action="store_true", help="also save overlay visualization")
    args = ap.parse_args()

    images_dir = Path(args.images_dir)
    labels_dir = Path(args.labels_dir)
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    assert images_dir.is_dir(), f"images_dir not found: {images_dir}"
    assert labels_dir.is_dir(), f"labels_dir not found: {labels_dir}"
    assert Path(args.checkpoint).is_file(), f"checkpoint not found: {args.checkpoint}"

    device = args.device if args.device else ("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f">>> device: {device}")

    # Load MedSAM (same registry and usage as MedSAM_Inference.py)
    medsam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    medsam = medsam.to(device)
    medsam.eval()

    # Collect images
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    img_paths = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in exts])
    print(f">>> found images: {len(img_paths)}")

    done, skipped = 0, 0
    for i, img_path in enumerate(img_paths, 1):
        stem = img_path.stem
        txt_path = labels_dir / f"{stem}.txt"
        norm_box = read_single_yolo_txt(txt_path)
        if norm_box is None:
            print(f"[{i}/{len(img_paths)}] SKIP {stem}: label missing/invalid -> {txt_path}")
            skipped += 1
            continue

        img_rgb = load_image_rgb(img_path)
        img_1024, img_1024_tensor, H, W = preprocess_1024(img_rgb)

        # YOLO normalised bounding box -> original pixel-based bounding box -> 1024 coordinate box (map the box to 1024 according to your script)
        box_xyxy = yolo_norm_to_xyxy(norm_box, W, H)  # Original image coordinates
        box_1024 = (box_xyxy / np.array([W, H, W, H], dtype=np.float32)) * 1024.0
        box_1024 = box_1024[None, :]  # (1,4)

        with torch.no_grad():
            img_1024_tensor = img_1024_tensor.to(device)  # (1,3,1024,1024)
            img_embed = medsam.image_encoder(img_1024_tensor)  # (1,256,64,64)

        mask = medsam_from_box(medsam, img_embed, box_1024, H, W)

        # Save
        out_png = out_dir / f"{stem}_mask.png"
        out_npy = out_dir / f"{stem}_mask.npy"
        cv2.imwrite(str(out_png), (mask * 255).astype(np.uint8))
        np.save(out_npy, mask)

        if args.save_overlay:
            overlay = img_rgb.copy()
            overlay[mask.astype(bool)] = [255, 0, 0]
            vis = cv2.addWeighted(img_rgb, 0.6, overlay, 0.4, 0)
            vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(out_dir / f"{stem}_overlay.jpg"), vis_bgr)

        print(f"[{i}/{len(img_paths)}] OK {stem}: box={box_xyxy.astype(int).tolist()}")
        done += 1

    print(f">>> finished. saved={done}, skipped={skipped}. out={out_dir}")

if __name__ == "__main__":
    main()
