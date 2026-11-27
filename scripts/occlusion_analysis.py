"""Exploratory border and matched-area lung blur on external positive CXRs."""

import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F

from scripts.lesion_roi import bootstrap_mean
from scripts.shenzhen_roi import MEAN, STD, model_for


def matched_lung_masks(lung, count, seed, repeats=3):
    indices = np.flatnonzero(lung)
    if len(indices) < count:
        raise ValueError("La máscara pulmonar es menor que el borde de comparación")
    rng = np.random.default_rng(seed)
    masks = []
    for _ in range(repeats):
        field = cv2.resize(rng.random((7, 7), dtype=np.float32), lung.shape[::-1], interpolation=cv2.INTER_LINEAR)
        selected = indices[np.argpartition(field.ravel()[indices], -count)[-count:]]
        mask = np.zeros(lung.shape, dtype=bool)
        mask.ravel()[selected] = True
        masks.append(mask)
    return masks


def to_batch(images, device):
    arrays = [np.repeat(gray[:, :, None], 3, axis=2) for gray in images]
    batch = np.stack(arrays).astype(np.float32)
    return torch.from_numpy(((batch - MEAN) / STD).transpose(0, 3, 1, 2)).to(device)


def run(data_root, results):
    data_root, results = Path(data_root), Path(results)
    split = pd.read_csv(results / "splits.csv")
    frame = split[(split.split == "external") & (split.label == 1)].sort_values("name").reset_index(drop=True)
    if len(frame) != 58:
        raise ValueError("Conteo inesperado de positivos externos")
    files = {(path.parent.name, path.name): path for path in data_root.rglob("*.png")
             if path.name in set(frame.name) and path.parent.name in ("CXR_png", "leftMask", "rightMask")}
    size = 224
    width = 5
    border = np.ones((size, size), dtype=bool)
    border[width:-width, width:-width] = False
    cases = []
    for i, name in enumerate(frame.name):
        gray = np.asarray(Image.open(files[("CXR_png", name)]).convert("L").resize((size, size), Image.Resampling.BILINEAR), dtype=np.float32) / 255
        left = np.asarray(Image.open(files[("leftMask", name)]).convert("L").resize((size, size), Image.Resampling.NEAREST)) > 0
        right = np.asarray(Image.open(files[("rightMask", name)]).convert("L").resize((size, size), Image.Resampling.NEAREST)) > 0
        lung = left | right
        blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=10)
        variants = [gray, np.where(border, blurred, gray), np.where(lung, blurred, gray)]
        variants += [np.where(mask, blurred, gray) for mask in matched_lung_masks(lung, border.sum(), i + 481)]
        cases.append((name, variants, float(lung.mean())))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    for model_name in ("densenet121", "resnet50"):
        for seed in (0, 1, 2):
            model, _ = model_for(model_name)
            model.load_state_dict(torch.load(results / f"{model_name}_s{seed}.pt", map_location="cpu", weights_only=True))
            model = model.to(device).eval()
            with torch.no_grad():
                for name, variants, lung_area in cases:
                    probabilities = F.softmax(model(to_batch(variants, device)), 1)[:, 1].cpu().numpy()
                    original, border_blur, full_lung_blur = probabilities[:3]
                    matched_lung_blur = probabilities[3:].mean()
                    rows.append(dict(model=model_name, seed=seed, name=name,
                                     probability_tb=float(original), border_drop=float(original - border_blur),
                                     matched_lung_drop=float(original - matched_lung_blur),
                                     full_lung_drop=float(original - full_lung_blur),
                                     border_minus_matched_lung=float(matched_lung_blur - border_blur),
                                     lung_area=lung_area))
            del model
            if device == "cuda": torch.cuda.empty_cache()
    scores = pd.DataFrame(rows)
    scores.to_csv(results / "occlusion_scores.csv", index=False)
    per_image = scores.groupby(["model", "name"], as_index=False).agg(
        border_drop=("border_drop", "mean"), matched_lung_drop=("matched_lung_drop", "mean"),
        full_lung_drop=("full_lung_drop", "mean"),
        border_minus_matched_lung=("border_minus_matched_lung", "mean"))
    summary = {"n_positive_external": len(frame), "border_area_fraction": float(border.mean()),
               "lung_mask_repeats": 3, "models": {}}
    for model_name in ("densenet121", "resnet50"):
        part = per_image[per_image.model == model_name]
        summary["models"][model_name] = {key: bootstrap_mean(part[key]) for key in
                                          ("border_drop", "matched_lung_drop", "full_lung_drop", "border_minus_matched_lung")}
    (results / "occlusion_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.1), sharey=True)
    for ax, model_name in zip(axs, ("densenet121", "resnet50")):
        part = per_image[per_image.model == model_name]
        ax.boxplot([part.border_drop, part.matched_lung_drop], tick_labels=["Borde", "Pulmón igual área"], showmeans=True)
        ax.axhline(0, color="gray", linestyle="--", linewidth=1)
        ax.set_title("DenseNet121" if model_name == "densenet121" else "ResNet50")
        ax.tick_params(axis="x", labelsize=7)
        ax.grid(alpha=.25)
    axs[0].set_ylabel("Caída de probabilidad de TB")
    fig.tight_layout(pad=.8)
    fig.savefig(results / "fig_occlusion.png", dpi=220)
    plt.close(fig)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--results", default="results")
    args = parser.parse_args()
    print(json.dumps(run(args.data_root, args.results), indent=2))
