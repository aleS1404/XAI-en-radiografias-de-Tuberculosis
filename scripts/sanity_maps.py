"""A learned-weights sanity check for TB-target Grad-CAM on Montgomery."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from scipy.stats import spearmanr
import torch

from scripts.lesion_roi import bootstrap_mean, boxes_to_mask, roi_scores
from scripts.shenzhen_roi import MEAN, STD, model_for


def heatmaps(model, layer, tensors, device):
    cam = GradCAM(model=model, target_layers=[layer])
    maps = []
    for i in range(0, len(tensors), 2):
        batch = tensors[i:i+2].to(device)
        maps.extend(cam(input_tensor=batch, targets=[ClassifierOutputTarget(1)] * len(batch)))
    cam.activations_and_grads.release()
    return maps


def run(data_root, out_dir):
    data_root, out_dir = Path(data_root), Path(out_dir)
    annotations = pd.read_csv("data/montgomery_consensus_roi.csv")
    names = sorted(annotations.patientId.unique())
    if len(names) != 55:
        raise ValueError("Conteo inesperado de radiografías anotadas")
    size = 224
    tensors, masks, lungs = [], {}, {}
    for name in names:
        image = np.asarray(Image.open(next(p for p in data_root.rglob(name) if p.parent.name == "CXR_png"))
                           .convert("L").resize((size, size), Image.Resampling.BILINEAR), dtype=np.float32) / 255
        rgb = np.repeat(image[:, :, None], 3, axis=2)
        tensors.append(torch.from_numpy(((rgb - MEAN) / STD).transpose(2, 0, 1)))
        left = np.asarray(Image.open(next(p for p in data_root.rglob(name) if p.parent.name == "leftMask"))
                          .convert("L").resize((size, size), Image.Resampling.NEAREST)) > 0
        right = np.asarray(Image.open(next(p for p in data_root.rglob(name) if p.parent.name == "rightMask"))
                           .convert("L").resize((size, size), Image.Resampling.NEAREST)) > 0
        lungs[name] = left | right
        masks[name] = boxes_to_mask(annotations[annotations.patientId == name], size)
    X = torch.stack(tensors)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    for model_name in ("densenet121", "resnet50"):
        model, layer = model_for(model_name)
        model.load_state_dict(torch.load(out_dir / f"{model_name}_s0.pt", map_location="cpu", weights_only=True))
        model = model.to(device).eval()
        learned = heatmaps(model, layer, X, device)
        del model
        if device == "cuda": torch.cuda.empty_cache()
        torch.manual_seed(914)
        model, layer = model_for(model_name)
        model = model.to(device).eval()
        randomized = heatmaps(model, layer, X, device)
        del model
        if device == "cuda": torch.cuda.empty_cache()
        for name, trained_map, random_map in zip(names, learned, randomized):
            trained_score = roi_scores(trained_map, masks[name], lungs[name])
            random_score = roi_scores(random_map, masks[name], lungs[name])
            rho = float(spearmanr(trained_map.ravel()[::16], random_map.ravel()[::16]).statistic) if trained_map.sum() > 0 and random_map.sum() > 0 else np.nan
            rows.append(dict(model=model_name, name=name, correlation=rho,
                             trained_gain=trained_score["gain_lung"],
                             random_weights_gain=random_score["gain_lung"],
                             trained_map_valid=bool(trained_map.sum() > 0),
                             random_weights_map_valid=bool(random_map.sum() > 0)))
    scores = pd.DataFrame(rows)
    scores.to_csv(out_dir / "sanity_maps_scores.csv", index=False)
    summary = {"n_images": len(names), "models": {}}
    for model_name in ("densenet121", "resnet50"):
        frame = scores[scores.model == model_name]
        rho = frame.correlation.median()
        summary["models"][model_name] = {
            "median_map_spearman": None if pd.isna(rho) else float(rho),
            "valid_map_pairs": int(frame.correlation.notna().sum()),
            "trained_valid_maps": int(frame.trained_map_valid.sum()),
            "random_weights_valid_maps": int(frame.random_weights_map_valid.sum()),
            "trained_gain": bootstrap_mean(frame.trained_gain),
            "random_weights_gain": bootstrap_mean(frame.random_weights_gain),
        }
    (out_dir / "sanity_maps_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--results", default="results")
    args = parser.parse_args()
    print(json.dumps(run(args.data_root, args.results), indent=2))
