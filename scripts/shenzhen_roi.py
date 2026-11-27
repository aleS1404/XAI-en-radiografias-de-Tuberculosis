"""Exploratory TB finding localization on held-out Shenzhen radiographs."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import models

from scripts.lesion_roi import bootstrap_mean, boxes_to_mask, roi_scores


MEAN = np.array([.485, .456, .406], dtype=np.float32)
STD = np.array([.229, .224, .225], dtype=np.float32)


def model_for(name):
    if name == "densenet121":
        model = models.densenet121(weights=None)
        model.classifier = torch.nn.Sequential(torch.nn.Dropout(.3), torch.nn.Linear(model.classifier.in_features, 2))
        layer = model.features[-1]
    elif name == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = torch.nn.Sequential(torch.nn.Dropout(.3), torch.nn.Linear(model.fc.in_features, 2))
        layer = model.layer4[-1]
    else:
        raise ValueError(name)
    return model, layer


def run(data_root, out_dir):
    data_root, out_dir = Path(data_root), Path(out_dir)
    split = pd.read_csv(out_dir / "splits.csv")
    annotations = pd.read_csv("data/shenzhen_consensus_roi.csv")
    frame = split[(split.split == "test") & split.name.isin(annotations.patientId)].sort_values("name")
    if len(frame) != 18 or not (frame.label == 1).all():
        raise ValueError(f"Se esperaban 18 casos anotados positivos en el test, hay {len(frame)}")
    images = {p.name: p for p in data_root.rglob("*.png") if p.parent.name == "CXR_png" and p.name.startswith("CHNCXR")}
    size = 224
    tensors = []
    for name in frame.name:
        gray = np.asarray(Image.open(images[name]).convert("L").resize((size, size), Image.Resampling.BILINEAR), dtype=np.float32) / 255
        rgb = np.repeat(gray[:, :, None], 3, axis=2)
        tensors.append(torch.from_numpy(((rgb - MEAN) / STD).transpose(2, 0, 1)))
    X = torch.stack(tensors)
    masks = {name: boxes_to_mask(annotations[annotations.patientId == name], size) for name in frame.name}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    for model_name in ("densenet121", "resnet50"):
        for seed in (0, 1, 2):
            model, layer = model_for(model_name)
            model.load_state_dict(torch.load(out_dir / f"{model_name}_s{seed}.pt", map_location="cpu", weights_only=True))
            model = model.to(device).eval()
            cam = GradCAM(model=model, target_layers=[layer])
            maps = []
            for i in range(0, len(X), 2):
                batch = X[i:i+2].to(device)
                maps.extend(cam(input_tensor=batch, targets=[ClassifierOutputTarget(1)] * len(batch)))
            cam.activations_and_grads.release()
            rng = np.random.default_rng(seed + 777)
            for i, name in enumerate(frame.name):
                random_map = cv2.resize(rng.random((7, 7), dtype=np.float32), (size, size), interpolation=cv2.INTER_LINEAR)
                for method, heatmap in (("Grad-CAM", maps[i]), ("Aleatorio", random_map)):
                    score = roi_scores(heatmap, masks[name], np.ones((size, size), dtype=bool))
                    rows.append(dict(model=model_name, seed=seed, method=method, name=name,
                                     gain_image=score["gain_image"], point_image=score["point_lung"],
                                     roi_area=score["lesion_area_image"]))
            del cam, model
            if device == "cuda":
                torch.cuda.empty_cache()
    scores = pd.DataFrame(rows)
    scores.to_csv(out_dir / "shenzhen_test_roi_scores.csv", index=False)
    per_image = scores.groupby(["model", "method", "name"], as_index=False).agg(
        gain_image=("gain_image", "mean"), point_image=("point_image", "mean"), roi_area=("roi_area", "mean"))
    summary = {"n_test_images": len(frame), "n_boxes": int(len(annotations[annotations.patientId.isin(frame.name)])),
               "models": {}}
    for model_name in ("densenet121", "resnet50"):
        summary["models"][model_name] = {}
        for method in ("Grad-CAM", "Aleatorio"):
            part = per_image[(per_image.model == model_name) & (per_image.method == method)]
            summary["models"][model_name][method] = {
                "gain_image": bootstrap_mean(part.gain_image),
                "point_image": bootstrap_mean(part.point_image),
                "valid_images": int(part.gain_image.notna().sum()),
                "mean_roi_area": float(part.roi_area.mean())}
    (out_dir / "shenzhen_test_roi_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--results", default="results")
    args = parser.parse_args()
    print(json.dumps(run(args.data_root, args.results), indent=2))
