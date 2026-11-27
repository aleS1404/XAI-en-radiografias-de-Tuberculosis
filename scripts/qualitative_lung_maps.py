"""Same-case, positive-class Grad-CAM and LIME examples for the IEEE article."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import torch
import torch.nn.functional as F

from scripts.matched_lesion import image_files, read_case, select_regions, SIZE, tensor
from scripts.shenzhen_roi import MEAN, STD, model_for

LIME_SAMPLES = 800
LIME_SEED = 271
N_CASES = 4


def lime_positive(model, image, device):
    """Positive LIME coefficients on SLIC superpixels, using the same input scale."""
    rgb = np.repeat(np.round(image[:, :, None] * 255).astype(np.uint8), 3, axis=2)
    segmenter = SegmentationAlgorithm("slic", n_segments=60, compactness=10,
                                      sigma=1, start_label=1)

    @torch.no_grad()
    def predict_fn(batch):
        images = np.asarray(batch, dtype=np.float32) / 255.0
        x = torch.from_numpy(((images - MEAN) / STD).transpose(0, 3, 1, 2)).to(device)
        return F.softmax(model(x).float(), 1).cpu().numpy()

    explainer = lime_image.LimeImageExplainer(random_state=LIME_SEED)
    kwargs = dict(top_labels=2, hide_color=0, num_samples=LIME_SAMPLES,
                  batch_size=8, segmentation_fn=segmenter)
    try:
        explanation = explainer.explain_instance(rgb, predict_fn, progress_bar=False, **kwargs)
    except TypeError:
        explanation = explainer.explain_instance(rgb, predict_fn, **kwargs)
    coefficients = dict(explanation.local_exp[1])
    heat = np.zeros((SIZE, SIZE), dtype=np.float32)
    for segment in np.unique(explanation.segments):
        heat[explanation.segments == segment] = max(0.0, coefficients.get(int(segment), 0.0))
    return heat / heat.max() if heat.max() > 0 else heat


def draw_panel(ax, image, heat, lung, index, controls, title, draw_controls=False):
    ax.imshow(image, cmap="gray", vmin=0, vmax=1)
    if heat is not None:
        ax.imshow(heat, cmap="jet", vmin=0, vmax=1,
                  alpha=np.clip(heat, 0, 1) * .55)
    ax.contour(lung, levels=[.5], colors="yellow", linewidths=.65)
    ax.contour(index, levels=[.5], colors="red", linewidths=.9)
    if draw_controls:
        for mask in controls:
            ax.contour(mask, levels=[.5], colors="cyan", linewidths=.5)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def run(data_root, results, annotations):
    results = Path(results)
    eligible = pd.read_csv(results / "matched_eligibility.csv")
    names = sorted(eligible.loc[eligible.eligible, "name"])[:N_CASES]
    if len(names) != N_CASES:
        raise ValueError("No hay cuatro casos elegibles para ilustrar")
    annotations = pd.read_csv(annotations)
    files = image_files(data_root)
    cases = {}
    for name in names:
        image, lungs = read_case(name, files)
        region, reason = select_regions(annotations[annotations.patientId == name], lungs)
        if reason:
            raise ValueError(f"Cambió la elegibilidad de {name}: {reason}")
        cases[name] = image, region
    preds = pd.read_csv(results / "predictions.csv")
    cfg = json.loads((results / "config.json").read_text(encoding="utf-8"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    map_arrays = {}
    records = []
    for model_name in ("densenet121", "resnet50"):
        model, layer = model_for(model_name)
        model.load_state_dict(torch.load(results / f"{model_name}_s0.pt", map_location="cpu", weights_only=True))
        model = model.to(device).eval()
        cam = GradCAM(model=model, target_layers=[layer])
        fig, axes = plt.subplots(N_CASES, 3, figsize=(10.2, 12.0))
        for row, name in enumerate(names):
            image, region = cases[name]
            with torch.no_grad():
                probability = float(F.softmax(model(tensor([image], device)).float(), 1)[0, 1].cpu())
            threshold = float(cfg["thresholds"][f"{model_name}_s0"])
            predicted = int(probability >= threshold)
            gradcam = np.asarray(cam(input_tensor=tensor([image], device),
                                     targets=[ClassifierOutputTarget(1)])[0], dtype=np.float32)
            lime = lime_positive(model, image, device)
            map_arrays[f"{model_name}_{name}_gradcam"] = gradcam
            map_arrays[f"{model_name}_{name}_lime"] = lime
            records.append(dict(model=model_name, seed=0, name=name, true_label=1,
                                target_class=1, probability_tb=probability, threshold=threshold,
                                predicted=predicted, lime_samples=LIME_SAMPLES,
                                lime_seed=LIME_SEED, gradcam_valid=bool(gradcam.sum() > 0),
                                lime_valid=bool(lime.sum() > 0)))
            draw_panel(axes[row, 0], image, None, region["lung"], region["index"],
                       region["controls"], f"{name}\np(TB)={probability:.3f}; pred.={'TB' if predicted else 'normal'}", True)
            draw_panel(axes[row, 1], image, gradcam, region["lung"], region["index"],
                       region["controls"], "Grad-CAM, clase TB")
            draw_panel(axes[row, 2], image, lime, region["lung"], region["index"],
                       region["controls"], "LIME, clase TB")
        fig.suptitle("DenseNet121" if model_name == "densenet121" else "ResNet50", fontsize=14)
        fig.tight_layout()
        fig.savefig(results / f"fig_lung_xai_{model_name}.png", dpi=230)
        plt.close(fig)
        cam.activations_and_grads.release()
        del cam, model
        if device == "cuda":
            torch.cuda.empty_cache()
    np.savez_compressed(results / "qualitative_lung_maps.npz", **map_arrays)
    table = pd.DataFrame(records)
    table.to_csv(results / "qualitative_lung_cases.csv", index=False)
    return table


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--results", default="results")
    parser.add_argument("--annotations", default="data/montgomery_consensus_roi.csv")
    args = parser.parse_args()
    print(run(args.data_root, args.results, args.annotations).to_string(index=False))
