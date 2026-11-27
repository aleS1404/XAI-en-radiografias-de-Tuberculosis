"""Rebuild XAI figures with transparent zero relevance and valid-map filtering."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from scripts.shenzhen_roi import MEAN, STD, model_for


def run(data_root, results):
    data_root, results = Path(data_root), Path(results)
    sub_scores = pd.read_csv(results / "xai_subset_scores.csv")
    full_scores = pd.read_csv(results / "xai_full_scores.csv")
    area = float(full_scores[(full_scores.model == "densenet121") & (full_scores["set"] == "mg")].area.mean())
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 3), sharey=True)
    for ax, (model_name, title) in zip(axs, (("densenet121", "DenseNet121"), ("resnet50", "ResNet50"))):
        groups = [sub_scores[(sub_scores.model == model_name) & (sub_scores.method == method)].eil.dropna().values
                  for method in ("Grad-CAM", "LIME", "Aleatorio")]
        ax.boxplot(groups, showmeans=True, tick_labels=["Grad-CAM", "LIME", "Aleat."])
        ax.axhline(area, ls="--", color="gray", lw=1)
        ax.set_title(title)
        ax.grid(alpha=.25)
    axs[0].set_ylabel("Energía dentro del pulmón (EIL)")
    fig.tight_layout(pad=.8)
    fig.savefig(results / "fig_xai_box.png", dpi=220)
    plt.close(fig)

    predictions = pd.read_csv(results / "predictions.csv")
    mg = pd.read_csv(results / "splits.csv")
    mg = mg[mg.split == "external"].sort_values("name").reset_index(drop=True)
    rng = np.random.RandomState(0)
    ids = []
    for label in (0, 1):
        pool = mg.index[mg.label == label].values
        ids.extend(rng.choice(pool, size=min(20, len(pool)), replace=False).tolist())
    sub = mg.loc[ids].reset_index(drop=True)
    pred = predictions[(predictions.model == "densenet121") & (predictions.seed == 0) & (predictions.dataset == "mg")].set_index("name")
    predicted = np.array([int(pred.loc[name, "predicted"]) for name in sub.name])
    tb_ok = np.flatnonzero((sub.label == 1) & (predicted == 1))
    normal_ok = np.flatnonzero((sub.label == 0) & (predicted == 0))
    wrong = np.flatnonzero(sub.label.to_numpy() != predicted)
    picks = [("TB acertado", tb_ok[0] if len(tb_ok) else None),
             ("Normal acertado", normal_ok[0] if len(normal_ok) else None),
             ("Error", wrong[0] if len(wrong) else None)]
    picks = [(tag, i) for tag, i in picks if i is not None]
    if not picks:
        return
    files = {}
    for path in data_root.rglob("*.png"):
        if path.name in set(sub.name) and path.parent.name in ("CXR_png", "leftMask", "rightMask"):
            files[(path.parent.name, path.name)] = path
    size = 224
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, layer = model_for("densenet121")
    model.load_state_dict(torch.load(results / "densenet121_s0.pt", map_location="cpu", weights_only=True))
    model = model.to(device).eval()
    mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def predict_fn(batch):
        x = torch.from_numpy(np.asarray(batch, dtype=np.float32) / 255).permute(0, 3, 1, 2).to(device)
        return F.softmax(model((x - mean) / std), 1).cpu().numpy()

    cam = GradCAM(model=model, target_layers=[layer])
    explainer = lime_image.LimeImageExplainer(random_state=0)
    segment = SegmentationAlgorithm("slic", n_segments=60, compactness=10, sigma=1, start_label=1)
    fig, axs = plt.subplots(len(picks), 3, figsize=(6.6, 2.25 * len(picks)))
    axs = np.atleast_2d(axs)
    for row, (tag, idx) in enumerate(picks):
        name = sub.name[idx]
        image = np.asarray(Image.open(files[("CXR_png", name)]).convert("L").resize((size, size), Image.Resampling.BILINEAR))
        left = np.asarray(Image.open(files[("leftMask", name)]).convert("L").resize((size, size), Image.Resampling.NEAREST)) > 0
        right = np.asarray(Image.open(files[("rightMask", name)]).convert("L").resize((size, size), Image.Resampling.NEAREST)) > 0
        lung = left | right
        rgb = np.repeat(image[:, :, None], 3, axis=2).astype(np.uint8)
        x = torch.from_numpy(((rgb.astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1)).unsqueeze(0).to(device)
        target = ClassifierOutputTarget(int(predicted[idx]))
        grad = cam(input_tensor=x, targets=[target])[0]
        exp = explainer.explain_instance(rgb, predict_fn, top_labels=2, hide_color=0,
                                         num_samples=800, batch_size=8, segmentation_fn=segment)
        weights = dict(exp.local_exp[int(predicted[idx])])
        lime = np.zeros(exp.segments.shape, np.float32)
        for segment_id in np.unique(exp.segments):
            lime[exp.segments == segment_id] = max(weights.get(segment_id, 0.0), 0.0)
        if lime.max() > 0:
            lime /= lime.max()
        for col, heatmap, title in ((0, None, tag), (1, grad, "Grad-CAM"), (2, lime, "LIME")):
            ax = axs[row, col]
            ax.imshow(image, cmap="gray")
            if heatmap is not None:
                ax.imshow(heatmap, cmap="jet", alpha=np.clip(heatmap, 0, 1) * .55, vmin=0, vmax=1)
            ax.contour(lung, levels=[.5], colors="white" if col else "yellow", linewidths=.7)
            subtitle = f"real={'TB' if sub.label[idx] else 'N'}, pred={'TB' if predicted[idx] else 'N'}" if col == 0 else ""
            ax.set_title(title + ("\n" + subtitle if subtitle else ""), fontsize=8)
            ax.axis("off")
    fig.tight_layout(pad=.5)
    fig.savefig(results / "fig_cam_examples.png", dpi=220)
    plt.close(fig)
    cam.activations_and_grads.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--results", default="results")
    args = parser.parse_args()
    run(args.data_root, args.results)
