"""Exploratory localization against NLM consensus TB finding boxes.

The unit of uncertainty is an image. Maps from repeated training seeds are
averaged per image before confidence intervals and paired comparisons.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def boxes_to_mask(rows, size, original_size=1024):
    """Union rectangular consensus regions in the notebook's square image grid."""
    mask = np.zeros((size, size), dtype=bool)
    for row in rows.itertuples(index=False):
        x0 = max(0, min(size, int(np.floor(row.x_dis * size / original_size))))
        y0 = max(0, min(size, int(np.floor(row.y_dis * size / original_size))))
        x1 = max(0, min(size, int(np.ceil((row.x_dis + row.width_dis) * size / original_size))))
        y1 = max(0, min(size, int(np.ceil((row.y_dis + row.height_dis) * size / original_size))))
        mask[y0:y1, x0:x1] = True
    return mask


def roi_scores(heatmap, lesion_mask, lung_mask):
    h = np.clip(np.asarray(heatmap, dtype=np.float64), 0, None)
    lesion = np.asarray(lesion_mask, dtype=bool)
    lung = np.asarray(lung_mask, dtype=bool)
    if h.shape != lesion.shape or h.shape != lung.shape:
        raise ValueError("Los mapas y las máscaras tienen dimensiones distintas")
    if h.sum() <= 0 or not lesion.any() or not lung.any():
        return dict(gain_image=np.nan, gain_lung=np.nan, point_lung=np.nan,
                    lesion_area_image=float(lesion.mean()), lesion_area_lung=np.nan,
                    roi_inside_lung_fraction=np.nan)
    gain_image = (h[lesion].sum() / h.sum()) / lesion.mean()
    lesion_in_lung = lesion & lung
    area_lung = lesion_in_lung.sum() / lung.sum()
    if h[lung].sum() <= 0 or area_lung <= 0:
        gain_lung = point_lung = np.nan
    else:
        gain_lung = (h[lesion_in_lung].sum() / h[lung].sum()) / area_lung
        candidate = np.where(lung, h, -np.inf)
        point_lung = float(lesion_in_lung.flat[int(candidate.argmax())])
    return dict(gain_image=float(gain_image), gain_lung=float(gain_lung),
                point_lung=float(point_lung), lesion_area_image=float(lesion.mean()),
                lesion_area_lung=float(area_lung),
                roi_inside_lung_fraction=float(lesion_in_lung.sum() / lesion.sum()))


def bootstrap_mean(values, seed=371, n_resamples=3000):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return [None, None, None]
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(n_resamples, len(values)))].mean(axis=1)
    return [float(values.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def run_lesion_analysis(*, csv_path, mg, mask_cache, get_tensor,
                        load_trained, gradcam_maps, random_maps, class_probs,
                        model_names, seeds, size, out_dir):
    csv_path, out_dir = Path(csv_path), Path(out_dir)
    annotations = pd.read_csv(csv_path)
    expected = {"patientId", "x_dis", "y_dis", "width_dis", "height_dis", "Labels"}
    if not expected.issubset(annotations.columns):
        raise ValueError("Columnas inesperadas en las anotaciones NLM")
    if set(annotations.Labels) != {"TB"}:
        raise ValueError("Las anotaciones contienen etiquetas inesperadas")
    frame = mg[mg.name.isin(annotations.patientId)].reset_index(drop=True)
    if len(frame) != 55 or not (frame.label == 1).all():
        raise ValueError(f"Se esperaban 55 radiografías positivas anotadas, hay {len(frame)}")
    if not all(name in mask_cache for name in frame.name):
        raise ValueError("Faltan máscaras pulmonares para el análisis de lesiones")
    masks = {name: boxes_to_mask(annotations[annotations.patientId == name], size)
             for name in frame.name}
    if not all(mask.any() for mask in masks.values()):
        raise ValueError("Hay una máscara de lesión vacía")
    X = get_tensor(frame)
    targets = np.ones(len(frame), dtype=int)  # Misma clase TB aun en falsos negativos.
    rows = []
    for model_name in model_names:
        for seed in seeds:
            model = load_trained(model_name, seed)
            probabilities = class_probs(model, X)[:, 1]
            maps = gradcam_maps(model, model_name, X, targets)
            controls = random_maps(len(frame), seed=seed + 919)
            for i, item in frame.iterrows():
                for method, heatmap in (("Grad-CAM", maps[i]), ("Aleatorio", controls[i])):
                    metrics = roi_scores(heatmap, masks[item["name"]], mask_cache[item["name"]])
                    rows.append(dict(model=model_name, seed=int(seed), method=method,
                                     name=item["name"], probability_tb=float(probabilities[i]),
                                     **metrics))
            del model
    scores = pd.DataFrame(rows)
    scores.to_csv(out_dir / "lesion_roi_scores.csv", index=False)
    per_image = scores.groupby(["model", "method", "name"], as_index=False).agg(
        gain_image=("gain_image", "mean"), gain_lung=("gain_lung", "mean"),
        point_lung=("point_lung", "mean"), lesion_area_lung=("lesion_area_lung", "mean"),
        roi_inside_lung_fraction=("roi_inside_lung_fraction", "mean"))
    n_misaligned = sum((masks[name] & mask_cache[name]).sum() / masks[name].sum() < .5
                       for name in frame.name)
    result = {"n_images": len(frame), "n_boxes": len(annotations),
              "n_roi_majority_outside_lung": int(n_misaligned), "models": {}}
    for model_name in model_names:
        model_result = {}
        for method in ("Grad-CAM", "Aleatorio"):
            part = per_image[(per_image.model == model_name) & (per_image.method == method)]
            raw_part = scores[(scores.model == model_name) & (scores.method == method)]
            zero_included = raw_part.assign(gain_lung=raw_part.gain_lung.fillna(0)).groupby("name").gain_lung.mean()
            model_result[method] = {
                "gain_lung": bootstrap_mean(part.gain_lung),
                "gain_lung_null_maps_as_zero": bootstrap_mean(zero_included),
                "gain_image": bootstrap_mean(part.gain_image),
                "point_lung": bootstrap_mean(part.point_lung),
                "mean_roi_lung_area": float(part.lesion_area_lung.mean()),
                "valid_images": int(part.gain_lung.notna().sum()),
                "valid_maps": int(raw_part.gain_lung.notna().sum()),
                "total_maps": int(len(raw_part)),
            }
        wide = per_image[per_image.model == model_name].pivot(index="name", columns="method", values="gain_lung")
        differences = (wide["Grad-CAM"] - wide["Aleatorio"]).dropna()
        model_result["paired_gain_lung_difference"] = bootstrap_mean(differences)
        model_result["wilcoxon_p"] = float(stats.wilcoxon(differences).pvalue) if len(differences) > 5 else None
        valid_names = per_image[(per_image.model == model_name) & (per_image.method == "Grad-CAM") &
                                (per_image.roi_inside_lung_fraction >= .5)].name
        model_result["gain_lung_excluding_misaligned_roi"] = bootstrap_mean(
            per_image[(per_image.model == model_name) & (per_image.method == "Grad-CAM") &
                      (per_image.name.isin(valid_names))].gain_lung)
        result["models"][model_name] = model_result
    (out_dir / "lesion_roi_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    fig, axs = plt.subplots(1, len(model_names), figsize=(7.4, 3.5), sharey=True)
    axs = np.atleast_1d(axs)
    for ax, model_name in zip(axs, model_names):
        values = [per_image[(per_image.model == model_name) & (per_image.method == method)].gain_lung.dropna()
                  for method in ("Grad-CAM", "Aleatorio")]
        ax.boxplot(values, tick_labels=["Grad-CAM", "Aleatorio"], showmeans=True)
        ax.axhline(1, color="gray", linestyle="--", linewidth=1)
        ax.set_title("DenseNet121" if model_name == "densenet121" else "ResNet50")
        ax.grid(alpha=.25)
    axs[0].set_ylabel("Ganancia en lesión / azar")
    fig.tight_layout(pad=0.8)
    fig.savefig(out_dir / "fig_lesion_roi.png", dpi=220)
    plt.close(fig)
    return result
