"""Patient-level uncertainty analysis of the saved experiment predictions."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def metrics(y, p, threshold):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    predicted = p >= threshold
    tp = int(np.sum((y == 1) & predicted))
    tn = int(np.sum((y == 0) & ~predicted))
    fp = int(np.sum((y == 0) & predicted))
    fn = int(np.sum((y == 1) & ~predicted))
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if precision + sensitivity else 0.0
    return dict(auc=float(roc_auc_score(y, p)), sensitivity=sensitivity,
                specificity=specificity, accuracy=(tp + tn) / len(y), f1=f1,
                brier=float(np.mean((p-y)**2)), tp=tp, tn=tn, fp=fp, fn=fn)


def stratified_indices(y, rng):
    groups = [np.flatnonzero(np.asarray(y) == label) for label in (0, 1)]
    return np.concatenate([rng.choice(group, len(group), replace=True) for group in groups])


def interval(values):
    values = np.asarray(values, dtype=float)
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def analyze(predictions, n_resamples=3000):
    primary = predictions[predictions.seed == predictions.seed.min()].copy()
    rng = np.random.default_rng(20261124)
    result = {"primary_seed": int(primary.seed.min()), "n_resamples": n_resamples,
              "models": {}, "paired_auc_difference_dense_minus_res": {}}
    for model in sorted(primary.model.unique()):
        result["models"][model] = {}
        for dataset in ("sz", "mg"):
            frame = primary[(primary.model == model) & (primary.dataset == dataset)].sort_values("name")
            y, p = frame.label.to_numpy(), frame.probability_tb.to_numpy()
            threshold = float(frame.threshold.iloc[0])
            point = metrics(y, p, threshold)
            boot = {key: [] for key in ("auc", "sensitivity", "specificity", "accuracy", "f1", "brier")}
            for _ in range(n_resamples):
                idx = stratified_indices(y, rng)
                draw = metrics(y[idx], p[idx], threshold)
                for key in boot:
                    boot[key].append(draw[key])
            result["models"][model][dataset] = {
                "n": len(frame), "positive": int(y.sum()), "threshold": threshold,
                "point": point, "ci95": {key: interval(values) for key, values in boot.items()}}
        sh = primary[(primary.model == model) & (primary.dataset == "sz")].sort_values("name")
        mg = primary[(primary.model == model) & (primary.dataset == "mg")].sort_values("name")
        gap = []
        for _ in range(n_resamples):
            isz, img = stratified_indices(sh.label, rng), stratified_indices(mg.label, rng)
            gap.append(roc_auc_score(mg.label.to_numpy()[img], mg.probability_tb.to_numpy()[img]) -
                       roc_auc_score(sh.label.to_numpy()[isz], sh.probability_tb.to_numpy()[isz]))
        result["models"][model]["external_minus_internal_auc"] = {
            "point": result["models"][model]["mg"]["point"]["auc"] - result["models"][model]["sz"]["point"]["auc"],
            "ci95": interval(gap)}
    for dataset in ("sz", "mg"):
        wide = primary[primary.dataset == dataset].pivot(index="name", columns="model", values="probability_tb")
        labels = primary[(primary.dataset == dataset) & (primary.model == "densenet121")].set_index("name").label
        wide = wide.join(labels).sort_index()
        if wide.isna().any().any():
            raise ValueError("Faltan predicciones pareadas")
        y = wide.label.to_numpy()
        dense, res = wide.densenet121.to_numpy(), wide.resnet50.to_numpy()
        differences = []
        for _ in range(n_resamples):
            idx = stratified_indices(y, rng)
            differences.append(roc_auc_score(y[idx], dense[idx]) - roc_auc_score(y[idx], res[idx]))
        result["paired_auc_difference_dense_minus_res"][dataset] = {
            "point": float(roc_auc_score(y, dense) - roc_auc_score(y, res)), "ci95": interval(differences)}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args()
    predictions = pd.read_csv(args.results / "predictions.csv")
    report = analyze(predictions)
    (args.results / "performance_ci.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
