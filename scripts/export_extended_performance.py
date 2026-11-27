"""Export per-seed classification measures from saved radiograph predictions."""

import argparse
from pathlib import Path
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, matthews_corrcoef, roc_auc_score


def run(results):
    results = Path(results)
    predictions = pd.read_csv(results / "predictions.csv")
    rows = []
    for (model, seed, dataset), part in predictions.groupby(["model", "seed", "dataset"]):
        y, prob, pred = part.label.to_numpy(), part.probability_tb.to_numpy(), part.predicted.to_numpy()
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        rows.append(dict(model=model, seed=int(seed), dataset=dataset, n=len(part),
                         prevalence=float(y.mean()), threshold=float(part.threshold.iloc[0]),
                         ROC_AUC=float(roc_auc_score(y, prob)), AP=float(average_precision_score(y, prob)),
                         sensitivity=float(tp / (tp + fn)), specificity=float(tn / (tn + fp)),
                         F1=float(2 * tp / (2 * tp + fp + fn)), MCC=float(matthews_corrcoef(y, pred)),
                         Brier=float(brier_score_loss(y, prob)),
                         TN=int(tn), FP=int(fp), FN=int(fn), TP=int(tp)))
    output = pd.DataFrame(rows).sort_values(["model", "seed", "dataset"])
    output.to_csv(results / "performance_extended.csv", index=False)
    selected = output[output.seed == 0]
    tex = [r"\begin{tabular}{llcccccc}", r"\toprule",
           r"Modelo & Prueba & AUC & AP & F1 & MCC & Brier & TN/FP/FN/TP\\", r"\midrule"]
    for model in ("densenet121", "resnet50"):
        for dataset, label in (("sz", "Shenzhen"), ("mg", "Montgomery")):
            row = selected[(selected.model == model) & (selected.dataset == dataset)].iloc[0]
            tex.append(f"{'DenseNet121' if model == 'densenet121' else 'ResNet50'} & {label} & "
                       f"{row.ROC_AUC:.3f} & {row.AP:.3f} & {row.F1:.3f} & {row.MCC:.3f} & "
                       f"{row.Brier:.3f} & {int(row.TN)}/{int(row.FP)}/{int(row.FN)}/{int(row.TP)}" + r"\\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    (results / "tab_perf_extended.tex").write_text("\n".join(tex), encoding="utf-8")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results")
    args = parser.parse_args()
    print(run(args.results).to_string(index=False))
