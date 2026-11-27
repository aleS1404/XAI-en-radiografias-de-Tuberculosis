"""Summarize post hoc geometry sensitivity without changing the primary test."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import matched_lesion as matched

CONFIGS = (("principal", 5.0, 12.0, Path(".")),
           ("borde_10", 10.0, 12.0, Path("edge10_separation12")),
           ("borde_10_centros_6", 10.0, 6.0, Path("relaxed_geometry")))


def run(data_root, results, annotations):
    results = Path(results)
    ann = pd.read_csv(annotations)
    files = matched.image_files(data_root)
    lungs_by_name = {name: matched.read_case(name, files)[1]
                     for name in sorted(ann.patientId.unique())}
    rows = []
    for label, edge_tol, separation, subdir in CONFIGS:
        matched.EDGE_TOLERANCE_PX = edge_tol
        matched.CONTROL_CENTER_SEPARATION_PX = separation
        folder = results / subdir
        summary = json.loads((folder / "matched_summary.json").read_text(encoding="utf-8"))
        edge_gaps, overlaps = [], []
        for name, lungs in lungs_by_name.items():
            region, _ = matched.select_regions(ann[ann.patientId == name], lungs)
            if region is None:
                continue
            edge_gaps.append(float(np.mean(np.abs(np.asarray(region["control_edge_px"]) -
                                                  region["index_edge_px"]))))
            controls = region["controls"]
            overlaps.append(max(float((controls[i] & controls[j]).sum() /
                                      (controls[i] | controls[j]).sum())
                                for i in range(len(controls)) for j in range(i + 1, len(controls))))
        for model in matched.MODELS:
            test = summary["models"][model][str(matched.BLUR_SIGMA)]
            rows.append(dict(protocol=label, status="principal" if label == "principal" else "sensibilidad_poshoc",
                             model=model, n=test["n"], median_D=test["median_ci"][0],
                             ci_low=test["median_ci"][1], ci_high=test["median_ci"][2],
                             wilcoxon_p=test["p"], sign_p=test["sign_p"],
                             positive=test["positive"], median_max_control_iou=float(np.median(overlaps)),
                             median_mean_edge_gap_px=float(np.median(edge_gaps)),
                             edge_tolerance_px=edge_tol, center_separation_px=separation))
    frame = pd.DataFrame(rows)
    frame.to_csv(results / "geometry_sensitivity.csv", index=False)
    table = [r"\begin{tabular}{lccccccc}", r"\toprule",
             r"Reglas (borde/centros) & $n$ & DenseNet121: $D$ [IC] & $p$ & ResNet50: $D$ [IC] & $p$ & IoU máx. med. & Brecha borde med.\\",
             r"\midrule"]
    for label, edge_tol, separation, _ in CONFIGS:
        group = frame[frame.protocol == label].set_index("model")
        dense, res = group.loc["densenet121"], group.loc["resnet50"]
        value = lambda item: f"{item.median_D:.3f} [{item.ci_low:.3f},{item.ci_high:.3f}]"
        table.append(f"{edge_tol:g}/{separation:g} & {dense.n} & {value(dense)} & {dense.wilcoxon_p:.5f} & "
                     f"{value(res)} & {res.wilcoxon_p:.5f} & {dense.median_max_control_iou:.3f} & "
                     f"{dense.median_mean_edge_gap_px:.2f} px" + r"\\")
    table += [r"\bottomrule", r"\end{tabular}"]
    (results / "tab_geometry_sensitivity.tex").write_text("\n".join(table), encoding="utf-8")
    return frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--results", default="results")
    parser.add_argument("--annotations", default="data/montgomery_consensus_roi.csv")
    args = parser.parse_args()
    print(run(args.data_root, args.results, args.annotations).to_string(index=False))
