"""Exclude null heatmaps from deletion summaries and regenerate the XAI table."""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


MODEL_LABEL = {"densenet121": "DenseNet121", "resnet50": "ResNet50"}
MODEL_KEY = {"densenet121": "Dense", "resnet50": "Res"}
METHOD_KEY = {"Grad-CAM": "Gc", "LIME": "Lime", "Aleatorio": "Rnd"}


def summarize(values):
    values = pd.Series(values).dropna()
    if values.empty:
        return None
    return float(values.mean()), float(values.std(ddof=1) if len(values) > 1 else 0)


def latex(values, digits=2):
    pair = summarize(values)
    return "--" if pair is None else f"{pair[0]:.{digits}f}$\\pm${pair[1]:.{digits}f}"


def render_row(model, method, frame):
    n = int(frame.border.notna().sum())
    point = summarize(frame.point)
    fields = [MODEL_LABEL[model], method, str(n), latex(frame.eil, 3), latex(frame.gain),
              "--" if point is None else f"{point[0]:.2f}", latex(frame.border), latex(frame.deletion, 3)]
    return " & ".join(fields) + " " + chr(92) * 2


def run(results):
    results = Path(results)
    subset_path = results / "xai_subset_scores.csv"
    subset = pd.read_csv(subset_path)
    full = pd.read_csv(results / "xai_full_scores.csv")
    if "valid" not in subset.columns or "valid" not in full.columns:
        raise ValueError("Falta la columna de validez de mapas")
    subset.loc[~subset.valid.astype(bool), "deletion"] = np.nan
    subset.to_csv(subset_path, index=False)
    validity = []
    for model in MODEL_LABEL:
        for method in METHOD_KEY:
            part = subset[(subset.model == model) & (subset.method == method)]
            validity.append(dict(model=model, method=method, valid=int(part.valid.sum()), total=len(part)))
        for dataset in ("mg", "sz"):
            part = full[(full.model == model) & (full["set"] == dataset)]
            validity.append(dict(model=model, method="Grad-CAM " + dataset,
                                 valid=int(part.valid.sum()), total=len(part)))
    pd.DataFrame(validity).to_csv(results / "map_validity.csv", index=False)
    lines = ["\\begin{tabular}{llcccccc}", "\\toprule",
             "Modelo & Método & $n$ & EIL & Ganancia & Punt. & Borde & Del.$\\downarrow$ " + chr(92) * 2,
             "\\midrule"]
    macro_path = results / "macros.tex"
    try:
        macros = macro_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # The initial Windows notebook run used the platform's cp1252 default.
        macros = macro_path.read_text(encoding="cp1252")
    perf_table = results / "tab_perf.tex"
    try:
        perf_table.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        perf_table.write_text(perf_table.read_text(encoding="cp1252"), encoding="utf-8")
    for model in MODEL_LABEL:
        for method in METHOD_KEY:
            part = subset[(subset.model == model) & (subset.method == method)]
            lines.append(render_row(model, method, part))
            macro_name = f"xai{MODEL_KEY[model]}{METHOD_KEY[method]}Del"
            macro_value = latex(part.deletion, 3)
            pattern = re.compile(r"(\\def\\" + re.escape(macro_name) + r"\{)[^}]*\}")
            if pattern.search(macros):
                macros = pattern.sub(lambda match: match.group(1) + macro_value + "}", macros)
            else:
                macros += f"\\def\\{macro_name}{{{macro_value}}}\n"
        for dataset, method in (("mg", "Grad-CAM (Mont. completo)"), ("sz", "Grad-CAM (Shen. test)")):
            part = full[(full.model == model) & (full["set"] == dataset)].copy()
            part["deletion"] = np.nan
            lines.append(render_row(model, method, part))
        if model != "resnet50":
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (results / "tab_xai.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    macro_path.write_text(macros, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args()
    run(args.results)
    print("XAI finalizado:", args.results)
