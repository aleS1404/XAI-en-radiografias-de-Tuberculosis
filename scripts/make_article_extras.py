"""Render supplementary, traceable LaTeX figures from saved analysis JSON."""

import argparse
import json
from pathlib import Path


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def ci(values, digits=2):
    if values is None or any(value is None for value in values):
        return "--"
    return f"{values[0]:.{digits}f} [{values[1]:.{digits}f}, {values[2]:.{digits}f}]"


def point_ci(item, digits=2):
    return ci([item["point"], *item["ci95"]], digits)


def make(results):
    results = Path(results)
    perf = read_json(results / "performance_ci.json")
    lesion = read_json(results / "lesion_roi_summary.json")
    shenzhen = read_json(results / "shenzhen_test_roi_summary.json")
    sanity = read_json(results / "sanity_maps_summary.json")
    occlusion = read_json(results / "occlusion_summary.json")
    macros = {}
    model_info = (("densenet121", "Dense", "DenseNet121"), ("resnet50", "Res", "ResNet50"))
    for name, short, label in model_info:
        p = perf["models"][name]
        l = lesion["models"][name]
        s = shenzhen["models"][name]
        a = sanity["models"][name]
        o = occlusion["models"][name]
        macros[f"ext{short}AucCi"] = point_ci({"point": p["mg"]["point"]["auc"], "ci95": p["mg"]["ci95"]["auc"]}, 3)
        macros[f"ext{short}SensCi"] = point_ci({"point": p["mg"]["point"]["sensitivity"], "ci95": p["mg"]["ci95"]["sensitivity"]}, 2)
        macros[f"gap{short}AucCi"] = point_ci(p["external_minus_internal_auc"], 3)
        macros[f"lesion{short}Gain"] = ci(l["Grad-CAM"]["gain_lung"])
        macros[f"lesion{short}NullZero"] = ci(l["Grad-CAM"]["gain_lung_null_maps_as_zero"])
        macros[f"lesion{short}Aligned"] = ci(l["gain_lung_excluding_misaligned_roi"])
        macros[f"lesion{short}Delta"] = ci(l["paired_gain_lung_difference"])
        macros[f"lesion{short}Valid"] = f"{l['Grad-CAM']['valid_maps']}/{l['Grad-CAM']['total_maps']}"
        macros[f"lesion{short}Point"] = ci(l["Grad-CAM"]["point_lung"])
        macros[f"shenzhen{short}RoiGain"] = ci(s["Grad-CAM"]["gain_image"])
        macros[f"shenzhen{short}RoiValid"] = f"{s['Grad-CAM']['valid_images']}/{shenzhen['n_test_images']}"
        rho = a["median_map_spearman"]
        macros[f"sanity{short}Rho"] = "--" if rho is None else f"{rho:.2f}"
        macros[f"sanity{short}Pairs"] = str(a["valid_map_pairs"])
        macros[f"occl{short}Border"] = ci(o["border_drop"], 3)
        macros[f"occl{short}Lung"] = ci(o["matched_lung_drop"], 3)
        macros[f"occl{short}Delta"] = ci(o["border_minus_matched_lung"], 3)
        macros[f"occl{short}FullLung"] = ci(o["full_lung_drop"], 3)
    macros["pairedExternalAucDifference"] = point_ci(perf["paired_auc_difference_dense_minus_res"]["mg"], 3)
    macros["lesionRoiN"] = str(lesion["n_images"])
    macros["shenzhenRoiN"] = str(shenzhen["n_test_images"])
    macros["shenzhenRandomRoiGain"] = ci(shenzhen["models"]["densenet121"]["Aleatorio"]["gain_image"])
    macros["occlBorderArea"] = f"{occlusion['border_area_fraction']:.3f}"
    (results / "extra_macros.tex").write_text(
        "% Generado por scripts/make_article_extras.py\n" +
        "\n".join(f"\\def\\{key}{{{value}}}" for key, value in macros.items()) + "\n",
        encoding="utf-8")
    lines = ["\\begin{tabular}{lccccc}", "\\toprule",
             "Modelo & Mapas válidos & Ganancia Grad-CAM & Ganancia aleatoria & Diferencia & Puntería \\\\",
             "\\midrule"]
    for name, short, label in model_info:
        l = lesion["models"][name]
        lines.append(f"{label} & {macros[f'lesion{short}Valid']} & {ci(l['Grad-CAM']['gain_lung'])} & "
                     f"{ci(l['Aleatorio']['gain_lung'])} & {ci(l['paired_gain_lung_difference'])} & "
                     f"{ci(l['Grad-CAM']['point_lung'])} " + chr(92) * 2)
    lines += ["\\bottomrule", "\\end{tabular}"]
    (results / "tab_lesion.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return macros


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args()
    for name, value in make(args.results).items():
        print(f"{name}: {value}")
