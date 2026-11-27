"""Retrospective, geometry-locked matched intrapulmonary lesion perturbation.

Eligibility is computed without loading model outputs. The independent unit is
the radiograph; seeds and control regions are averaged within that unit.
"""

import argparse
import hashlib
import json
import platform
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats
from scipy.integrate import trapezoid
import scipy
import torch
import torch.nn.functional as F
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from scripts.lesion_roi import boxes_to_mask
from scripts.shenzhen_roi import MEAN, STD, model_for

SIZE = 224
MIN_BOX_COVERAGE = 0.50
MIN_PIXELS = 25
EDGE_TOLERANCE_PX = 5.0
MIN_CONTROLS = 3
MAX_CONTROLS = 5
CONTROL_CENTER_SEPARATION_PX = 12
BLUR_SIGMA = 10.0
ALT_BLUR_SIGMA = 5.0
SEEDS = (0, 1, 2)
MODELS = ("densenet121", "resnet50")


def image_files(root):
    kinds = ("CXR_png", "leftMask", "rightMask")
    found = {}
    for p in Path(root).rglob("*.png"):
        if p.parent.name in kinds and p.name.startswith("MCUCXR"):
            key = (p.parent.name, p.name)
            if key in found:
                raise ValueError(f"Archivo duplicado: {key}")
            found[key] = p
    return found


def read_case(name, files):
    def gray(kind, method):
        return np.asarray(Image.open(files[(kind, name)]).convert("L").resize((SIZE, SIZE), method))
    image = gray("CXR_png", Image.Resampling.BILINEAR).astype(np.float32) / 255.0
    left = gray("leftMask", Image.Resampling.NEAREST) > 0
    right = gray("rightMask", Image.Resampling.NEAREST) > 0
    if np.any(left & right):
        # The released masks occasionally touch after resizing; assignments
        # remain deterministic because the index is selected by greater area.
        right = right & ~left
    return image, (left, right)


def select_regions(rows, lungs):
    """Return index and translated same-shape controls, or an exclusion reason."""
    union_boxes = boxes_to_mask(rows, SIZE)
    candidates = []
    for box_number, (_, row) in enumerate(rows.iterrows()):
        box = boxes_to_mask(pd.DataFrame([row]), SIZE)
        for side, lung in enumerate(lungs):
            index = box & lung
            area = int(index.sum())
            coverage = area / max(1, int(box.sum()))
            if area >= MIN_PIXELS and coverage >= MIN_BOX_COVERAGE:
                candidates.append((-area, side, box_number, index, box, coverage))
    if not candidates:
        return None, "sin_caja_con_cobertura_y_area_minima"
    _, side, box_number, index, box, coverage = min(candidates, key=lambda t: t[:3])
    lung = lungs[side]
    yy, xx = np.where(index)
    y0, y1, x0, x1 = yy.min(), yy.max()+1, xx.min(), xx.max()+1
    template = index[y0:y1, x0:x1].astype(np.uint8)
    th, tw = template.shape
    area = int(template.sum())
    centroid_y, centroid_x = float(yy.mean()), float(xx.mean())
    third = min(2, int(3 * centroid_y / SIZE))
    distance = cv2.distanceTransform(lung.astype(np.uint8), cv2.DIST_L2, 5)
    index_edge = float(np.mean(distance[index]))
    allowed = (lung & ~union_boxes).astype(np.uint8)
    # MatchTemplate counts eligible pixels under the *same digital shape*.
    counts = cv2.matchTemplate(allowed, template, cv2.TM_CCORR)
    positions = np.argwhere(counts >= area - 0.25)
    ranked = []
    local_y, local_x = np.where(template)
    for top, left in positions:
        cy = float(top + local_y.mean())
        cx = float(left + local_x.mean())
        if min(2, int(3 * cy / SIZE)) != third:
            continue
        edge = float(np.mean(distance[top+local_y, left+local_x]))
        edge_gap = abs(edge - index_edge)
        if edge_gap > EDGE_TOLERANCE_PX:
            continue
        # Fixed geometry-only ranking; ties resolve by y then x.
        cost = (edge_gap, abs(cy-centroid_y), abs(cx-centroid_x), int(top), int(left))
        ranked.append((cost, int(top), int(left), cy, cx, edge))
    ranked.sort(key=lambda t: t[0])
    controls = []
    for _, top, left, cy, cx, edge in ranked:
        if any((cy-q[1])**2 + (cx-q[2])**2 < CONTROL_CENTER_SEPARATION_PX**2 for q in controls):
            continue
        mask = np.zeros((SIZE, SIZE), bool)
        mask[top+local_y, left+local_x] = True
        assert int(mask.sum()) == area and not np.any(mask & union_boxes)
        assert np.all(lung[mask])
        controls.append((mask, cy, cx, edge))
        if len(controls) == MAX_CONTROLS:
            break
    if len(controls) < MIN_CONTROLS:
        return None, f"controles_insuficientes_{len(controls)}"
    return dict(index=index, controls=[v[0] for v in controls], side=side,
                box_number=box_number, box_coverage=coverage, index_area=area,
                centroid_y=centroid_y, centroid_x=centroid_x, third=third,
                index_edge_px=index_edge,
                control_edge_px=[v[3] for v in controls],
                control_centroids=[[v[1], v[2]] for v in controls],
                all_boxes=union_boxes, lung=lung), None


def perturb(image, mask, sigma):
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma)
    result = image.copy()
    result[mask] = blurred[mask]
    return result


def tensor(images, device):
    rgb = np.repeat(np.stack(images)[:, :, :, None], 3, axis=3).astype(np.float32)
    return torch.from_numpy(((rgb - MEAN) / STD).transpose(0, 3, 1, 2)).to(device)


def cam_iou_top10(heat, lung, index):
    """IoU with the top 10% lung pixels; flat index resolves map ties."""
    positions = np.flatnonzero(lung)
    count = max(1, int(np.ceil(.10 * len(positions))))
    ranked = positions[np.lexsort((positions, -heat.ravel()[positions]))]
    top = np.zeros(heat.size, bool)
    top[ranked[:count]] = True
    top = top.reshape(lung.shape)
    return float((top & index).sum() / (top | index).sum())


def fidelity_curves(model, image, heat, lung, device):
    """Deletion/insertion of ranked lung pixels using one blurred background."""
    fractions = np.linspace(0, .5, 6)
    positions = np.flatnonzero(lung)
    ranked = positions[np.lexsort((positions, -heat.ravel()[positions]))]
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=BLUR_SIGMA)
    full_background = image.copy()
    full_background[lung] = blurred[lung]
    deletion, insertion = [], []
    for fraction in fractions:
        chosen = ranked[:int(np.floor(fraction * len(positions)))]
        d = image.copy()
        i = full_background.copy()
        d.ravel()[chosen] = blurred.ravel()[chosen]
        i.ravel()[chosen] = image.ravel()[chosen]
        deletion.append(d)
        insertion.append(i)
    variants = deletion + insertion
    probs = []
    with torch.no_grad():
        for start in range(0, len(variants), 8):
            p = F.softmax(model(tensor(variants[start:start+8], device)).float(), 1)[:, 1]
            probs.extend(p.cpu().numpy().tolist())
    d, i = np.asarray(probs[:len(fractions)]), np.asarray(probs[len(fractions):])
    area = fractions[-1] - fractions[0]
    return dict(deletion_auc=float(trapezoid(d, fractions) / area),
                insertion_auc=float(trapezoid(i, fractions) / area),
                deletion_curve=json.dumps(d.tolist()), insertion_curve=json.dumps(i.tolist()))


def bootstrap_median(values, seed=193, n=5000):
    a = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    b = np.median(a[rng.integers(len(a), size=(n, len(a)))], axis=1)
    return [float(np.median(a)), *map(float, np.quantile(b, [.025, .975]))]


def signed_rank(values):
    a = np.asarray(values, float)
    a = a[np.isfinite(a)]
    nonzero = a[a != 0]
    out = dict(n=int(len(a)), zeros=int((a == 0).sum()), positive=int((a > 0).sum()),
               negative=int((a < 0).sum()), median_ci=bootstrap_median(a) if len(a) else None,
               scipy=scipy.__version__, method="auto", zero_method="wilcox", alternative="two-sided",
               hypothesis="distribucion_de_D_simetrica_alrededor_de_cero")
    if not len(nonzero):
        out.update(statistic=None, p=None, sign_p=None, rank_biserial=None, status="sin_diferencias_no_nulas")
        return out
    test = stats.wilcoxon(a, zero_method="wilcox", correction=False, alternative="two-sided", method="auto")
    ranks = stats.rankdata(abs(nonzero), method="average")
    pos, neg = ranks[nonzero > 0].sum(), ranks[nonzero < 0].sum()
    out.update(statistic=float(test.statistic), p=float(test.pvalue),
               sign_p=float(stats.binomtest(int((nonzero > 0).sum()), len(nonzero), .5).pvalue),
               rank_biserial=float((pos-neg)/(pos+neg)), status="ok")
    return out


def run(data_root, results, annotations_path, geometry_only=False, output=None):
    data_root, results = Path(data_root), Path(results)
    out = Path(output) if output else results
    out.mkdir(parents=True, exist_ok=True)
    splits = pd.read_csv(results / "splits.csv")
    ann = pd.read_csv(annotations_path)
    required = {"patientId", "x_dis", "y_dis", "width_dis", "height_dis", "Labels"}
    if not required <= set(ann) or set(ann.Labels) != {"TB"}:
        raise ValueError("Formato de cajas NLM inesperado")
    names = sorted(set(ann.patientId))
    frame = splits[(splits.split == "external") & (splits.label == 1)]
    if set(names) - set(frame.name):
        raise ValueError("Cajas que no corresponden a positivos externos")
    files = image_files(data_root)
    manifest, cases = [], {}
    for name in names:
        image, lungs = read_case(name, files)
        region, reason = select_regions(ann[ann.patientId == name], lungs)
        row = dict(name=name, label=1, eligible=region is not None, reason=reason or "",
                   n_boxes=int((ann.patientId == name).sum()),
                   n_controls=len(region["controls"]) if region else 0,
                   side=("leftMask" if region["side"] == 0 else "rightMask") if region else "",
                   box_number=region["box_number"] if region else None,
                   box_coverage=region["box_coverage"] if region else None,
                   index_area=region["index_area"] if region else None,
                   index_edge_px=region["index_edge_px"] if region else None,
                   control_edge_px=json.dumps(region["control_edge_px"]) if region else "",
                   control_centroids=json.dumps(region["control_centroids"]) if region else "",
                   third=region["third"] if region else None)
        manifest.append(row)
        if region:
            assert np.array_equal(perturb(image, np.zeros_like(image, bool), BLUR_SIGMA), image)
            cases[name] = (image, region)
    pd.DataFrame(manifest).to_csv(out / "matched_eligibility.csv", index=False)
    protocol = dict(size=SIZE, min_box_coverage=MIN_BOX_COVERAGE, min_pixels=MIN_PIXELS,
                    edge_tolerance_px=EDGE_TOLERANCE_PX, min_controls=MIN_CONTROLS,
                    max_controls=MAX_CONTROLS, center_separation_px=CONTROL_CENTER_SEPARATION_PX,
                    blur_sigma=BLUR_SIGMA, alternative_blur_sigma=ALT_BLUR_SIGMA,
                    selection="largest box intersection with one lung; ties side then CSV order; controls geometry-only",
                    cam_iou_top_fraction=.10, map_tie_break="ascending flat pixel index",
                    fidelity_fractions=[0, .1, .2, .3, .4, .5],
                    fidelity_background="Gaussian sigma 10 inside lung; outside lung original",
                    map_baselines={"uniform": 1, "smoothed_random_repeats": 10,
                                   "random_grid": [7, 7], "fixed_center_sigma_px": .20*SIZE,
                                   "status": "exploratory diagnostic"},
                    n_annotated=len(names), n_eligible=len(cases),
                    annotation_sha256=hashlib.sha256(Path(annotations_path).read_bytes()).hexdigest(),
                    mode="geometry_only" if geometry_only else "full",
                    hardware=torch.cuda.get_device_name(0) if torch.cuda.is_available() else platform.processor())
    (out / "matched_protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    if geometry_only:
        return protocol
    if not cases:
        raise ValueError("No hay radiografías geométricamente analizables")
    # Descriptive map baselines; generated from geometry only, independent of
    # classifier outputs. Their use is diagnostic, not part of the primary test.
    baseline_rows = []
    grid_y, grid_x = np.mgrid[:SIZE, :SIZE]
    center_map = np.exp(-((grid_x-(SIZE-1)/2)**2 + (grid_y-(SIZE-1)/2)**2) /
                        (2*(.20*SIZE)**2))
    for name, (_, region) in cases.items():
        rng = np.random.default_rng(int(hashlib.sha256(name.encode()).hexdigest()[:8], 16))
        maps = [("Uniforme", 0, np.ones((SIZE, SIZE), float)),
                ("Centro fijo", 0, center_map)]
        maps += [("Aleatorio suavizado", j,
                  cv2.resize(rng.random((7, 7), dtype=np.float32), (SIZE, SIZE),
                             interpolation=cv2.INTER_LINEAR)) for j in range(10)]
        for method, repeat, h in maps:
            total = h[region["lung"]].sum()
            index_mass = h[region["index"]].sum() / total
            controls = [h[m].sum() / total for m in region["controls"]]
            baseline_rows.append(dict(name=name, method=method, repeat=repeat,
                                      index_mass=float(index_mass),
                                      control_median_mass=float(np.median(controls)),
                                      mass_D=float(index_mass-np.median(controls)),
                                      index_iou_top10=cam_iou_top10(h, region["lung"], region["index"])))
    pd.DataFrame(baseline_rows).to_csv(out / "matched_map_baselines.csv", index=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    example_name = next(iter(cases))
    example_heat = None
    for model_name in MODELS:
        for seed in SEEDS:
            model, layer = model_for(model_name)
            model.load_state_dict(torch.load(results / f"{model_name}_s{seed}.pt", map_location="cpu", weights_only=True))
            model = model.to(device).eval()
            cam = GradCAM(model=model, target_layers=[layer])
            for name, (image, region) in cases.items():
                masks = [region["index"], *region["controls"]]
                for sigma in (BLUR_SIGMA, ALT_BLUR_SIGMA):
                    variants = [image, *[perturb(image, m, sigma) for m in masks]]
                    with torch.no_grad():
                        logits = model(tensor(variants, device)).float()
                        score = (logits[:, 1] - logits[:, 0]).cpu().numpy()
                        probs = F.softmax(logits, 1)[:, 1].cpu().numpy()
                    drops = score[0] - score[1:]
                    prob_drops = probs[0] - probs[1:]
                    x = tensor([image], device)
                    heat = np.asarray(cam(input_tensor=x, targets=[ClassifierOutputTarget(1)])[0], float)
                    if model_name == MODELS[0] and seed == SEEDS[0] and name == example_name and sigma == BLUR_SIGMA:
                        example_heat = heat.copy()
                    valid_map = bool(np.isfinite(heat).all() and heat[region["lung"]].sum() > 0)
                    if valid_map:
                        h = np.clip(heat, 0, None)
                        lung_mass = h[region["lung"]].sum()
                        index_mass = h[masks[0]].sum() / lung_mass
                        control_mass = [h[m].sum() / lung_mass for m in masks[1:]]
                        iou10 = cam_iou_top10(h, region["lung"], region["index"])
                        fidelity = fidelity_curves(model, image, h, region["lung"], device) if sigma == BLUR_SIGMA else {}
                    else:
                        index_mass, control_mass = np.nan, [np.nan] * (len(masks)-1)
                        iou10, fidelity = np.nan, {}
                    rows.append(dict(model=model_name, seed=seed, name=name, sigma=sigma,
                                     original_logit=float(score[0]), original_probability=float(probs[0]),
                                     index_drop=float(drops[0]), control_drops=json.dumps(drops[1:].tolist()),
                                     D=float(drops[0]-np.median(drops[1:])),
                                     index_probability_drop=float(prob_drops[0]),
                                     control_probability_drops=json.dumps(prob_drops[1:].tolist()),
                                     probability_D=float(prob_drops[0]-np.median(prob_drops[1:])),
                                     valid_map=valid_map, index_cam_mass=float(index_mass),
                                     control_cam_mass=json.dumps(control_mass),
                                     cam_mass_D=float(index_mass-np.median(control_mass)),
                                     cam_iou_top10=float(iou10), **fidelity))
            cam.activations_and_grads.release()
            del cam, model
            if device == "cuda": torch.cuda.empty_cache()
    scores = pd.DataFrame(rows)
    scores.to_csv(out / "matched_scores.csv", index=False)
    per_image = scores.groupby(["model", "sigma", "name"], as_index=False).agg(
        D=("D", "mean"), probability_D=("probability_D", "mean"),
        cam_mass_D=("cam_mass_D", "mean"), valid_maps=("valid_map", "sum"))
    per_image.to_csv(out / "matched_per_image.csv", index=False)
    summary = dict(protocol=protocol, models={})
    for model_name in MODELS:
        summary["models"][model_name] = {}
        for sigma in (BLUR_SIGMA, ALT_BLUR_SIGMA):
            part = per_image[(per_image.model == model_name) & (per_image.sigma == sigma)]
            result = signed_rank(part.D)
            result["n_excluded_geometry"] = len(names) - len(part)
            result["probability_median_ci"] = bootstrap_median(part.probability_D)
            rho_data = part[["D", "cam_mass_D"]].dropna()
            if len(rho_data) >= 4 and rho_data.D.nunique() > 1 and rho_data.cam_mass_D.nunique() > 1:
                result["spearman_rho"] = float(stats.spearmanr(rho_data.D, rho_data.cam_mass_D).statistic)
                rng = np.random.default_rng(992)
                boot = []
                a = rho_data.to_numpy()
                for _ in range(3000):
                    sample = a[rng.integers(len(a), size=len(a))]
                    if len(np.unique(sample[:, 0])) > 1 and len(np.unique(sample[:, 1])) > 1:
                        boot.append(stats.spearmanr(sample[:, 0], sample[:, 1]).statistic)
                result["spearman_ci"] = list(map(float, np.quantile(boot, [.025, .975])))
            else:
                result["spearman_rho"] = result["spearman_ci"] = None
            summary["models"][model_name][str(sigma)] = result
    # ResNet50 is the prespecified secondary model contrast; Holm also covers
    # the alternative intensity as a sensitivity family, separate from primary.
    secondary = [(m, s) for m, s in (("resnet50", BLUR_SIGMA), ("densenet121", ALT_BLUR_SIGMA), ("resnet50", ALT_BLUR_SIGMA))]
    pvals = [summary["models"][m][str(s)]["p"] for m, s in secondary]
    order = np.argsort(pvals)
    adjusted = np.empty(len(pvals))
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(pvals)-rank) * pvals[idx])
        adjusted[idx] = min(1.0, running)
    for (m, s), q in zip(secondary, adjusted):
        summary["models"][m][str(s)]["holm_secondary_p"] = float(q)
    (out / "matched_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    table = [r"\begin{tabular}{lccccc}", r"\toprule",
             r"Modelo & $n$ & Mediana $D$ [IC] & $W$ & $p$ & $r_{rb}$\\", r"\midrule"]
    for model_name, label in (("densenet121", "DenseNet121"), ("resnet50", "ResNet50")):
        item = summary["models"][model_name][str(BLUR_SIGMA)]
        median, lo, hi = item["median_ci"]
        table.append(f"{label} & {item['n']} & {median:.3f} [{lo:.3f},{hi:.3f}] & "
                     f"{item['statistic']:.0f} & {item['p']:.5f} & {item['rank_biserial']:.3f}" + r"\\")
    table += [r"\bottomrule", r"\end{tabular}"]
    (out / "tab_matched.tex").write_text("\n".join(table), encoding="utf-8")
    fig, axs = plt.subplots(1, 2, figsize=(8, 3.3), sharey=True)
    for ax, m in zip(axs, MODELS):
        p = per_image[(per_image.model == m) & (per_image.sigma == BLUR_SIGMA)].sort_values("D")
        ax.plot(np.arange(1, len(p)+1), p.D, ".", color="tab:blue" if m == MODELS[0] else "tab:orange")
        ax.axhline(0, color="black", lw=.8)
        ax.set(title=m, xlabel="Radiografías ordenadas por D")
        ax.grid(alpha=.2)
    axs[0].set_ylabel("D: diferencia pareada de logit")
    fig.tight_layout(); fig.savefig(out / "fig_matched_d.png", dpi=220); plt.close(fig)
    image, region = cases[example_name]
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))
    for ax, with_cam in zip(axes, (False, True)):
        ax.imshow(image, cmap="gray")
        if with_cam and example_heat is not None:
            ax.imshow(example_heat, cmap="jet", alpha=np.clip(example_heat, 0, 1) * .5, vmin=0, vmax=1)
        ax.contour(region["lung"], levels=[.5], colors="yellow", linewidths=.6)
        ax.contour(region["index"], levels=[.5], colors="red", linewidths=1)
        for mask in region["controls"]: ax.contour(mask, levels=[.5], colors="cyan", linewidths=.6)
        ax.set(title="Geometría" if not with_cam else "Grad-CAM positivo, semilla 0")
        ax.axis("off")
    fig.tight_layout(); fig.savefig(out / "fig_matched_example.png", dpi=220); plt.close(fig)
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--results", default="results")
    p.add_argument("--annotations", default="data/montgomery_consensus_roi.csv")
    p.add_argument("--geometry-only", action="store_true")
    p.add_argument("--output", help="Directorio de la sensibilidad; lee pesos y particiones de --results")
    p.add_argument("--edge-tolerance", type=float, default=EDGE_TOLERANCE_PX)
    p.add_argument("--control-separation", type=float, default=CONTROL_CENTER_SEPARATION_PX)
    a = p.parse_args()
    EDGE_TOLERANCE_PX = a.edge_tolerance
    CONTROL_CENTER_SEPARATION_PX = a.control_separation
    print(json.dumps(run(a.data_root, a.results, a.annotations, a.geometry_only, a.output), indent=2))
