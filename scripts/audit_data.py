"""Audit the local NLM/Kaggle image collection and annotation correspondence."""

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
import pandas as pd


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit(root):
    root = Path(root)
    images = [path for path in root.rglob("*.png") if path.parent.name == "CXR_png"]
    left = {path.name for path in root.rglob("*.png") if path.parent.name == "leftMask"}
    right = {path.name for path in root.rglob("*.png") if path.parent.name == "rightMask"}
    names = [path.name for path in images]
    if len(names) != len(set(names)):
        raise ValueError("Radiografías con nombres duplicados")
    source = lambda name: "Montgomery" if name.startswith("MCUCXR") else "Shenzhen" if name.startswith("CHNCXR") else "Otro"
    counts = Counter((source(name), name.rsplit("_", 1)[-1].removesuffix(".png")) for name in names)
    mg = {name for name in names if source(name) == "Montgomery"}
    if len(images) != 800 or len(mg) != 138 or len(mg & left & right) != 138:
        raise ValueError("Conteo o correspondencia de máscaras inesperado")
    csvs = {}
    for location in (Path("data/montgomery_consensus_roi.csv"), Path("data/shenzhen_consensus_roi.csv")):
        rows = list(csv.DictReader(location.open(encoding="utf-8")))
        unique_names = {row["patientId"] for row in rows}
        csvs[location.name] = {"boxes": len(rows), "images": len(unique_names),
                               "matched_images": len(unique_names & set(names)), "sha256": sha256(location)}
        if unique_names - set(names):
            raise ValueError(f"Anotaciones sin radiografía en {location}")
    manifest = "\n".join(f"{path.relative_to(root).as_posix()}\t{path.stat().st_size}" for path in sorted(images))
    return {"source": "kmader/pulmonary-chest-xray-abnormalities", "version": 1,
            "images": len(images), "labels": {f"{kind}_{label}": count for (kind, label), count in counts.items()},
            "montgomery_complete_mask_pairs": len(mg & left & right),
            "file_manifest_sha256": hashlib.sha256(manifest.encode()).hexdigest(),
            "roi_csv": csvs}


def audit_split_hashes(root, splits_path, out_path):
    """Hash image bytes and check both IDs and content across every split."""
    root = Path(root)
    split = pd.read_csv(splits_path)
    if split.name.duplicated().any():
        raise ValueError("Identificadores duplicados entre particiones")
    lookup = {p.name: p for p in root.rglob("*.png") if p.parent.name == "CXR_png"}
    if set(split.name) != set(lookup):
        raise ValueError("El manifiesto de particiones no cubre exactamente las radiografías")
    rows = []
    for row in split.itertuples(index=False):
        rows.append(dict(name=row.name, split=row.split, source=row.source,
                         label=int(row.label), sha256=sha256(lookup[row.name])))
    frame = pd.DataFrame(rows)
    frame.to_csv(out_path, index=False)
    repeats = frame[frame.sha256.duplicated(keep=False)].sort_values("sha256")
    return {"n_images_hashed": len(frame),
            "duplicate_content_groups": int(repeats.sha256.nunique()),
            "duplicate_content_across_splits": int(repeats.groupby("sha256").split.nunique().gt(1).sum()),
            "duplicate_rows": repeats[["name", "split", "sha256"]].to_dict("records")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--results", default="results")
    args = parser.parse_args()
    report = audit(args.data_root)
    output = Path(args.results) / "data_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    split_path = Path(args.results) / "splits.csv"
    if split_path.exists():
        report["split_hash_audit"] = audit_split_hashes(args.data_root, split_path,
                                                         Path(args.results) / "image_hashes.csv")
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
