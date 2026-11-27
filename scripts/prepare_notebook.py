"""Apply the locally verified execution fixes to the original notebook."""

import json
from pathlib import Path


path = Path(__file__).resolve().parents[1] / "deteccion_tb_xai.ipynb"
notebook = json.loads(path.read_text(encoding="utf-8"))


def source(index):
    return "".join(notebook["cells"][index]["source"])


def set_source(index, value):
    notebook["cells"][index]["source"] = value.splitlines(keepends=True)
    notebook["cells"][index]["execution_count"] = None
    notebook["cells"][index]["outputs"] = []


def replace(index, old, new):
    value = source(index)
    if old not in value:
        raise RuntimeError(f"No se encontró el fragmento esperado en la celda {index}: {old!r}")
    set_source(index, value.replace(old, new))


set_source(2, "# Dependencias instaladas en .venv; usar el kernel 'DeepLinv (CUDA)'.\n")

replace(3, "IMG_SIZE=128 if QUICK else 224, BATCH=8 if QUICK else 32,",
        "IMG_SIZE=128 if QUICK else 224, BATCH=4, ACCUM_STEPS=1 if QUICK else 8,")
replace(3, "NUM_WORKERS = 2 if device.type == \"cuda\" else 0",
        "NUM_WORKERS = 0 if os.name == \"nt\" else (2 if device.type == \"cuda\" else 0)")

replace(5,
        'df = (pd.DataFrame(rows).drop_duplicates("name").sort_values("name").reset_index(drop=True))',
        'df = pd.DataFrame(rows)\nassert not df.empty, "No se encontraron radiografías."\nassert not df.name.duplicated().any(), "Hay nombres de radiografía duplicados."\ndf = df.sort_values("name").reset_index(drop=True)')
replace(5, 'assert set(df.source) == {"shenzhen", "montgomery"}, "No se encontraron ambos conjuntos; revisa la ruta del dataset."',
        'assert set(df.source) == {"shenzhen", "montgomery"}, "No se encontraron ambos conjuntos; revisa la ruta del dataset."\nassert (df.source.value_counts().to_dict() == {"shenzhen": 662, "montgomery": 138}), "Conteo de imágenes distinto de la fuente NLM."\nassert df.loc[df.source == "montgomery", "has_mask"].all(), "Faltan máscaras de Montgomery."\nassert set(df.label) == {0, 1}, "Etiquetas inesperadas."')

replace(8, 'display(split_tbl)',
        'display(split_tbl)\nSPLITS = pd.concat([d.assign(split=name) for name, d in [("train", sz_train), ("val", sz_val), ("test", sz_test), ("external", mg)]], ignore_index=True)\nassert SPLITS.name.is_unique, "Hay fuga de imágenes entre particiones."\nSPLITS[["name", "source", "label", "split"]].to_csv(OUT / "splits.csv", index=False)')

set_source(12, '''def run_epoch(model, loader, criterion, opt=None, scaler=None):
    train = opt is not None
    model.train(train)
    total, n, correct, probs, ys = 0.0, 0, 0, [], []
    if train:
        opt.zero_grad(set_to_none=True)
    with torch.set_grad_enabled(train):
        for step, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=train and device.type == "cuda"):
                out = model(x)
                loss = criterion(out, y)
            if train:
                group_start = (step // CFG["ACCUM_STEPS"]) * CFG["ACCUM_STEPS"]
                group_size = min(CFG["ACCUM_STEPS"], len(loader) - group_start)
                scaler.scale(loss / group_size).backward()
                if (step + 1) % CFG["ACCUM_STEPS"] == 0 or step + 1 == len(loader):
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad(set_to_none=True)
            total += loss.item() * len(y)
            n += len(y)
            correct += (out.argmax(1) == y).sum().item()
            probs.append(F.softmax(out.float(), 1)[:, 1].detach().cpu().numpy())
            ys.append(y.cpu().numpy())
    probs, ys = np.concatenate(probs), np.concatenate(ys)
    auc = roc_auc_score(ys, probs) if len(np.unique(ys)) > 1 else float("nan")
    return total / n, correct / n, auc

def train_model(name, seed):
    set_seed(seed)
    model = build_model(name, CFG["PRETRAINED"]).to(device)
    cnt = np.bincount(sz_train.label.values, minlength=2)
    weights = torch.tensor(len(sz_train) / (2.0 * cnt), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["LR"], weight_decay=CFG["WD"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CFG["EPOCHS"])
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    dl_tr, dl_va = make_loader(sz_train, True, seed), make_loader(sz_val, False)
    hist = {k: [] for k in ["train_loss", "train_acc", "val_loss", "val_acc", "val_auc"]}
    best_auc, best_ep, bad = -1, 0, 0
    ckpt = OUT / f"{name}_s{seed}.pt"
    for ep in range(CFG["EPOCHS"]):
        tl, ta, _ = run_epoch(model, dl_tr, criterion, opt, scaler)
        sched.step()
        vl, va, vauc = run_epoch(model, dl_va, criterion)
        for k, v in zip(hist, [tl, ta, vl, va, vauc]):
            hist[k].append(float(v))
        if vauc > best_auc:
            best_auc, best_ep, bad = vauc, ep, 0
            torch.save(model.state_dict(), ckpt)
        else:
            bad += 1
        print(f"  [{name} s{seed}] ep {ep+1:02d} | loss {tl:.3f}/{vl:.3f} | acc {ta:.3f}/{va:.3f} | val AUC {vauc:.3f}", flush=True)
        if bad >= CFG["PATIENCE"]:
            print("  early stopping")
            break
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    return model, hist, best_ep + 1

@torch.no_grad()
def predict_proba(model, frame):
    model.eval()
    ps = []
    for x, _ in make_loader(frame, False, bs=8):
        ps.append(F.softmax(model(x.to(device)), 1)[:, 1].cpu().numpy())
    return np.concatenate(ps), frame.label.values

def youden_threshold(y, p):
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[np.argmax(tpr - fpr)])

def compute_metrics(y, p, thr):
    yhat = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    sens, spec = tp / max(tp + fn, 1), tn / max(tn + fp, 1)
    prec = tp / max(tp + fp, 1)
    return dict(acc=(tp + tn) / len(y), sens=sens, spec=spec, prec=prec,
                f1=2 * prec * sens / max(prec + sens, 1e-12), auc=roc_auc_score(y, p))
''')

replace(18, 'torch.load(OUT / f"{mname}_s{seed}.pt", map_location=device)',
        'torch.load(OUT / f"{mname}_s{seed}.pt", map_location=device, weights_only=True)')
replace(18, 'X[i:i + 64]', 'X[i:i + 8]')
replace(18, 'range(0, len(X), 64)', 'range(0, len(X), 8)')
replace(18, 'classes, bs=16', 'classes, bs=2')
replace(18, 'batch_size=64, segmentation_fn=seg', 'batch_size=8, segmentation_fn=seg')
replace(18, '''    rng = np.random.RandomState(seed)
    order = np.argsort(-(h.ravel() + 1e-6 * rng.rand(h.size)))''',
        '''    rng = np.random.RandomState(seed)
    gray = (x[0].cpu().numpy() * IMAGENET_STD[0] + IMAGENET_MEAN[0]).astype(np.float32)
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=10)
    background = torch.stack([(torch.from_numpy(blurred) - IMAGENET_MEAN[c]) / IMAGENET_STD[c] for c in range(3)])
    order = np.argsort(-(h.ravel() + 1e-6 * rng.rand(h.size)))''')
replace(18, 'xx[:, rr, cc] = 0.0   # 0 = media de ImageNet',
        'xx[:, rr, cc] = background[:, rr, cc]   # fondo gris desenfocado, normalizado por canal')

path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print("Notebook actualizado:", path)
