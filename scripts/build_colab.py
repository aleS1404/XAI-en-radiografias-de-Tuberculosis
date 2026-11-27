"""Build a self-contained Colab notebook from the archived source notebook."""

import copy
from pathlib import Path
import nbformat

ROOT = Path(__file__).resolve().parents[1]
original = nbformat.read(ROOT / "deteccion_tb_xai.ipynb", as_version=4)
out = nbformat.v4.new_notebook(metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}})
cells = [copy.deepcopy(c) for c in original.cells]
for c in cells:
    if c.cell_type == "code":
        c.outputs = []
        c.execution_count = None

cells[0] = nbformat.v4.new_markdown_cell("""# Evaluación externa de explicaciones de redes convolucionales mediante perturbaciones pulmonares en radiografías con hallazgos compatibles con tuberculosis

**Extensión retrospectiva con evaluación externa.** Se entrenan DenseNet121 y ResNet50 en Shenzhen y se evalúan en Montgomery. El contraste principal compara, por radiografía anotada, la pérdida de puntuación positiva tras desenfocar una caja índice con la mediana de regiones sin anotación de igual forma digital en el mismo pulmón. Las etiquetas indican hallazgos radiográficos compatibles con TB; no son confirmación microbiológica.

Ejecutar de arriba abajo en Colab con GPU. `QUICK_RUN=1` activa una prueba técnica; sus resultados no se usan como evidencia empírica. La descarga de Kaggle puede requerir autenticación mediante los secretos de Colab o una copia local configurada en `CXR_DATA_ROOT`. Las anotaciones NLM se descargan de su sitio oficial y también pueden cargarse como CSV locales. Los checkpoints existentes solo se reutilizan si su configuración coincide.
""")
cells[2] = nbformat.v4.new_code_cell("""# En Colab se instalan dependencias Linux compatibles antes de importar el estudio.
import sys, subprocess
if 'google.colab' in sys.modules:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
        'kagglehub>=0.3,<2', 'grad-cam==1.5.7', 'lime==0.2.0.1',
        'scikit-learn>=1.5,<2', 'scipy>=1.13,<2', 'opencv-python-headless>=4.8,<5'])
""")
cells[3].source = cells[3].source.replace('import os, sys, json, random, time, warnings, zipfile, platform',
                                         'import os, sys, json, random, time, warnings, zipfile, platform, hashlib')
cells[4] = nbformat.v4.new_markdown_cell("""## 1. Datos y procedencia
Las etiquetas `_0` y `_1` identifican radiografías normales y radiografías con manifestaciones compatibles con TB según la NLM. El identificador disponible es el de radiografía; no se ha verificado un identificador independiente de paciente. La descarga de imágenes se realiza desde Kaggle; las cajas auxiliares proceden de NLM.
""")
cells[7] = nbformat.v4.new_markdown_cell("""## 2. Preprocesamiento y particiones
Se conserva la partición estratificada original de Shenzhen (462/67/133; semilla 42), el redimensionado a 224×224 y la normalización ImageNet. Montgomery permanece fuera del entrenamiento, la parada temprana y la selección de umbral. En modo rápido se reducen resolución y cómputo para verificar el flujo; sus cifras no se mezclan con el estudio.
""")
cells[22] = nbformat.v4.new_markdown_cell("""## 7. Localización histórica de cajas de hallazgos
Se explica la clase positiva fija, incluso en falsos negativos. Este análisis de cajas es distinto del análisis general anterior, que explicaba la clase predicha. Las cajas son aproximaciones anatómicas, no segmentaciones de lesión.
""")

for cell in cells:
    if cell.cell_type == 'code' and 'model, hist, best_ep = train_model(mname, seed)' in cell.source:
        cell.source = cell.source.replace('RUNS = {}\nt_all = time.time()', '''RUNS = {}
# Se reutiliza una corrida completa solo si la configuración y el manifiesto
# de particiones coinciden y están todos los archivos requeridos.
checkpoint_signature = dict(cfg=CFG, splits_sha256=hashlib.sha256((OUT / 'splits.csv').read_bytes()).hexdigest())
signature_file = OUT / 'checkpoint_signature.json'
history_file = OUT / 'training_history.json'
saved_signature = json.loads(signature_file.read_text(encoding='utf-8')) if signature_file.exists() else None
can_resume = (saved_signature == checkpoint_signature and history_file.exists() and
              all((OUT / f'{m}_s{s}.pt').exists() for m in CFG['MODELS'] for s in CFG['SEEDS']))
saved_history = json.loads(history_file.read_text(encoding='utf-8')) if can_resume else {}
signature_file.write_text(json.dumps(checkpoint_signature, indent=2), encoding='utf-8')
print('Reanudación verificada:', can_resume)
t_all = time.time()''')
        cell.source = cell.source.replace('model, hist, best_ep = train_model(mname, seed)', '''if can_resume:
            model = build_model(mname, pretrained=False).to(device)
            model.load_state_dict(torch.load(OUT / f'{mname}_s{seed}.pt', map_location=device, weights_only=True))
            model.eval()
            hist = saved_history[f'{mname}_s{seed}']
            best_ep = int(np.argmax(hist['val_auc'])) + 1
        else:
            model, hist, best_ep = train_model(mname, seed)''')
        break

acquire = nbformat.v4.new_code_cell("""# Anotaciones oficiales de NLM: descarga directa o CSV ya presentes/cargados.
import hashlib, urllib.request
DATA_DIR = Path('data'); DATA_DIR.mkdir(exist_ok=True)
ROI_URLS = {
 'montgomery_consensus_roi.csv': 'https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Montgomery-County-CXR-Set/MontgomerySet/montgomery_consensus_roi.csv',
 'shenzhen_consensus_roi.csv': 'https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Shenzhen-Hospital-CXR-Set/shenzhen_consensus_roi.csv'}
ROI_STATUS = {}
for name, url in ROI_URLS.items():
    path = DATA_DIR / name
    if not path.exists():
        try: urllib.request.urlretrieve(url, path)
        except Exception as exc: ROI_STATUS[name] = 'PENDIENTE: ' + str(exc)
    if path.exists():
        table = pd.read_csv(path)
        assert {'patientId','x_dis','y_dis','width_dis','height_dis','Labels'} <= set(table)
        ROI_STATUS[name] = dict(url=url, boxes=len(table), images=table.patientId.nunique(),
                                sha256=hashlib.sha256(path.read_bytes()).hexdigest())
print(json.dumps(ROI_STATUS, indent=2, ensure_ascii=False))
if not (DATA_DIR / 'montgomery_consensus_roi.csv').exists():
    raise FileNotFoundError('Cargue el CSV oficial NLM en data/ y vuelva a ejecutar desde esta celda.')
""")
cells.insert(22, acquire)

lesion_source = (ROOT / "scripts/lesion_roi.py").read_text(encoding="utf-8")
cells.insert(24, nbformat.v4.new_code_cell(lesion_source.replace('"""Exploratory localization against NLM consensus TB finding boxes.\n\nThe unit of uncertainty is an image. Maps from repeated training seeds are\naveraged per image before confidence intervals and paired comparisons.\n"""', '')))
for cell in cells:
    if cell.cell_type == 'code' and 'from scripts.lesion_roi import run_lesion_analysis' in cell.source:
        cell.source = cell.source.replace('from scripts.lesion_roi import run_lesion_analysis\n', '')
        break

matching = (ROOT / "scripts/matched_lesion.py").read_text(encoding="utf-8")
matching = matching.split('if __name__ == "__main__":')[0]
matching = matching.replace('from scripts.lesion_roi import boxes_to_mask\n', '')
matching = matching.replace('from scripts.shenzhen_roi import MEAN, STD, model_for\n', '')
matching = matching.replace('SEEDS = (0, 1, 2)', 'SEEDS = tuple(CFG["SEEDS"])')
matching = matching.replace('MODELS = ("densenet121", "resnet50")', 'MODELS = tuple(CFG["MODELS"])')
model_for = """MEAN = np.array(IMAGENET_MEAN, dtype=np.float32)
STD = np.array(IMAGENET_STD, dtype=np.float32)
def model_for(name):
    m = build_model(name, pretrained=False)
    return m, target_layer(m, name)
"""
qualitative = (ROOT / "scripts/qualitative_lung_maps.py").read_text(encoding="utf-8")
qualitative = qualitative.split('if __name__ == "__main__":')[0]
qualitative = qualitative.replace('from scripts.matched_lesion import image_files, read_case, select_regions, SIZE, tensor\n', '')
qualitative = qualitative.replace('from scripts.shenzhen_roi import MEAN, STD, model_for\n', '')
qualitative = qualitative.replace('def run(data_root, results, annotations):', 'def run_qualitative(data_root, results, annotations):')
out.cells = cells + [
    nbformat.v4.new_markdown_cell("""## 8. Extensión: intervención pareada dentro del pulmón
La elegibilidad y la elección de controles se cierran mediante geometría antes de inferir. Se conserva la forma digital, el número de píxeles, el pulmón y el tercio vertical. Los controles son **regiones sin anotación** y pueden contener hallazgos no marcados. La unidad inferencial es la radiografía; las tres semillas se promedian dentro de cada una. `D > 0` indica una mayor caída del logit positivo menos el normal al intervenir la región índice. Es evidencia de dependencia del clasificador bajo este operador, no causalidad clínica.
"""),
    nbformat.v4.new_code_cell(model_for + '\n' + matching),
    nbformat.v4.new_code_cell("""# La prueba rápida verifica adquisición y elegibilidad; la corrida completa infiere con los seis checkpoints.
MATCHED = run(ROOT, OUT, DATA_DIR / 'montgomery_consensus_roi.csv', geometry_only=QUICK)
print(json.dumps(MATCHED, indent=2, ensure_ascii=False))
"""),
    nbformat.v4.new_code_cell("""# Sensibilidades geométricas posteriores: nunca sustituyen el contraste principal.
if not QUICK:
    main_edge, main_separation = EDGE_TOLERANCE_PX, CONTROL_CENTER_SEPARATION_PX
    sensitivity_rows = []
    for tag, edge, separation, folder in [('principal', 5, 12, OUT),
                                           ('borde_10', 10, 12, OUT / 'edge10_separation12'),
                                           ('borde_10_centros_6', 10, 6, OUT / 'relaxed_geometry')]:
        if tag == 'principal': summary = MATCHED
        else:
            EDGE_TOLERANCE_PX, CONTROL_CENTER_SEPARATION_PX = edge, separation
            summary = run(ROOT, OUT, DATA_DIR / 'montgomery_consensus_roi.csv', output=folder)
        for model_name in MODELS:
            item = summary['models'][model_name][str(BLUR_SIGMA)]
            sensitivity_rows.append(dict(protocol=tag, status='principal' if tag == 'principal' else 'sensibilidad_poshoc',
              model=model_name, n=item['n'], median_D=item['median_ci'][0], ci_low=item['median_ci'][1],
              ci_high=item['median_ci'][2], wilcoxon_p=item['p'], sign_p=item['sign_p'],
              edge_tolerance_px=edge, center_separation_px=separation))
    EDGE_TOLERANCE_PX, CONTROL_CENTER_SEPARATION_PX = main_edge, main_separation
    pd.DataFrame(sensitivity_rows).to_csv(OUT / 'geometry_sensitivity.csv', index=False)
    print(pd.DataFrame(sensitivity_rows).to_string(index=False))
"""),
    nbformat.v4.new_markdown_cell("""## 9. Ejemplos de pulmones con Grad-CAM y LIME
Las cuatro primeras radiografías elegibles por identificador se muestran con la misma clase positiva y semilla 0 en ambas arquitecturas. Estas láminas son cualitativas y no alteran el contraste principal.
"""),
    nbformat.v4.new_code_cell(qualitative),
    nbformat.v4.new_code_cell("""if not QUICK:
    QUALITATIVE_CASES = run_qualitative(ROOT, OUT, DATA_DIR / 'montgomery_consensus_roi.csv')
    display(QUALITATIVE_CASES)
"""),
    nbformat.v4.new_code_cell("""# Métricas adicionales a partir de las predicciones exportadas.
from sklearn.metrics import average_precision_score, matthews_corrcoef, brier_score_loss
PRED = pd.read_csv(OUT / 'predictions.csv')
extra = []
for (model_name, seed, dataset), part in PRED.groupby(['model','seed','dataset']):
    y, p, pred = part.label.to_numpy(), part.probability_tb.to_numpy(), part.predicted.to_numpy()
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0,1]).ravel()
    extra.append(dict(model=model_name, seed=int(seed), dataset=dataset, n=len(part), prevalence=float(y.mean()),
      threshold=float(part.threshold.iloc[0]), ROC_AUC=float(roc_auc_score(y,p)),
      AP=float(average_precision_score(y,p)), F1=float(2*tp/(2*tp+fp+fn)),
      MCC=float(matthews_corrcoef(y,pred)), Brier=float(brier_score_loss(y,p)),
      TN=int(tn), FP=int(fp), FN=int(fn), TP=int(tp)))
extra_df = pd.DataFrame(extra)
extra_df.to_csv(OUT / 'performance_extended.csv', index=False)
tex = [r'\\begin{tabular}{llcccccc}', r'\\toprule',
       r'Modelo & Prueba & AUC & AP & F1 & MCC & Brier & TN/FP/FN/TP\\\\', r'\\midrule']
for model_name in CFG['MODELS']:
    for dataset, label in [('sz','Shenzhen'), ('mg','Montgomery')]:
        item = extra_df[(extra_df.model == model_name) & (extra_df.seed == 0) & (extra_df.dataset == dataset)].iloc[0]
        model_label = 'DenseNet121' if model_name == 'densenet121' else 'ResNet50'
        tex.append(f"{model_label} & {label} & {item.ROC_AUC:.3f} & {item.AP:.3f} & {item.F1:.3f} & "
                   f"{item.MCC:.3f} & {item.Brier:.3f} & {int(item.TN)}/{int(item.FP)}/{int(item.FN)}/{int(item.TP)}" + r'\\\\')
tex += [r'\\bottomrule', r'\\end{tabular}']
(OUT / 'tab_perf_extended.tex').write_text('\\n'.join(tex), encoding='utf-8')
print(extra_df.to_string(index=False))
"""),
    nbformat.v4.new_code_cell("""# Empaquetado final; se excluyen pesos por tamaño.
with zipfile.ZipFile(OUT.parent / f'{OUT.name}.zip', 'w', zipfile.ZIP_DEFLATED) as z:
    for path in sorted(OUT.rglob('*')):
        if path.is_file() and path.suffix != '.pt': z.write(path, path.relative_to(OUT).as_posix())
print('Exportación:', OUT.parent / f'{OUT.name}.zip')
""")]
nbformat.validate(out)
nbformat.write(out, ROOT / "auditoria_xai_tb_colab.ipynb")
print(ROOT / "auditoria_xai_tb_colab.ipynb")
