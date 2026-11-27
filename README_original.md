# Auditoría de explicaciones en radiografías de tórax

El estudio entrena DenseNet121 y ResNet50 con Shenzhen, evalúa ambos modelos en Montgomery y compara Grad-CAM y LIME con controles aleatorios. Las etiquetas originales distinguen radiografías normales de radiografías anormales con manifestaciones **compatibles con TB**; no representan una confirmación clínica independiente.

La guía independiente `guia_xai_tuberculosis.html` explica en español qué es XAI, cómo funcionan los modelos y el protocolo, cómo interpretar los resultados y por qué este experimento no valida un dispositivo clínico. Se puede abrir directamente en un navegador sin ejecutar Python ni instalar dependencias.

## Entorno local (Windows, RTX 3050)

Desde PowerShell en esta carpeta:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
.\.venv\Scripts\python.exe -m pip install -r environment.lock.txt
.\.venv\Scripts\python.exe -m ipykernel install --user --name deeplinv-cuda --display-name "DeepLinv (CUDA)"
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

La última línea debe imprimir `True`. El notebook usa ese kernel. Los paquetes de CUDA incluyen los componentes necesarios para usar la GPU; se necesita un controlador NVIDIA compatible.

## Datos

El notebook usa `kmader/pulmonary-chest-xray-abnormalities` y busca las imágenes bajo `CXR_DATA_ROOT`, `/kaggle/input`, `./pulmonary-chest-xray-abnormalities` o `./data`. Si no encuentra imágenes, intenta descargarlas con `kagglehub`. Exige 662 imágenes de Shenzhen, 138 de Montgomery y máscaras izquierda y derecha para las 138 de Montgomery.

Para el análisis de regiones compatibles con TB usa `data/montgomery_consensus_roi.csv` y `data/shenzhen_consensus_roi.csv`, descargados del [repositorio oficial de NLM](https://www.lhncbc.nlm.nih.gov/LHC-publications/pubs/TuberculosisChestXrayDatasets.html). Sus coordenadas corresponden a imágenes redimensionadas a 1024 × 1024. Montgomery tiene 55 radiografías positivas anotadas; 18 radiografías positivas anotadas de Shenzhen pertenecen al test fijado antes del análisis de regiones.

## Ejecución

La primera ejecución puede descargar los pesos de ImageNet. Para una prueba corta:

```powershell
$env:QUICK_RUN='1'
$env:RESULTS_DIR='results_smoke'
.\.venv\Scripts\jupyter-nbconvert.exe --to notebook --execute --ExecutePreprocessor.timeout=-1 --output deteccion_tb_xai_smoke.ipynb deteccion_tb_xai.ipynb
```

Para el estudio completo, en una sesión nueva de PowerShell:

```powershell
$env:CXR_DATA_ROOT=(Resolve-Path 'data\kagglehub_cache\datasets\kmader\pulmonary-chest-xray-abnormalities\versions\1').Path
.\.venv\Scripts\python.exe scripts\audit_data.py --data-root $env:CXR_DATA_ROOT --results results
.\.venv\Scripts\jupyter-nbconvert.exe --to notebook --execute --ExecutePreprocessor.timeout=-1 --output deteccion_tb_xai_ejecutado.ipynb deteccion_tb_xai.ipynb
.\.venv\Scripts\python.exe scripts\finalize_xai.py --results results
.\.venv\Scripts\python.exe scripts\analyze_performance.py --results results
.\.venv\Scripts\python.exe -m scripts.shenzhen_roi --data-root $env:CXR_DATA_ROOT --results results
.\.venv\Scripts\python.exe -m scripts.sanity_maps --data-root $env:CXR_DATA_ROOT --results results
.\.venv\Scripts\python.exe -m scripts.occlusion_analysis --data-root $env:CXR_DATA_ROOT --results results
.\.venv\Scripts\python.exe -m scripts.refresh_figures --data-root $env:CXR_DATA_ROOT --results results
.\.venv\Scripts\python.exe scripts\make_article_extras.py --results results
.\.venv\Scripts\python.exe scripts\package_results.py --results results
pdflatex -interaction=nonstopmode articulo_ieee.tex
pdflatex -interaction=nonstopmode articulo_ieee.tex
```

Los resultados principales están en `results/`: particiones, pesos, predicciones por imagen, tablas, figuras, métricas de explicabilidad, intervalos bootstrap por radiografía y análisis de cajas de consenso, pesos aleatorios y perturbaciones de igual área. `results.zip` contiene los resultados sin los pesos `.pt`; el artículo compilado es `articulo_ieee.pdf`. El notebook con las salidas completas es `deteccion_tb_xai_ejecutado.ipynb`. `QUICK_RUN` reduce tamaño, épocas y muestras; sus cifras no sirven para el artículo.

## Lectura de los hallazgos

Los AUC medios de tres semillas fueron 0.938 (DenseNet121) y 0.930 (ResNet50) en Shenzhen, y 0.850 y 0.792 en Montgomery. El umbral elegido en Shenzhen produjo gran variación de sensibilidad externa entre semillas. Grad-CAM concentró energía en las cajas de hallazgos de Montgomery por encima del control aleatorio, pero la puntería del máximo dentro de la caja fue modesta. Al desenfocar áreas iguales en casos positivos externos, la probabilidad de la clase positiva cayó más al intervenir el pulmón que al intervenir el borde. Estos análisis son exploratorios y no convierten las etiquetas radiográficas en diagnósticos microbiológicos.

Para verificar los cálculos auxiliares: `.\.venv\Scripts\python.exe -m unittest discover -s tests -v`. Antes de enviar el PDF a una revista o congreso, completar los nombres, afiliación y correo en `articulo_ieee.tex`.
