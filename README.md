# Auditoría XAI de radiografías compatibles con tuberculosis

Esta carpeta conserva el PDF, los tres cuadernos originales y una copia del [README anterior](README_original.md). La ampliación está en [articulo_ieee.pdf](articulo_ieee.pdf), [articulo_ieee.tex](articulo_ieee.tex) y [auditoria_xai_tb_colab.ipynb](auditoria_xai_tb_colab.ipynb). La autoría y afiliación proceden del PDF original. Las etiquetas positivas son hallazgos radiográficos compatibles con TB, no confirmación microbiológica.

## Estado de la evidencia

- **Histórico verificable:** seis checkpoints, predicciones y puntuaciones XAI de tres semillas. Shenzhen: 662 radiografías (326/336) divididas en 462/67/133. Montgomery: 138 (80/58), con dos máscaras por caso. Las cajas oficiales NLM comprenden 109 anotaciones en 55 positivos de Montgomery. El análisis general histórico explica la clase predicha; el de cajas, la clase positiva fija. Los valores exactos de Wilcoxon para EIL Grad-CAM frente a LIME fueron `0.001440798383555375` y `5.844412953592837e-09`.
- **Extensión ejecutada:** [scripts/matched_lesion.py](scripts/matched_lesion.py) seleccionó regiones y controles solo por geometría antes de inferir, y analizó 17/55 radiografías elegibles con los seis checkpoints. DenseNet121: mediana de `D` 0,1716, IC bootstrap 95 % 0,0653–0,6323, Wilcoxon bilateral `p=0.010986328125`; prueba de signos `p=0.049041748046875`. ResNet50: mediana 0,0636, IC −0,0049–0,3358. La restricción geométrica y el uso anterior de Montgomery hacen exploratorio el resultado.
- **Sensibilidad geométrica posterior:** con tolerancia al borde de 10 píxeles y la separación original de 12, se recuperaron 23 casos. Al permitir centros separados por 6 píxeles se obtuvieron 26, pero el solapamiento mediano máximo entre controles aumentó de 0,108 (reglas originales) a 0,461. Los resultados de ambas ampliaciones están en [geometry_sensitivity.csv](results/geometry_sensitivity.csv), separados del contraste principal porque las reglas se ajustaron después de ver el resultado inicial.
- **Pendiente de replicación independiente:** reentrenar y ejecutar el cuaderno completo desde una sesión limpia de Colab; evaluar otra cohorte independiente con anotaciones y máscaras adecuadas. La corrida rápida técnica no constituye evidencia empírica. La licencia específica de la copia de Kaggle no se pudo confirmar de forma concluyente en los archivos locales; verifique condiciones vigentes antes de redistribuir imágenes o pesos.

La [NLM](https://lhncbc.nlm.nih.gov/LHC-publications/pubs/TuberculosisChestXrayDatasets.html) es la fuente de las imágenes y anotaciones; [Kaggle](https://www.kaggle.com/datasets/kmader/pulmonary-chest-xray-abnormalities) fue el canal de descarga de radiografías. La copia local de KaggleHub fue la versión 1. [results/data_audit.json](results/data_audit.json) registra conteos y hashes de anotaciones; [results/image_hashes.csv](results/image_hashes.csv) contiene SHA-256 de las 800 radiografías. No se encontraron archivos de imagen idénticos entre particiones. Solo se dispone de identificadores de radiografía, sin una clave independiente de paciente.

## Colab

Abra `auditoria_xai_tb_colab.ipynb` en Colab con GPU y ejecute todas las celdas. El cuaderno instala dependencias Linux, busca `CXR_DATA_ROOT` o descarga `kmader/pulmonary-chest-xray-abnormalities` con KaggleHub, recupera los CSV oficiales NLM y valida su estructura. Si Kaggle exige autenticación, utilice secretos de Colab o cargue una copia local y configure `CXR_DATA_ROOT`; no escriba credenciales en el cuaderno. Si NLM no responde, cargue los CSV oficiales en `data/`. Los checkpoints solo se reanudan cuando coincide la firma de configuración y particiones. `QUICK_RUN=1` activa la prueba técnica y debe usar un directorio de resultados separado. Los pesos `.pt` y las radiografías descargadas no se publican en GitHub; una clonación limpia debe ejecutar la corrida completa para generar sus propios checkpoints antes de repetir la inferencia local.

Para reproducir localmente la ampliación con los pesos existentes:

```powershell
$env:CXR_DATA_ROOT=(Resolve-Path 'data/kagglehub_cache/datasets/kmader/pulmonary-chest-xray-abnormalities/versions/1').Path
.\.venv\Scripts\python.exe -m scripts.matched_lesion --data-root $env:CXR_DATA_ROOT --results results
.\.venv\Scripts\python.exe scripts/export_extended_performance.py --results results
.\.venv\Scripts\python.exe -m scripts.qualitative_lung_maps --data-root $env:CXR_DATA_ROOT --results results
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
bibtex articulo_ieee
pdflatex -interaction=nonstopmode -halt-on-error articulo_ieee.tex
pdflatex -interaction=nonstopmode -halt-on-error articulo_ieee.tex
```

Los archivos nuevos son `matched_eligibility.csv`, `matched_protocol.json`, `matched_scores.csv`, `matched_per_image.csv`, `matched_summary.json`, `matched_map_baselines.csv`, `performance_extended.csv`, `tab_matched.tex`, `geometry_sensitivity.csv`, las dos carpetas de sensibilidad y las figuras `fig_matched_*.png`, todos en `results/`. Las láminas [DenseNet121](results/fig_lung_xai_densenet121.png) y [ResNet50](results/fig_lung_xai_resnet50.png) muestran cuatro casos idénticos con Grad-CAM y LIME de la clase positiva; `qualitative_lung_cases.csv` registra sus probabilidades, predicciones y parámetros. El CSV de puntuaciones conserva los controles, curvas de eliminación e inserción, e IoU del 10 % pulmonar superior por radiografía y semilla. La inferencia se realizó con PyTorch 2.5.1+cu121 en NVIDIA RTX 3050 Laptop GPU; las pruebas estadísticas usaron SciPy 1.18.1. El PDF usa `IEEEtran` a dos columnas y compila en doce páginas.
