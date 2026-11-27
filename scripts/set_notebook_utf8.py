"""Use explicit UTF-8 for generated LaTeX artifacts on Windows."""

from pathlib import Path

import nbformat


path = Path("deteccion_tb_xai.ipynb")
notebook = nbformat.read(path, as_version=4)
source = notebook.cells[25].source
source = source.replace('with open(OUT / "macros.tex", "w") as f:',
                        'with open(OUT / "macros.tex", "w", encoding="utf-8") as f:')
source = source.replace('(OUT / "tab_perf.tex").write_text("\\n".join(L))',
                        '(OUT / "tab_perf.tex").write_text("\\n".join(L), encoding="utf-8")')
source = source.replace('(OUT / "tab_xai.tex").write_text("\\n".join(L))',
                        '(OUT / "tab_xai.tex").write_text("\\n".join(L), encoding="utf-8")')
source = source.replace('open(OUT / "config.json", "w")',
                        'open(OUT / "config.json", "w", encoding="utf-8")')
source = source.replace('(OUT / "training_history.json").write_text(json.dumps({f"{m}_s{s}": RUNS[(m, s)]["hist"] for m, s in RUNS}, indent=2))',
                        '(OUT / "training_history.json").write_text(json.dumps({f"{m}_s{s}": RUNS[(m, s)]["hist"] for m, s in RUNS}, indent=2), encoding="utf-8")')
notebook.cells[25].source = source
nbformat.write(notebook, path)
print("UTF-8 exports:", source.count('encoding="utf-8"'))
