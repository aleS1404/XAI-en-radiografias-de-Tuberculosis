"""Create a minimal Overleaf project from the finished IEEE article."""

from pathlib import Path
import re
import shutil
import zipfile


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "articulo_ieee.tex"
PROJECT = ROOT / "articulo_overleaf"
ARCHIVE = ROOT / "articulo_overleaf.zip"


def export():
    source = SOURCE.read_text(encoding="utf-8")
    figures = set(re.findall(r"\\figfile\{([^}]+)\}", source))
    tables = set(re.findall(r"\\tabfile\{([^}]+)\}", source))
    assets = {"macros.tex", "extra_macros.tex", *figures, *tables}
    missing = sorted(name for name in assets if not (ROOT / "results" / name).is_file())
    if missing:
        raise FileNotFoundError(f"Faltan archivos del artículo: {missing}")

    (PROJECT / "results").mkdir(parents=True, exist_ok=True)
    (PROJECT / "main.tex").write_text(source, encoding="utf-8")
    for name in sorted(assets):
        source_path = ROOT / "results" / name
        if name.endswith(".tex"):
            source_path.read_text(encoding="utf-8")
        shutil.copy2(source_path, PROJECT / "results" / name)

    instructions = (
        "PROYECTO LATEX PARA OVERLEAF\n\n"
        "1. En Overleaf: New Project > Upload Project. Selecciona articulo_overleaf.zip.\n"
        "2. Abre main.tex; usa pdfLaTeX como compilador (IEEEtran).\n"
        "3. Completa autoría, afiliación y contacto en el bloque \\author de main.tex.\n"
        "4. Las cifras, tablas y cinco figuras del artículo están en results/.\n"
        "   No necesitas subir el dataset, la GPU ni los pesos de los modelos.\n"
    )
    (PROJECT / "README_OVERLEAF.txt").write_text(instructions, encoding="utf-8")

    members = [Path("main.tex"), Path("README_OVERLEAF.txt")]
    members += [Path("results") / name for name in sorted(assets)]
    with zipfile.ZipFile(ARCHIVE, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for member in members:
            archive.write(PROJECT / member, member.as_posix())
    with zipfile.ZipFile(ARCHIVE) as archive:
        if archive.testzip() is not None or set(archive.namelist()) != {p.as_posix() for p in members}:
            raise RuntimeError("El archivo ZIP no pasó la verificación")
    print(f"Creado: {ARCHIVE} ({len(members)} archivos)")


if __name__ == "__main__":
    export()
