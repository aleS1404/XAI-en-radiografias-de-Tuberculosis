"""Pack the small reproducibility artifacts after all analyses finish."""

import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args()
    destination = args.results.with_suffix(".zip")
    with ZipFile(destination, "w", ZIP_DEFLATED) as archive:
        for path in sorted(args.results.rglob("*")):
            if path.is_file() and path.suffix != ".pt":
                archive.write(path, path.relative_to(args.results).as_posix())
    print(destination)
