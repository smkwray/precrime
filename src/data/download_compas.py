from __future__ import annotations

from pathlib import Path

from _download_utils import (
    load_manifest,
    manifest_path,
    raw_root,
    sync_download,
    write_manifest,
)

COMPAS_FILES = {
    "README": "https://raw.githubusercontent.com/propublica/compas-analysis/master/README",
    "compas-scores-two-years.csv": "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv",
    "compas-scores-two-years-violent.csv": "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years-violent.csv",
    "compas-scores-raw.csv": "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-raw.csv",
    "compas.db": "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas.db",
}


def main() -> None:
    raw_dir = raw_root()
    compas_dir = raw_dir / "compas"
    manifest = manifest_path()
    checksums = load_manifest(manifest)

    for filename, url in COMPAS_FILES.items():
        destination = compas_dir / filename
        rel_key = str(Path("compas") / filename)
        checksums[rel_key] = sync_download(url, destination, checksums, rel_key)

    write_manifest(manifest, checksums)
    print(f"updated checksum manifest: {manifest}")


if __name__ == "__main__":
    main()
