from __future__ import annotations

from pathlib import Path

from _download_utils import (
    load_manifest,
    manifest_path,
    raw_root,
    sync_download,
    write_manifest,
)

NIJ_FILES = {
    "nij-challenge2021_training_dataset.csv": "https://nij.ojp.gov/sites/g/files/xyckuh171/files/media/document/nij-challenge2021_training_dataset.csv",
    "nij-challenge2021_test_dataset_1.csv": "https://nij.ojp.gov/sites/g/files/xyckuh171/files/media/document/nij-challenge2021_test_dataset_1.csv",
    "nij-challenge2021_test_dataset_2.csv": "https://nij.ojp.gov/sites/g/files/xyckuh171/files/media/document/nij-challenge2021_test_dataset_2.csv",
    "nij-challenge2021_test_dataset_3.csv": "https://nij.ojp.gov/sites/g/files/xyckuh171/files/media/document/nij-challenge2021_test_dataset_3.csv",
    "nij-challenge2021_full_dataset.csv": "https://nij.ojp.gov/sites/g/files/xyckuh171/files/media/document/nij-challenge2021_full_dataset.csv",
    "recidivism-forecasting-challenge-appendix-2-codebook.pdf": "https://www.ojp.gov/pdffiles1/nij/304110.pdf",
}


def main() -> None:
    raw_dir = raw_root()
    nij_dir = raw_dir / "nij"
    manifest = manifest_path()
    checksums = load_manifest(manifest)

    for filename, url in NIJ_FILES.items():
        destination = nij_dir / filename
        rel_key = str(Path("nij") / filename)
        checksums[rel_key] = sync_download(url, destination, checksums, rel_key)

    write_manifest(manifest, checksums)
    print(f"updated checksum manifest: {manifest}")


if __name__ == "__main__":
    main()
