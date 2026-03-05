from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from typing import Dict
from urllib.request import Request, urlopen


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def raw_root() -> Path:
    path = repo_root() / "data" / "raw"
    path.mkdir(parents=True, exist_ok=True)
    return path


def manifest_path() -> Path:
    return raw_root() / "checksums.sha256"


def load_manifest(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    mapping: Dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        checksum, rel_path = line.split(maxsplit=1)
        mapping[rel_path.strip()] = checksum.strip()
    return mapping


def write_manifest(path: Path, checksums: Dict[str, str]) -> None:
    lines = [f"{checksums[key]}  {key}" for key in sorted(checksums)]
    path.write_text("\n".join(lines) + "\n")


def sha256sum(file_path: Path) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": "precrime-data-pipeline/1.0"})
    with tempfile.NamedTemporaryFile(delete=False, dir=destination.parent) as temp_handle:
        temp_path = Path(temp_handle.name)
        with urlopen(request) as response:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                temp_handle.write(chunk)
    temp_path.replace(destination)


def sync_download(url: str, destination: Path, checksums: Dict[str, str], rel_key: str) -> str:
    if destination.exists():
        current_checksum = sha256sum(destination)
        expected_checksum = checksums.get(rel_key)
        if expected_checksum and current_checksum == expected_checksum:
            print(f"skip  {rel_key} (checksum match)")
            return current_checksum
        if expected_checksum:
            print(f"refresh {rel_key} (checksum mismatch)")
        else:
            print(f"accept {rel_key} (no prior checksum)")
            return current_checksum
    else:
        print(f"fetch {rel_key}")

    download_file(url, destination)
    return sha256sum(destination)
