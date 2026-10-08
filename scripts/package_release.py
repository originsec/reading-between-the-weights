"""Package an explicitly prepared artifact tree; never reads a private workspace."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import tarfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def bundle(folder: Path, out: Path, tag: str) -> tuple[str, dict[str, Any]]:
    archive = out / f"{folder.name}.tar.gz"
    files = {}
    with (
        archive.open("wb") as raw,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=6) as compressed,
        tarfile.open(fileobj=compressed, mode="w") as tar,
    ):
        for path in sorted(folder.rglob("*")):
            if path.is_symlink():
                raise ValueError("Symlinks are not release artifacts")
            if not path.is_file():
                continue
            name = path.relative_to(folder).as_posix()
            info = tar.gettarinfo(str(path), arcname=name)
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = 0
            info.mode = 0o644
            with path.open("rb") as source:
                tar.addfile(info, source)
            files[name] = {"sha256": digest(path), "bytes": path.stat().st_size}
    return folder.name, {
        "filename": archive.name,
        "bytes": archive.stat().st_size,
        "sha256": digest(archive),
        "url": f"https://github.com/originsec/reading-between-the-weights/releases/download/{tag}/{archive.name}",
        "files": files,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("payload", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--tag", default="v0.1.0")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [
            pool.submit(bundle, path, args.output, args.tag)
            for path in sorted(args.payload.iterdir())
            if path.is_dir()
        ]
        bundles = dict(f.result() for f in futures)
    args.manifest.write_text(
        json.dumps({"schema_version": 1, "release": args.tag, "bundles": bundles}, indent=2) + "\n"
    )
    (args.output / "SHA256SUMS").write_text(
        "".join(f"{v['sha256']}  {v['filename']}\n" for v in bundles.values())
    )
    print(json.dumps({k: round(v["bytes"] / 1e6, 2) for k, v in bundles.items()}, indent=2))


if __name__ == "__main__":
    main()
