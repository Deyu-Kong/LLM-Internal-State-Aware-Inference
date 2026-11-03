"""Pinned source downloads with verified cache reuse and no overwrite."""
import hashlib
import os
from pathlib import Path
import tempfile
import urllib.request

MAX_SOURCE_BYTES = 8 * 1024 * 1024


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def gsm8k_sources(raw_dir, settings, allow_download=False):
    folder = Path(raw_dir) / "gsm8k" / settings["revision"]
    sources = []
    for name, expected in settings["sha256"].items():
        path = folder / name
        suffix = "LICENSE" if name == "LICENSE" else "grade_school_math/data/" + name
        url = f"https://raw.githubusercontent.com/openai/grade-school-math/{settings['revision']}/{suffix}"
        if not path.exists():
            if not allow_download:
                raise FileNotFoundError(f"Missing source {path}; pass --download to fetch pinned data")
            with urllib.request.urlopen(url, timeout=45) as response:
                payload = response.read(MAX_SOURCE_BYTES + 1)
            if len(payload) > MAX_SOURCE_BYTES or hashlib.sha256(payload).hexdigest() != expected:
                raise ValueError(f"Source size/hash mismatch for {name}")
            folder.mkdir(parents=True, exist_ok=True)
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=folder, delete=False) as stream:
                    temporary = Path(stream.name)
                    stream.write(payload)
                # Atomic no-overwrite publication, including concurrent preparations.
                try:
                    os.link(temporary, path)
                except FileExistsError:
                    pass  # Verify the concurrently written file below.
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        if sha256_file(path) != expected:
            raise ValueError(f"Cached source hash mismatch for {name}; refusing to overwrite")
        sources.append({"name": name, "path": str(path), "url": url,
                        "revision": settings["revision"], "sha256": expected, "bytes": path.stat().st_size})
    return sources
