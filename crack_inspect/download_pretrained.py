"""Best-effort downloader for publicly available cracked-egg YOLOv8 weights.

This is intentionally a thin helper: it tries a few known URLs, validates
the downloaded checkpoint with torch, and copies it into the project
root. If every URL fails, the script exits non-zero so the caller can
fall back to training from scratch.

Examples
--------
python download_pretrained.py                 # download to ./best.pt
python download_pretrained.py --out my.pt     # custom target path
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


CANDIDATE_URLS: List[Tuple[str, str]] = [
    # (label, url)
    # The original upstream repo does not ship weights; below are
    # community / academic releases of cracked-egg detection models. Each
    # is tried in order until one succeeds.
    (
        "roboflow_universe_cracked_eggs_yolov8_seg",
        "https://universe.roboflow.com/ds/YOUR_DOWNLOAD_KEY/best.pt",  # placeholder
    ),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download a pretrained cracked-egg YOLOv8-seg model")
    p.add_argument("--out", type=str, default="best.pt")
    p.add_argument("--urls", type=str, nargs="*", default=None,
                   help="Override candidate URLs (one per line or space-separated)")
    p.add_argument("--timeout", type=int, default=60)
    return p.parse_args()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _try_download(url: str, dst: Path, timeout: int) -> Optional[str]:
    if "YOUR_DOWNLOAD_KEY" in url or not url.startswith("http"):
        return f"placeholder URL: {url}"
    print(f"[download] GET {url}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "egg-inspect-downloader/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp, dst.open("wb") as fh:
            shutil.copyfileobj(resp, fh)
    except urllib.error.HTTPError as exc:
        return f"HTTP {exc.code}"
    except urllib.error.URLError as exc:
        return f"URL error: {exc.reason}"
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    return None


def _validate_weights(path: Path) -> bool:
    try:
        import torch
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        return isinstance(ckpt, dict) and ("model" in ckpt or "state_dict" in ckpt)
    except Exception:
        return False


def download(out: Path, urls: Iterable[str], timeout: int) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    for url in urls:
        err = _try_download(url, out, timeout)
        if err is None:
            sha = _sha256(out)
            print(f"[download] saved {out} ({out.stat().st_size} bytes, sha256={sha[:12]}...)")
            if _validate_weights(out):
                print(f"[download] weights look like a valid torch checkpoint")
                return 0
            print(f"[download] WARNING: weights at {out} did not pass torch.load validation")
            out.unlink(missing_ok=True)
        else:
            print(f"[download]   failed: {err}")

    print("[download] No candidate URL succeeded.")
    print("[download] Train your own weights via train_inspector.py.")
    return 1


def main() -> int:
    args = parse_args()
    out_path = Path(args.out)
    urls = args.urls or [u for _, u in CANDIDATE_URLS]
    return download(out_path, urls, args.timeout)


if __name__ == "__main__":
    sys.exit(main())