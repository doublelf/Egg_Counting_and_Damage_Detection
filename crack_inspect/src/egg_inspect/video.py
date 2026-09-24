from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Union

import cv2
import numpy as np


@dataclass
class FrameMeta:
    index: int
    timestamp: float
    path: Optional[Path]


class VideoSource:
    """Webcam / video file / image folder source."""

    def __init__(
        self,
        source: Union[str, int],
        loop: bool = False,
        resize_width: Optional[int] = None,
        resize_height: Optional[int] = None,
    ) -> None:
        self.source = source
        self.loop = loop
        self.resize_width = resize_width
        self.resize_height = resize_height
        self._cap: Optional[cv2.VideoCapture] = None
        self._image_iter: Optional[Iterator[Path]] = None
        self._fps: float = 30.0
        self._index = 0
        self._kind: str = "unknown"

    def __enter__(self) -> "VideoSource":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @property
    def kind(self) -> str:
        return self._kind

    @property
    def fps(self) -> float:
        return self._fps

    def open(self) -> None:
        src = self.source
        if isinstance(src, str) and src.isdigit():
            src = int(src)
        if isinstance(src, int):
            self._cap = cv2.VideoCapture(src)
            if not self._cap.isOpened():
                raise RuntimeError(f"Could not open webcam index {src}")
            self._kind = "webcam"
            self._fps = float(self._cap.get(cv2.CAP_PROP_FPS) or 30.0)
            return

        path = Path(str(src))
        if path.is_dir():
            exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
            files: List[Path] = sorted(
                p for p in path.iterdir() if p.is_file() and p.suffix.lower() in exts
            )
            if not files:
                raise RuntimeError(f"No image files found in {path}")
            self._image_iter = iter(files)
            self._kind = "images"
            self._fps = 5.0
            return

        if not path.exists():
            raise FileNotFoundError(f"Video/image source not found: {path}")
        self._cap = cv2.VideoCapture(str(path))
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open video: {path}")
        self._kind = "video"
        self._fps = float(self._cap.get(cv2.CAP_PROP_FPS) or 30.0)

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._image_iter = None

    def _resize(self, frame: np.ndarray) -> np.ndarray:
        if self.resize_width is None and self.resize_height is None:
            return frame
        h, w = frame.shape[:2]
        new_w = self.resize_width or w
        new_h = self.resize_height or h
        if (new_w, new_h) == (w, h):
            return frame
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def read(self) -> Optional[tuple[np.ndarray, FrameMeta]]:
        if self._cap is not None:
            while True:
                ok, frame = self._cap.read()
                if not ok:
                    if self.loop:
                        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    return None
                self._index += 1
                ts = self._index / max(self._fps, 1e-6)
                frame = self._resize(frame)
                return frame, FrameMeta(index=self._index, timestamp=ts, path=None)
        if self._image_iter is not None:
            for p in self._image_iter:
                img = cv2.imread(str(p))
                if img is None:
                    continue
                self._index += 1
                img = self._resize(img)
                return img, FrameMeta(index=self._index, timestamp=self._index / self._fps, path=p)
        return None


def open_source(
    source: Union[str, int],
    loop: bool = False,
    resize_width: Optional[int] = None,
    resize_height: Optional[int] = None,
) -> VideoSource:
    return VideoSource(
        source=source,
        loop=loop,
        resize_width=resize_width,
        resize_height=resize_height,
    )