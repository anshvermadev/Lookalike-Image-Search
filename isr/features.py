from __future__ import annotations

from functools import lru_cache
from io import BytesIO
import os
from pathlib import Path
import warnings

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from .config import MODEL_PATH

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 20_000_000
MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


def validate_dimensions(width: int, height: int) -> None:
    if min(width, height) < 1 or width * height > MAX_PIXELS:
        raise ValueError("Please use an image with no more than 20 million pixels.")
    if max(width, height) > 20000 or max(width, height) / min(width, height) > 50:
        raise ValueError("This image is too long or narrow. Crop it to a conventional photo shape.")


def normalize(vectors: np.ndarray) -> np.ndarray:
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim not in (1, 2) or not np.all(np.isfinite(array)):
        raise ValueError("Features must be finite vectors.")
    norms = np.linalg.norm(array, axis=-1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("Cannot normalize an empty or zero feature vector.")
    return np.ascontiguousarray(array / norms, dtype=np.float32)


def open_rgb(source: Path | bytes | Image.Image) -> Image.Image:
    if isinstance(source, Image.Image):
        validate_dimensions(*source.size)
        return ImageOps.exif_transpose(source).convert("RGB")
    if isinstance(source, bytes) and len(source) > MAX_UPLOAD_BYTES:
        raise ValueError("Please upload an image smaller than 10 MB.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(source) if isinstance(source, bytes) else source) as img:
                validate_dimensions(*img.size)
                return ImageOps.exif_transpose(img).convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as error:
        raise ValueError("This file could not be read as an image. Try a valid JPG, PNG, or WebP.") from error


def histogram(image: Image.Image) -> np.ndarray:
    img = open_rgb(image)
    img.thumbnail((256, 256), Image.Resampling.LANCZOS)
    pixels = np.asarray(img.convert("HSV")).reshape(-1, 3)
    hist, _ = np.histogramdd(pixels, bins=(8, 8, 8), range=((0, 256),) * 3)
    return normalize(hist.ravel())


def clip_pixels(image: Image.Image) -> np.ndarray:
    """CLIP's RGB, shortest-edge bicubic resize, center crop, and normalization."""
    img = open_rgb(image)
    width, height = img.size
    if width <= height:
        size = (224, int(224 * height / width))
    else:
        size = (int(224 * width / height), 224)
    img = img.resize(size, Image.Resampling.BICUBIC)
    left, top = (img.width - 224) // 2, (img.height - 224) // 2
    pixels = np.asarray(img.crop((left, top, left + 224, top + 224)), dtype=np.float32) / 255.0
    return np.ascontiguousarray(((pixels - MEAN) / STD).transpose(2, 0, 1))


@lru_cache(maxsize=2)
def _session(path: str, mtime_ns: int):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = min(4, os.cpu_count() or 1)
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    session = ort.InferenceSession(path, sess_options=options, providers=["CPUExecutionProvider"])
    if "pixel_values" not in {item.name for item in session.get_inputs()}:
        raise ValueError("The downloaded model is not the expected CLIP vision encoder.")
    if "image_embeds" not in {item.name for item in session.get_outputs()}:
        raise ValueError("The model does not expose projected CLIP image embeddings.")
    return session


class Encoder:
    def __init__(self, method: str, model_path: Path = MODEL_PATH):
        if method not in {"histogram", "clip"}:
            raise ValueError(f"Unknown retrieval method: {method}")
        self.method = method
        self.session = None
        if method == "clip":
            if not model_path.is_file():
                raise FileNotFoundError("CLIP model is not installed. Run scripts/download_model.py once.")
            self.session = _session(str(model_path.resolve()), model_path.stat().st_mtime_ns)

    def encode(self, images: list[Image.Image]) -> np.ndarray:
        if not images:
            return np.empty((0, 512), dtype=np.float32)
        if self.method == "histogram":
            return np.stack([histogram(img) for img in images])
        batch = np.stack([clip_pixels(img) for img in images])
        embeddings = self.session.run(["image_embeds"], {"pixel_values": batch})[0]
        return normalize(embeddings)
