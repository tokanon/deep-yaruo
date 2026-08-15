from __future__ import annotations

import base64
import io

import cv2
import numpy as np
from PIL import Image


def decode_image(data: bytes) -> np.ndarray:
    encoded = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("The uploaded file is not a supported image.")
    return image


def decode_color_image(data: bytes) -> np.ndarray:
    """Decode an image as uint8 BGR while preserving visible color."""
    encoded = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError("The uploaded file is not a supported image.")
    if image.dtype == np.uint16:
        image = np.rint(image.astype(np.float32) / 257.0).astype(np.uint8)
    elif image.dtype != np.uint8:
        values = image.astype(np.float32)
        if values.size and float(values.max()) <= 1.0:
            values *= 255.0
        image = np.clip(values, 0, 255).astype(np.uint8)
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim != 3:
        raise ValueError(f"Unsupported decoded image shape: {image.shape}")
    if image.shape[2] == 3:
        return image
    if image.shape[2] == 4:
        color = image[:, :, :3].astype(np.float32)
        alpha = image[:, :, 3:4].astype(np.float32) / 255.0
        return np.rint(color * alpha + 255.0 * (1.0 - alpha)).astype(np.uint8)
    if image.shape[2] == 2:
        gray = image[:, :, :1].astype(np.float32)
        alpha = image[:, :, 1:2].astype(np.float32) / 255.0
        composited = np.rint(gray * alpha + 255.0 * (1.0 - alpha)).astype(np.uint8)
        return np.repeat(composited, 3, axis=2)
    raise ValueError(f"Unsupported decoded image channel count: {image.shape}")


def image_to_data_url(image: Image.Image | np.ndarray) -> str:
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{payload}"
