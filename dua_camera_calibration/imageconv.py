"""
ROS image encodings and compressed images to 8-bit grayscale, with numpy (no cv_bridge).

dotX Automation s.r.l. <info@dotxautomation.com>

September 28, 2026
"""

# Copyright 2026 dotX Automation s.r.l.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cv2
import numpy as np


class ImageError(Exception):
    """Image conversion failure: code is 'UNSUPPORTED_ENCODING', 'DECODE_FAILED' or 'BAD_SIZE'."""

    def __init__(self, code: str, **params):
        """Create the error with a machine-readable code and its parameters."""
        super().__init__(code, params)
        self.code = code
        self.params = params


# encoding -> (channels, bytes per channel, cvtColor code to gray or None)
_ENCODINGS = {
    'mono8': (1, 1, None),
    '8UC1': (1, 1, None),
    'mono16': (1, 2, None),
    '16UC1': (1, 2, None),
    'bgr8': (3, 1, cv2.COLOR_BGR2GRAY),
    '8UC3': (3, 1, cv2.COLOR_BGR2GRAY),
    'rgb8': (3, 1, cv2.COLOR_RGB2GRAY),
    'bgra8': (4, 1, cv2.COLOR_BGRA2GRAY),
    '8UC4': (4, 1, cv2.COLOR_BGRA2GRAY),
    'rgba8': (4, 1, cv2.COLOR_RGBA2GRAY),
    'bayer_rggb8': (1, 1, cv2.COLOR_BayerRGGB2GRAY),
    'bayer_bggr8': (1, 1, cv2.COLOR_BayerBGGR2GRAY),
    'bayer_gbrg8': (1, 1, cv2.COLOR_BayerGBRG2GRAY),
    'bayer_grbg8': (1, 1, cv2.COLOR_BayerGRBG2GRAY),
    'bayer_rggb16': (1, 2, cv2.COLOR_BayerRGGB2GRAY),
    'bayer_bggr16': (1, 2, cv2.COLOR_BayerBGGR2GRAY),
    'bayer_gbrg16': (1, 2, cv2.COLOR_BayerGBRG2GRAY),
    'bayer_grbg16': (1, 2, cv2.COLOR_BayerGRBG2GRAY),
    'yuv422': (2, 1, None),
    'uyvy': (2, 1, None),
    'yuv422_yuy2': (2, 1, None),
    'yuyv': (2, 1, None),
    '32FC1': (1, 4, None),
}


def _to8(img16: np.ndarray) -> np.ndarray:
    """Shift 16-bit data right by (effective bits - 8), effective bits in [8, 16] from the max."""
    bits = min(max(int(img16.max()).bit_length(), 8), 16)
    return (img16 >> (bits - 8)).astype(np.uint8)


def image_to_gray(encoding: str, width: int, height: int, step: int, data,
                  is_bigendian: bool = False) -> np.ndarray:
    """
    Convert a sensor_msgs/Image payload to a C-contiguous uint8 (height, width) array.

    data may be bytes, bytearray, array.array or memoryview; row padding (step) is honoured and
    the result never shares memory with data. Raises ImageError.
    """
    if encoding not in _ENCODINGS:
        raise ImageError('UNSUPPORTED_ENCODING', encoding=encoding)
    ch, bpc, code = _ENCODINGS[encoding]
    buf = np.frombuffer(data, dtype=np.uint8)
    row = width * ch * bpc
    if width <= 0 or height <= 0 or step < row or buf.size < step * (height - 1) + row:
        raise ImageError('BAD_SIZE', encoding=encoding, width=width, height=height, step=step,
                         size=int(buf.size))
    dtype = np.dtype({1: 'u1', 2: 'u2', 4: 'f4'}[bpc]).newbyteorder('>' if is_bigendian else '<')
    img = np.ndarray((height, width, ch), dtype=dtype, buffer=buf, strides=(step, ch * bpc, bpc))

    if encoding in ('yuv422', 'uyvy'):
        return np.ascontiguousarray(img[:, :, 1])   # U Y V Y: luma is the second byte
    if encoding in ('yuv422_yuy2', 'yuyv'):
        return np.ascontiguousarray(img[:, :, 0])   # Y U Y V: luma is the first byte
    if encoding == '32FC1':
        f = np.nan_to_num(img[:, :, 0].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        m = float(f.max())
        return np.rint(np.clip(f * (255.0 / m), 0, 255) if m > 0 else f * 0).astype(np.uint8)
    if bpc == 2:
        img = _to8(img[:, :, 0])
    if code is None:
        return np.array(img.reshape(height, width), dtype=np.uint8, order='C')
    return cv2.cvtColor(np.ascontiguousarray(img), code)


def compressed_to_gray(fmt: str, data) -> tuple:
    """
    Decode a sensor_msgs/CompressedImage payload to (uint8 gray, lossy).

    lossy is True for JPEG (format string or FF D8 magic). EXIF orientation is ignored so pixel
    coordinates stay sensor coordinates. Raises ImageError('DECODE_FAILED').
    """
    buf = np.frombuffer(data, dtype=np.uint8)
    img = None
    if buf.size > 0:
        img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE | cv2.IMREAD_IGNORE_ORIENTATION)
    if img is None:
        raise ImageError('DECODE_FAILED', format=fmt, size=int(buf.size))
    f = fmt.lower()
    lossy = 'jpeg' in f or 'jpg' in f or bytes(buf[:2]) == b'\xff\xd8'
    return img, lossy
