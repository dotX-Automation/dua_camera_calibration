"""
Tests of the ROS image to grayscale conversions.

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

import array

import cv2
from dua_camera_calibration.imageconv import compressed_to_gray, image_to_gray, ImageError
import numpy as np
import pytest

W, H, PAD = 8, 6, 5
GRAY = np.random.default_rng(0).integers(0, 256, (H, W)).astype(np.uint8)


def _pack(pixels: np.ndarray, pad: int = PAD) -> tuple:
    """Serialize rows with pad garbage bytes each; return (data, step)."""
    rows = pixels.reshape(pixels.shape[0], -1).view(np.uint8)
    garbage = np.full((rows.shape[0], pad), 0xAB, np.uint8)
    return np.hstack([rows, garbage]).tobytes(), rows.shape[1] + pad


def _check(out: np.ndarray, data) -> None:
    """Assert the output contract: uint8 (H, W), C-contiguous, not sharing memory."""
    assert out.dtype == np.uint8 and out.shape == (H, W) and out.flags.c_contiguous
    assert not np.shares_memory(out, np.frombuffer(data, np.uint8))


@pytest.mark.parametrize('encoding', ['mono8', '8UC1'])
@pytest.mark.parametrize('kind', [bytes, bytearray, memoryview, 'array'])
def test_mono8(encoding, kind):
    """Mono 8-bit from every buffer type, with row padding."""
    data, step = _pack(GRAY)
    data = array.array('B', data) if kind == 'array' else kind(data)
    out = image_to_gray(encoding, W, H, step, data)
    _check(out, data)
    assert np.array_equal(out, GRAY)


@pytest.mark.parametrize('encoding', ['mono16', '16UC1'])
@pytest.mark.parametrize('bits', [16, 12, 10])
@pytest.mark.parametrize('bigendian', [False, True])
def test_mono16_effective_bits(encoding, bits, bigendian):
    """Mono 16-bit is shifted by its effective bit depth, both endiannesses."""
    img16 = (GRAY.astype(np.uint16) << (bits - 8)).astype('>u2' if bigendian else '<u2')
    img16[0, 0] = (1 << bits) - 1      # full scale present: shift is exactly bits - 8
    data, step = _pack(img16)
    out = image_to_gray(encoding, W, H, step, data, bigendian)
    _check(out, data)
    ref = GRAY.copy()
    ref[0, 0] = 255
    assert np.array_equal(out, ref)


def test_mono16_dark_image_not_shifted_below_8_bits():
    """An 8-bit range in 16-bit data is not amplified."""
    data, step = _pack(GRAY.astype('<u2'))
    assert np.array_equal(image_to_gray('mono16', W, H, step, data), GRAY)


@pytest.mark.parametrize('encoding,order', [
    ('bgr8', 'bgr'), ('8UC3', 'bgr'), ('rgb8', 'rgb'),
    ('bgra8', 'bgra'), ('8UC4', 'bgra'), ('rgba8', 'rgba')])
def test_color(encoding, order):
    """Colour encodings honour their channel order."""
    bgr = np.random.default_rng(1).integers(0, 256, (H, W, 3)).astype(np.uint8)
    ref = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    chan = dict(zip('bgr', np.moveaxis(bgr, -1, 0)), a=np.full_like(bgr[..., 0], 7))
    pix = np.stack([chan[k] for k in order], axis=-1)
    data, step = _pack(pix)
    out = image_to_gray(encoding, W, H, step, data)
    _check(out, data)
    assert np.array_equal(out, ref)


@pytest.mark.parametrize('pattern', ['rggb', 'bggr', 'gbrg', 'grbg'])
@pytest.mark.parametrize('depth', [8, 16])
def test_bayer(pattern, depth):
    """Bayer encodings are demosaiced with the right pattern."""
    # Uniform colour R=200, G=100, B=50: gray 124; a wrong pattern gives 97 (R/B swapped).
    value = {'r': 200, 'g': 100, 'b': 50}
    tile = np.array([[value[pattern[0]], value[pattern[1]]],
                     [value[pattern[2]], value[pattern[3]]]])
    raw = np.tile(tile, (H // 2, W // 2))
    raw = raw.astype(np.uint8) if depth == 8 else (raw.astype(np.uint16) << 4).astype('<u2')
    data, step = _pack(raw)
    out = image_to_gray(f'bayer_{pattern}{depth}', W, H, step, data)
    _check(out, data)
    assert np.all(np.abs(out[2:-2, 2:-2].astype(int) - 124) <= 2)


@pytest.mark.parametrize('encoding,luma', [
    ('yuv422', 1), ('uyvy', 1), ('yuv422_yuy2', 0), ('yuyv', 0)])
def test_yuv422(encoding, luma):
    """YUV 4:2:2 takes the luma bytes."""
    pix = np.full((H, W, 2), 128, np.uint8)
    pix[..., luma] = GRAY
    data, step = _pack(pix)
    out = image_to_gray(encoding, W, H, step, data)
    _check(out, data)
    assert np.array_equal(out, GRAY)


@pytest.mark.parametrize('bigendian', [False, True])
def test_32fc1_scaled_by_max(bigendian):
    """Float images are scaled by their finite maximum."""
    f = (GRAY.astype(np.float32) * 0.01).astype('>f4' if bigendian else '<f4')
    f[0, 0], f[0, 1] = np.nan, 2.55    # NaN -> 0; max 2.55 -> 255
    data, step = _pack(f)
    out = image_to_gray('32FC1', W, H, step, data, bigendian)
    _check(out, data)
    ref = GRAY.copy()
    ref[0, 0], ref[0, 1] = 0, 255
    assert np.array_equal(out, ref)


def test_errors():
    """Bad inputs raise ImageError with the right code."""
    data, step = _pack(GRAY)
    with pytest.raises(ImageError) as e:
        image_to_gray('hsv8', W, H, step, data)
    assert e.value.code == 'UNSUPPORTED_ENCODING' and e.value.params['encoding'] == 'hsv8'
    for w, h, s, d in [(W, H, W - 1, data), (W, H, step, data[:-step]), (0, H, step, data)]:
        with pytest.raises(ImageError) as e:
            image_to_gray('mono8', w, h, s, d)
        assert e.value.code == 'BAD_SIZE'
    for payload in [b'', b'not an image']:
        with pytest.raises(ImageError) as e:
            compressed_to_gray('png', payload)
        assert e.value.code == 'DECODE_FAILED'


def test_compressed_png_and_jpeg():
    """PNG decodes exactly; JPEG is flagged lossy by format or magic bytes."""
    img = np.kron(GRAY, np.ones((8, 8), np.uint8))
    png = cv2.imencode('.png', img)[1].tobytes()
    out, lossy = compressed_to_gray('png', array.array('B', png))
    assert np.array_equal(out, img) and not lossy and out.dtype == np.uint8
    color = cv2.imencode('.png', cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))[1].tobytes()
    out, lossy = compressed_to_gray('bgr8; png compressed bgr8', color)
    assert np.array_equal(out, img) and not lossy
    jpg = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()
    for fmt in ['jpeg', 'rgb8; jpeg compressed bgr8', '']:     # '' -> JPEG magic bytes
        out, lossy = compressed_to_gray(fmt, jpg)
        assert lossy and out.shape == img.shape
        assert np.abs(out.astype(int) - img).mean() < 8
