"""
Adaptive Pre-processing — Analyze each image and apply targeted enhancements.

Detects:
1. Brightness/contrast issues (dark scans, faded text)
2. Small text / high-density content (newspaper columns)
3. Skew/rotation
4. Noise levels

Then applies the appropriate combination of fixes.
"""

import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance


def analyze_image(img_array: np.ndarray) -> dict:
    """Analyze image characteristics to decide preprocessing strategy."""
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) if len(img_array.shape) == 3 else img_array
    h, w = gray.shape

    # 1. Brightness analysis
    mean_brightness = np.mean(gray)
    is_dark = mean_brightness < 120
    is_bright = mean_brightness > 200

    # 2. Contrast analysis (std dev of pixel values)
    contrast = np.std(gray)
    is_low_contrast = contrast < 40
    is_high_contrast = contrast > 80

    # 3. Text density / small text detection
    #    Use edge density as proxy — more edges = more/smaller text
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / (h * w)
    is_dense_text = edge_density > 0.08  # newspaper-like columns
    is_sparse_text = edge_density < 0.02  # mostly blank or few lines

    # 4. Noise estimation (using Laplacian variance)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    is_noisy = laplacian_var > 1000
    is_clean = laplacian_var < 200

    # 5. Skew detection
    skew_angle = detect_skew_angle(gray)

    analysis = {
        "brightness": mean_brightness,
        "contrast": contrast,
        "edge_density": edge_density,
        "noise_level": laplacian_var,
        "skew_angle": skew_angle,
        "is_dark": is_dark,
        "is_bright": is_bright,
        "is_low_contrast": is_low_contrast,
        "is_dense_text": is_dense_text,
        "is_sparse_text": is_sparse_text,
        "is_noisy": is_noisy,
    }

    return analysis


def detect_skew_angle(gray: np.ndarray) -> float:
    """Detect dominant skew angle using Hough lines."""
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

    if lines is None or len(lines) == 0:
        return 0.0

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(angle) < 10:
            angles.append(angle)

    if not angles:
        return 0.0

    return float(np.median(angles))


def deskew(image: np.ndarray, angle: float) -> np.ndarray:
    """Correct skew if angle is meaningful."""
    if abs(angle) < 0.3 or abs(angle) > 5:
        return image
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def enhance_dark_image(image: np.ndarray) -> np.ndarray:
    """Brighten dark scans while preserving text contrast."""
    # Convert to LAB and boost lightness
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    # Aggressive CLAHE for dark images
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    # Additional gamma correction to lift shadows
    l = np.clip(l * 1.3, 0, 255).astype(np.uint8)

    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def enhance_low_contrast(image: np.ndarray) -> np.ndarray:
    """Boost contrast for faded typewriter text."""
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def enhance_dense_text(image: np.ndarray) -> np.ndarray:
    """Sharpen and clean up small/dense text (newspaper columns)."""
    # Sharpen to make small letters crisper
    kernel = np.array([
        [0, -0.5, 0],
        [-0.5, 3, -0.5],
        [0, -0.5, 0]
    ])
    sharpened = cv2.filter2D(image, -1, kernel)

    # Light contrast boost
    lab = cv2.cvtColor(sharpened, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])

    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def denoise_image(image: np.ndarray) -> np.ndarray:
    """Remove scan noise while preserving text edges."""
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    denoised = cv2.fastNlMeansDenoisingColored(bgr, None, 8, 8, 7, 21)
    return cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB)


def binarize_for_ocr(image: np.ndarray) -> np.ndarray:
    """Apply adaptive thresholding for very degraded scans.
    Returns a clean black-on-white image."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 10
    )
    # Convert back to RGB (model expects RGB input)
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)


def resize_for_ocr(image: Image.Image, max_pixels: int = 800_000) -> Image.Image:
    """Resize image so total pixels stays under max_pixels."""
    w, h = image.size
    total = w * h
    if total <= max_pixels:
        return image
    scale = (max_pixels / total) ** 0.5
    new_w = int(w * scale)
    new_h = int(h * scale)
    return image.resize((new_w, new_h), Image.LANCZOS)


def enhance_image(image: Image.Image) -> Image.Image:
    """Adaptive pre-processing pipeline.

    Analyzes the image first, then applies only the relevant enhancements.
    """
    img_array = np.array(image)
    analysis = analyze_image(img_array)

    # Log what we detected
    print(f"        Analysis: brightness={analysis['brightness']:.0f}, "
          f"contrast={analysis['contrast']:.0f}, "
          f"edges={analysis['edge_density']:.3f}, "
          f"noise={analysis['noise_level']:.0f}, "
          f"skew={analysis['skew_angle']:.1f}°")

    strategy = []

    # Step 1: Deskew if needed
    if abs(analysis['skew_angle']) >= 0.3:
        img_array = deskew(img_array, analysis['skew_angle'])
        strategy.append("deskew")

    # Step 2: Brightness/contrast fixes
    if analysis['is_dark']:
        img_array = enhance_dark_image(img_array)
        strategy.append("brighten")
    elif analysis['is_low_contrast']:
        img_array = enhance_low_contrast(img_array)
        strategy.append("contrast-boost")

    # Step 3: Text type specific
    if analysis['is_dense_text']:
        img_array = enhance_dense_text(img_array)
        strategy.append("sharpen-dense")

    # Step 4: Denoise if noisy (but not for dense text — denoising blurs small letters)
    if analysis['is_noisy'] and not analysis['is_dense_text']:
        img_array = denoise_image(img_array)
        strategy.append("denoise")

    if not strategy:
        strategy.append("none-needed")

    print(f"        Strategy: {', '.join(strategy)}")

    return Image.fromarray(img_array)