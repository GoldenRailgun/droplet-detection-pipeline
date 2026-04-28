"""
preprocess.py - Stage 1: Image Preprocessing
---------------------------------------------
Converts a raw grayscale image into a clean binary mask.
White pixels = droplet. Black pixels = background.

Pipeline:
    raw image -> grayscale -> gaussian blur -> otsu threshold -> morphological cleanup -> binary mask
"""

import cv2
import numpy as np

def load_image(image_path: str) -> np.ndarray:
    """
    Load image from disk.
    Raises clear error if path is wrong — better than a silent None crash.

    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    return img


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert to grayscale if needed.
    but this synthetic images are already grayscale — this handles real images too just in case if decided to move along with real images.
    """
    if len(image.shape) == 2:
        return image  # already single channel
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def apply_gaussian_blur(gray: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Smooth the image before thresholding.
    Reduces noise pixels so they don't get classified as droplets.
    Kernel must be odd — OpenCV requirement.
    Start with 5, increase to 7 if noise survives, decrease to 3 if droplets merge.
    """
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd")
    return cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)


def apply_threshold(blurred: np.ndarray) -> tuple:
    """
    Apply Otsu's method to find the optimal threshold automatically.
    Analyses the image histogram and finds the best cutoff between
    droplet pixels and background pixels — no manual tuning needed.
    Returns both the binary mask and the threshold value so we can
    log it and compare across images.
    """
    thresh_val, binary = cv2.threshold(
        blurred, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return binary, thresh_val


def clean_mask(binary: np.ndarray, open_kernel: int = 3, close_kernel: int = 5) -> np.ndarray:
    """
    Removing noise and fill holes in the binary mask.
    Opening (erode then dilate): removes small isolated noise blobs.
    Closing (dilate then erode): fills small holes inside droplets
    caused by specular highlights on the droplet surface.
    open_kernel should be smaller than the smallest droplet you expect.
    close_kernel should be smaller than the gap between adjacent droplets.
    """
    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel, open_kernel))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))

    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  k_open)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, k_close)
    return closed

def preprocess(image_path: str, blur_kernel: int = 5, open_kernel: int = 3, close_kernel: int = 5) -> dict:
    """
    Full preprocessing pipeline.
    Returns all intermediate stages as a dict so any step can be
    inspected during debugging — critical for root cause analysis.
    """
    raw     = load_image(image_path)
    gray    = to_grayscale(raw)
    blurred = apply_gaussian_blur(gray, kernel_size=blur_kernel)
    binary, thresh_val = apply_threshold(blurred)
    mask    = clean_mask(binary, open_kernel=open_kernel, close_kernel=close_kernel)

    print(f"[Preprocess] Otsu threshold value: {thresh_val:.1f}")

    return {
        "raw":        raw,
        "gray":       gray,
        "blurred":    blurred,
        "binary":     binary,
        "mask":       mask,
        "thresh_val": thresh_val,
    }


if __name__ == "__main__":
    import os

    # run on the first synthetic image we generated
    image_path = os.path.join("data", "sample_images", "images", "frame_0000.png")
    result = preprocess(image_path)

    # save each stage so we can visually inspect the pipeline
    cv2.imwrite("outputs/gray.png",    result["gray"])
    cv2.imwrite("outputs/blurred.png", result["blurred"])
    cv2.imwrite("outputs/binary.png",  result["binary"])
    cv2.imwrite("outputs/mask.png",    result["mask"])

    print(f"[Preprocess] Stages saved to outputs/")
    print(f"[Preprocess] Mask shape: {result['mask'].shape}")
    print(f"[Preprocess] White pixels in mask: {result['mask'].sum() // 255}")