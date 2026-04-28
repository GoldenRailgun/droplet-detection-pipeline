import cv2
import numpy as np
from dataclasses import dataclass
from typing import List


@dataclass
class DetectedDroplet:
    """One detected droplet in pixel units."""
    id: int
    cx: float        # centre x
    cy: float        # centre y
    radius_px: float
    diameter_px: float
    circularity: float

def detect_droplets(
    mask: np.ndarray,
    min_area_px: float = 50.0,
    min_circularity: float = 0.70,
    border_margin: int = 5,
) -> List[DetectedDroplet]:
    """
    Find droplets in a binary mask using contour detection.
    Filters by minimum area and circularity to reject noise and merged blobs.
    Rejects droplets too close to image border — partial droplets distort sizing.
    """
    h, w = mask.shape[:2]

    # RETR_EXTERNAL: only outer contours, ignores holes inside blobs
    # CHAIN_APPROX_SIMPLE: compresses contour points, saves memory
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    droplets: List[DetectedDroplet] = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area_px:
            continue  # too small — noise speck

        perimeter = cv2.arcLength(cnt, closed=True)
        if perimeter == 0:
            continue  # degenerate contour — skip

        circularity = (4 * np.pi * area) / (perimeter ** 2)
        if circularity < min_circularity:
            continue  # not circular enough — merged or irregular blob

        (cx, cy), radius = cv2.minEnclosingCircle(cnt)

        # reject partial droplets at image border
        if (cx - radius < border_margin or cx + radius > w - border_margin or
                cy - radius < border_margin or cy + radius > h - border_margin):
            continue

        droplets.append(DetectedDroplet(
            id=len(droplets),
            cx=round(float(cx), 2),
            cy=round(float(cy), 2),
            radius_px=round(float(radius), 2),
            diameter_px=round(float(radius) * 2, 2),
            circularity=round(float(circularity), 4),
        ))

    return droplets


if __name__ == "__main__":
    import os
    from preprocess import preprocess

    image_path = os.path.join("data", "sample_images", "images", "frame_0000.png")
    result = preprocess(image_path)
    droplets = detect_droplets(result["mask"])

    print(f"[Detect] Droplets found: {len(droplets)}")
    for d in droplets[:5]:  # print first 5 only
        print(f"  ID:{d.id} | centre:({d.cx},{d.cy}) | diameter:{d.diameter_px}px | circularity:{d.circularity}")