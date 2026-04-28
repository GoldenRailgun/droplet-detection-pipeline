import numpy as np
from dataclasses import dataclass
from typing import List
from detect import DetectedDroplet


@dataclass
class MeasuredDroplet:
    """
    Detected droplet with real-world size added.
    Keeps pixel measurements alongside micrometre values
    so results can be verified against raw image if needed.
    """
    id: int
    cx: float
    cy: float
    radius_px: float
    diameter_px: float
    circularity: float
    diameter_um: float      # real-world diameter in micrometres


def apply_calibration(
    droplets: List[DetectedDroplet],
    pixels_per_um: float = 0.5,
) -> List[MeasuredDroplet]:
    """
    Convert pixel measurements to real-world micrometres.
    pixels_per_um comes from a calibration target in the lab.
    Default 0.5 matches our synthetic data generator.
    In production: measure a reference object of known size in the image
    and pass the calculated ratio here.
    """
    if pixels_per_um <= 0:
        raise ValueError("pixels_per_um must be positive")

    measured = []
    for d in droplets:
        diameter_um = d.diameter_px / pixels_per_um

        measured.append(MeasuredDroplet(
            id=d.id,
            cx=d.cx,
            cy=d.cy,
            radius_px=d.radius_px,
            diameter_px=d.diameter_px,
            circularity=d.circularity,
            diameter_um=round(diameter_um, 2),
        ))

    return measured


def get_diameters_um(measured: List[MeasuredDroplet]) -> np.ndarray:
    """
    Extract diameter array for statistical analysis in analyse.py.
    Returns numpy array — ready for SciPy and Matplotlib directly.
    """
    return np.array([d.diameter_um for d in measured])



if __name__ == "__main__":
    import os
    from preprocess import preprocess
    from detect import detect_droplets

    image_path = os.path.join("data", "sample_images", "images", "frame_0000.png")
    
    result   = preprocess(image_path)
    droplets = detect_droplets(result["mask"])
    measured = apply_calibration(droplets, pixels_per_um=0.5)
    diameters = get_diameters_um(measured)

    print(f"[Measure] Droplets measured: {len(measured)}")
    print(f"[Measure] Diameter range: {diameters.min():.1f} - {diameters.max():.1f} um")
    print(f"[Measure] Mean diameter : {diameters.mean():.1f} um")
    for d in measured[:5]:
        print(f"  ID:{d.id} | {d.diameter_px}px | {d.diameter_um}um")