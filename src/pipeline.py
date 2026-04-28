import os
import json
import pandas as pd
from dataclasses import asdict

from preprocess import preprocess
from detect import detect_droplets
from measure import apply_calibration, get_diameters_um
from analyse import compute_statistics, print_statistics, plot_distribution

def run_pipeline(
    image_path: str,
    pixels_per_um: float = 0.5,
    output_dir: str = "outputs",
) -> dict:
    """
    Run full detection pipeline on a single image.
    Returns stats dict so results can be aggregated across multiple images.
    """
    print(f"\n[Pipeline] Processing: {image_path}")

    # Stage 1 — preprocess
    preprocessed = preprocess(image_path)

    # Stage 2 — detect
    droplets = detect_droplets(preprocessed["mask"])
    print(f"[Pipeline] Detected: {len(droplets)} droplets")

    # Stage 3 — measure
    measured  = apply_calibration(droplets, pixels_per_um=pixels_per_um)
    diameters = get_diameters_um(measured)

    if len(diameters) == 0:
        print("[Pipeline] WARNING: No droplets detected — check preprocessing parameters")
        return {}

    # Stage 4 — analyse
    stats_dict = compute_statistics(diameters)
    print_statistics(stats_dict)

    # save histogram
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    hist_path  = os.path.join(output_dir, "distributions", f"{image_name}_distribution.png")
    plot_distribution(diameters, stats_dict, output_path=hist_path)

    # save annotated image
    annotated_path = os.path.join(output_dir, "annotated", f"{image_name}_annotated.png")
    save_annotated_image(preprocessed["raw"], measured, annotated_path)

    # save CSV
    csv_path = os.path.join(output_dir, f"{image_name}_results.csv")
    save_csv(measured, csv_path)

    return stats_dict

def save_annotated_image(raw, measured, output_path: str) -> None:
    """
    Draw detected droplets on original image.
    Green circle = detected droplet boundary.
    Red dot = centre point.
    White text = diameter in micrometres.
    """
    import cv2
    import numpy as np

    # convert grayscale to BGR so we can draw coloured annotations
    if len(raw.shape) == 2:
        annotated = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
    else:
        annotated = raw.copy()

    for d in measured:
        cx, cy = int(d.cx), int(d.cy)
        r      = int(d.radius_px)

        # draw circle boundary
        cv2.circle(annotated, (cx, cy), r, (0, 255, 0), 2)
        # draw centre dot
        cv2.circle(annotated, (cx, cy), 3, (0, 0, 255), -1)
        # label diameter
        cv2.putText(
            annotated,
            f"{d.diameter_um:.0f}um",
            (cx - 20, cy - r - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4, (255, 255, 255), 1
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, annotated)
    print(f"[Pipeline] Annotated image saved -> {output_path}")


def save_csv(measured, output_path: str) -> None:
    """
    Export all detected droplets to CSV.
    One row per droplet — ready for Excel or further analysis.
    """
    df = pd.DataFrame([asdict(d) for d in measured])
    df.to_csv(output_path, index=False)
    print(f"[Pipeline] CSV saved -> {output_path}")

if __name__ == "__main__":
    import glob

    # run on all synthetic images
    image_paths = sorted(glob.glob("data/sample_images/images/*.png"))

    if not image_paths:
        print("[Pipeline] No images found in data/sample_images/images/")
        exit(1)

    all_stats = []
    for image_path in image_paths:
        stats = run_pipeline(image_path, pixels_per_um=0.5)
        if stats:
            all_stats.append(stats)

    # summary across all images
    if all_stats:
        import numpy as np
        mean_d50s = np.mean([s["d50"] for s in all_stats])
        print(f"\n[Pipeline] Processed {len(all_stats)} images")
        print(f"[Pipeline] Mean D50 across all frames: {mean_d50s:.1f} um")