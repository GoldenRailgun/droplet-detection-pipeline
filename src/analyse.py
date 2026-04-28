import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from typing import List
import os
from measure import MeasuredDroplet, get_diameters_um


def compute_statistics(diameters: np.ndarray) -> dict:
    """
    D10/D50/D90 are standard percentile metrics used in spray research.
    Span measures distribution width — smaller span = more consistent spray.
    """
    d10 = np.percentile(diameters, 10)
    d50 = np.percentile(diameters, 50)
    d90 = np.percentile(diameters, 90)
    span = (d90 - d10) / d50

    return {
        "count":  len(diameters),
        "mean":   round(float(np.mean(diameters)), 2),
        "std":    round(float(np.std(diameters)), 2),
        "min":    round(float(diameters.min()), 2),
        "max":    round(float(diameters.max()), 2),
        "d10":    round(float(d10), 2),
        "d50":    round(float(d50), 2),
        "d90":    round(float(d90), 2),
        "span":   round(float(span), 3),
    }


def print_statistics(stats_dict: dict) -> None:
    """Print formatted statistical summary to console."""
    print("\n[Analyse] Size Distribution Summary")
    print(f"  Droplets analysed : {stats_dict['count']}")
    print(f"  Mean diameter     : {stats_dict['mean']} um")
    print(f"  Std deviation     : {stats_dict['std']} um")
    print(f"  Min / Max         : {stats_dict['min']} / {stats_dict['max']} um")
    print(f"  D10               : {stats_dict['d10']} um")
    print(f"  D50 (median)      : {stats_dict['d50']} um")
    print(f"  D90               : {stats_dict['d90']} um")
    print(f"  Span (D90-D10)/D50: {stats_dict['span']}")


def plot_distribution(
    diameters: np.ndarray,
    stats_dict: dict,
    output_path: str = "outputs/distributions/size_distribution.png",
) -> None:
    """
    Plot droplet size distribution histogram with D10/D50/D90 markers.
    Saved as PNG — ready for research report or presentation.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # histogram
    ax.hist(diameters, bins="auto", color="steelblue", edgecolor="black", alpha=0.7)

    # D10/D50/D90 vertical lines
    ax.axvline(stats_dict["d10"], color="green",  linestyle="--", linewidth=1.5, label=f"D10 = {stats_dict['d10']} µm")
    ax.axvline(stats_dict["d50"], color="red",    linestyle="--", linewidth=1.5, label=f"D50 = {stats_dict['d50']} µm")
    ax.axvline(stats_dict["d90"], color="orange", linestyle="--", linewidth=1.5, label=f"D90 = {stats_dict['d90']} µm")

    ax.set_xlabel("Droplet Diameter (µm)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Droplet Size Distribution", fontsize=14)
    ax.legend(fontsize=11)

    # add span annotation
    ax.text(
        0.98, 0.95,
        f"Span = {stats_dict['span']}",
        transform=ax.transAxes,
        fontsize=10,
        ha="right", va="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[Analyse] Histogram saved -> {output_path}")

if __name__ == "__main__":
    import os
    from preprocess import preprocess
    from detect import detect_droplets
    from measure import apply_calibration, get_diameters_um

    image_path = os.path.join("data", "sample_images", "images", "frame_0000.png")

    result    = preprocess(image_path)
    droplets  = detect_droplets(result["mask"])
    measured  = apply_calibration(droplets, pixels_per_um=0.5)
    diameters = get_diameters_um(measured)

    stats_dict = compute_statistics(diameters)
    print_statistics(stats_dict)
    plot_distribution(diameters, stats_dict)