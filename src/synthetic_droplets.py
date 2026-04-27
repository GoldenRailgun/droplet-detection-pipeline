import cv2
import numpy as np
import json
import os


from dataclasses import dataclass, asdict
from typing import List, Tuple


@dataclass
class Droplet:
    """Ground truth record for one droplet."""
    id: int
    cx: int           # centre x in pixels
    cy: int           # centre y in pixels
    radius_px: int    # true radius in pixels
    diameter_um: float  # real-world diameter in micrometre

def generate_droplet_image(
    width: int = 640,
    height: int = 480,
    num_droplets: int = 30,
    min_radius: int = 8,
    max_radius: int = 40,
    noise_std: float = 12.0,
    blur_ksize: int = 3,
    pixels_per_um: float = 0.5,
    seed: int = None,
    
    ) -> Tuple[np.ndarray, List[Droplet]]:
    """
    Generate one synthetic droplet image with ground truth.

    Returns
    -------
    image    : grayscale uint8 array (H x W)
    droplets : list of Droplet ground truth objects
    """
    if seed is not None:
        np.random.seed(seed)

    # Dark background with random noise — mimics camera sensor noise
    image = np.random.normal(15, noise_std, (height, width)).clip(0, 255).astype(np.uint8)
    droplets: List[Droplet] = []

    placed = 0
    attempts = 0
    max_attempts = num_droplets * 20

    while placed < num_droplets and attempts < max_attempts:
        attempts += 1
        r = np.random.randint(min_radius, max_radius + 1)
        cx = np.random.randint(r, width - r)
        cy = np.random.randint(r, height - r)

        # Gaussian intensity profile — bright centre, fades toward edge
        # This mimics how real back-lit droplets appear on camera
        y_grid, x_grid = np.ogrid[:height, :width]
        dist = np.sqrt((x_grid - cx) ** 2 + (y_grid - cy) ** 2)
        sigma = r / 2.0
        intensity = 220 * np.exp(-(dist ** 2) / (2 * sigma ** 2))

        # Blend into image — supports overlapping droplets naturally
        image = np.clip(image.astype(np.float32) + intensity, 0, 255).astype(np.uint8)

        diameter_um = (2 * r) / pixels_per_um
        droplets.append(Droplet(
            id=placed,
            cx=cx, cy=cy,
            radius_px=r,
            diameter_um=round(diameter_um, 2)
        ))
        placed += 1

        # Light blur to simulate lens diffraction
    if blur_ksize > 0:
        image = cv2.GaussianBlur(image, (blur_ksize, blur_ksize), 0)

    return image, droplets

def generate_dataset(
    output_dir: str,
    num_images: int = 20,
    droplets_per_image: int = 30,
    seed_start: int = 42,
) -> None:
    """
    Generate multiple images and save with ground truth JSON files.
    
    output_dir/images/ -> PNG files
    output_dir/labels/ -> matching JSON ground truth files
    """
    img_dir = os.path.join(output_dir, "images")
    lbl_dir = os.path.join(output_dir, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    for i in range(num_images):
        img, droplets = generate_droplet_image(
            num_droplets=droplets_per_image,
            seed=seed_start + i      # different seed per image
        )

        img_path = os.path.join(img_dir, f"frame_{i:04d}.png")
        lbl_path = os.path.join(lbl_dir, f"frame_{i:04d}.json")

        cv2.imwrite(img_path, img)
        with open(lbl_path, "w") as f:
            json.dump([asdict(d) for d in droplets], f, indent=2)

    print(f"[DataGen] Generated {num_images} images -> {output_dir}")

if __name__ == "__main__":
    output_dir = "data/sample_images"
    
    generate_dataset(
        output_dir=output_dir,
        num_images=5,
        droplets_per_image=30,
        seed_start=42,
    )

    # Quick stats on the first image
    first_label = os.path.join(output_dir, "labels", "frame_0000.json")
    with open(first_label, "r") as f:
        droplets = json.load(f)

    diameters = [d["diameter_um"] for d in droplets]
    print(f"Droplets generated : {len(droplets)}")
    print(f"Diameter range     : {min(diameters):.1f} - {max(diameters):.1f} um")
    print(f"Mean diameter      : {np.mean(diameters):.1f} um")
    print(f"Median diameter    : {np.median(diameters):.1f} um")