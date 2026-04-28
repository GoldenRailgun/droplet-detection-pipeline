import streamlit as st
import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# add src to path so we can import our pipeline modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from preprocess import preprocess
from detect import detect_droplets
from measure import apply_calibration, get_diameters_um
from analyse import compute_statistics, plot_distribution

st.set_page_config(
    page_title="Droplet Detection Pipeline",
    page_icon="💧",
    layout="wide",
)

st.title("💧 Droplet Detection & Size Distribution Pipeline")
st.markdown("> Classical computer vision pipeline for spray droplet detection and sizing.")

st.sidebar.header("⚙️ Parameters")

pixels_per_um = st.sidebar.slider(
    "Calibration (pixels per µm)",
    min_value=0.1,
    max_value=2.0,
    value=0.5,
    step=0.1,
    help="Conversion factor from pixels to micrometres. Derived from a known reference object."
)

min_circularity = st.sidebar.slider(
    "Min Circularity",
    min_value=0.3,
    max_value=1.0,
    value=0.7,
    step=0.05,
    help="Droplets scoring below this are rejected as merged blobs or noise."
)

blur_kernel = st.sidebar.select_slider(
    "Blur Kernel Size",
    options=[3, 5, 7, 9],
    value=5,
    help="Gaussian blur strength. Increase if noise survives thresholding."
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Khush Patel** — Sheridan College")
st.sidebar.markdown("[GitHub](https://github.com/GoldenRailgun/droplet-detection-pipeline)")

st.markdown("---")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Input")
    
    source = st.radio(
        "Image source",
        ["Use synthetic sample", "Upload your own image"],
    )

    if source == "Upload your own image":
        uploaded = st.file_uploader("Upload a droplet image", type=["png", "jpg", "jpeg"])
        if uploaded:
            file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
            input_image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        else:
            input_image = None
    else:
        # load first synthetic image by default
        default_path = os.path.join("data", "sample_images", "images", "frame_0000.png")
        if os.path.exists(default_path):
            input_image = cv2.imread(default_path, cv2.IMREAD_GRAYSCALE)
        else:
            st.warning("Synthetic images not found. Run synthetic_droplets.py first.")
            input_image = None

    if input_image is not None:
        st.image(input_image, caption="Input Image", use_container_width=True)

with col2:
    st.subheader("Results")

    if input_image is not None:
        with st.spinner("Running pipeline..."):
            # save temp image so preprocess can load it
            temp_path = "temp_input.png"
            cv2.imwrite(temp_path, input_image)

            # run pipeline stages
            preprocessed = preprocess(
                temp_path,
                blur_kernel=blur_kernel,
            )
            droplets = detect_droplets(
                preprocessed["mask"],
                min_circularity=min_circularity,
            )
            measured  = apply_calibration(droplets, pixels_per_um=pixels_per_um)
            diameters = get_diameters_um(measured)

            os.remove(temp_path)

        if len(diameters) == 0:
            st.error("No droplets detected. Try lowering the circularity threshold or blur kernel.")
        else:
            # annotated image
            annotated = input_image.copy()
            annotated = cv2.cvtColor(annotated, cv2.COLOR_GRAY2BGR)
            for d in measured:
                cx, cy, r = int(d.cx), int(d.cy), int(d.radius_px)
                cv2.circle(annotated, (cx, cy), r, (0, 255, 0), 2)
                cv2.circle(annotated, (cx, cy), 3, (0, 0, 255), -1)
                cv2.putText(annotated, f"{d.diameter_um:.0f}um",
                    (cx - 20, cy - r - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            st.image(annotated, caption="Detected Droplets", use_container_width=True)

            # stats
            stats = compute_statistics(diameters)
            st.markdown("### Size Distribution Metrics")
            metric_cols = st.columns(4)
            metric_cols[0].metric("Droplets Found", stats["count"])
            metric_cols[1].metric("D50 (median)", f"{stats['d50']} µm")
            metric_cols[2].metric("Span", stats["span"])
            metric_cols[3].metric("Mean", f"{stats['mean']} µm")

            # histogram
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.hist(diameters, bins="auto", color="steelblue", edgecolor="black", alpha=0.7)
            ax.axvline(stats["d10"], color="green",  linestyle="--", linewidth=1.5, label=f"D10={stats['d10']}µm")
            ax.axvline(stats["d50"], color="red",    linestyle="--", linewidth=1.5, label=f"D50={stats['d50']}µm")
            ax.axvline(stats["d90"], color="orange", linestyle="--", linewidth=1.5, label=f"D90={stats['d90']}µm")
            ax.set_xlabel("Droplet Diameter (µm)")
            ax.set_ylabel("Count")
            ax.legend()
            st.pyplot(fig)
            plt.close()
    else:
        st.info("Select an image source on the left to begin.")