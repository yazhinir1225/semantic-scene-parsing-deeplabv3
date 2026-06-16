"""
app.py — Streamlit Web Application
====================================
PURPOSE:
    User-facing web interface for semantic scene parsing.
    Users can:
      1. Upload any street-scene image
      2. See the color-coded segmentation overlay in real time
      3. View per-class pixel coverage as a bar chart
      4. Download the segmentation result

USAGE:
    streamlit run app.py -- --checkpoint checkpoints/best_model.pth

    Or set the checkpoint path via the sidebar in the UI.

DESIGN NOTES:
    - Model is cached with @st.cache_resource so it loads only once per session
    - Handles missing checkpoints gracefully with demo mode using random colors
    - Fully responsive layout with two-column display
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go

# Add src/ to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from label_map import (
    colorize_label_map,
    TRAIN_ID_TO_NAME,
    TRAIN_ID_TO_COLOR,
    NUM_CLASSES,
    IGNORE_INDEX,
)


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Semantic Scene Parsing — Cityscapes",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Model loading (cached — runs only once)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading segmentation model…")
def load_predictor(checkpoint_path: str):
    """
    Load the SegmentationPredictor and cache it for the session lifetime.
    Returns None if checkpoint is not found (demo mode).
    """
    from inference import SegmentationPredictor

    ckpt = Path(checkpoint_path)
    if not ckpt.exists():
        return None

    try:
        predictor = SegmentationPredictor(checkpoint_path, device="auto")
        return predictor
    except Exception as e:
        st.error(f"Failed to load checkpoint: {e}")
        return None


# ---------------------------------------------------------------------------
# Demo mode: random colorized output when no checkpoint is available
# ---------------------------------------------------------------------------

def demo_predict(image: Image.Image) -> dict:
    """
    Generate a fake segmentation result for UI demonstration when no
    trained checkpoint is available.
    """
    import numpy as np
    from label_map import colorize_label_map

    w, h = 512, 256
    # Simple demo: divide image into horizontal bands to simulate road/sky/buildings
    demo_label = np.zeros((h, w), dtype=np.uint8)
    demo_label[:int(h * 0.25), :]  = 10   # sky
    demo_label[int(h * 0.25):int(h * 0.55), :] = 2   # building
    demo_label[int(h * 0.55):, :]  = 0    # road
    # Add some sidewalk
    demo_label[int(h * 0.7):, int(w * 0.1):int(w * 0.9)] = 1  # sidewalk strip

    pred_color = colorize_label_map(demo_label)
    resized = np.array(image.convert("RGB").resize((w, h), Image.BILINEAR))
    alpha = 0.55
    overlay = ((1 - alpha) * resized + alpha * pred_color).clip(0, 255).astype(np.uint8)

    coverage = {}
    total = demo_label.size
    for cls_id in np.unique(demo_label):
        cnt = np.sum(demo_label == cls_id)
        name = TRAIN_ID_TO_NAME.get(int(cls_id), f"class_{cls_id}")
        coverage[name] = round(float(cnt) / total, 4)

    return {
        "pred_color":    pred_color,
        "overlay":       overlay,
        "class_coverage": coverage,
    }


# ---------------------------------------------------------------------------
# Color legend component
# ---------------------------------------------------------------------------

def render_legend() -> None:
    """Render the Cityscapes 19-class color legend in the sidebar."""
    st.sidebar.markdown("### Class Color Legend")

    legend_items = []
    for cls_id in range(NUM_CLASSES):
        name = TRAIN_ID_TO_NAME.get(cls_id, f"class_{cls_id}")
        r, g, b = TRAIN_ID_TO_COLOR.get(cls_id, (128, 128, 128))
        color_hex = f"#{r:02x}{g:02x}{b:02x}"
        legend_items.append(
            f'<span style="background:{color_hex};display:inline-block;'
            f'width:14px;height:14px;margin-right:6px;border-radius:2px;"></span>'
            f'{name}'
        )

    legend_html = "<br>".join(legend_items)
    st.sidebar.markdown(
        f'<div style="line-height:2;">{legend_html}</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Coverage bar chart
# ---------------------------------------------------------------------------

def render_coverage_chart(coverage: dict) -> None:
    """Render a horizontal bar chart of detected class coverage."""
    if not coverage:
        return

    classes = list(coverage.keys())
    values  = [v * 100 for v in coverage.values()]
    colors  = []

    for cls_name in classes:
        # Find trainId by name
        train_id = next(
            (k for k, v in TRAIN_ID_TO_NAME.items() if v == cls_name),
            None,
        )
        r, g, b = TRAIN_ID_TO_COLOR.get(train_id, (128, 128, 128)) if train_id is not None else (128, 128, 128)
        colors.append(f"rgb({r},{g},{b})")

    fig = go.Figure(go.Bar(
        x=values,
        y=classes,
        orientation="h",
        marker_color=colors,
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        title="Detected Class Coverage (%)",
        xaxis_title="% of image pixels",
        height=max(250, len(classes) * 30),
        margin=dict(l=120, r=40, t=40, b=30),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12),
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main() -> None:
    # ---- Header ----
    st.title("🏙️ Semantic Scene Parsing")
    st.markdown(
        "Upload a street-scene image to get a **pixel-level semantic segmentation** "
        "using a DeepLabV3 model trained on the Cityscapes dataset (19 classes)."
    )
    st.divider()

    # ---- Sidebar: configuration ----
    st.sidebar.title("⚙️ Settings")

    checkpoint_path = st.sidebar.text_input(
        "Checkpoint path",
        value="checkpoints/best_model.pth",
        help="Path to the trained .pth file",
    )

    overlay_alpha = st.sidebar.slider(
        "Overlay transparency",
        min_value=0.1, max_value=1.0,
        value=0.55, step=0.05,
        help="Blend weight of the segmentation color map over the image",
    )

    st.sidebar.divider()
    render_legend()

    # ---- Load model ----
    predictor = load_predictor(checkpoint_path)

    if predictor is None:
        st.warning(
            f"⚠️  Checkpoint not found at **{checkpoint_path}**.\n\n"
            "Running in **demo mode** — predictions are illustrative only.\n\n"
            "To use real predictions:\n"
            "1. Train the model:  `python src/train.py`\n"
            "2. Enter the checkpoint path in the sidebar."
        )
        use_demo = True
    else:
        use_demo = False
        st.success(f"✅ Model loaded from `{checkpoint_path}`")

    # ---- File uploader ----
    uploaded = st.file_uploader(
        "Upload a street-scene image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        help="For best results use images from urban/driving scenes",
    )

    # ---- Sample image fallback ----
    if uploaded is None:
        st.info("👆 Upload an image to see the segmentation result.")

        # Show example images if available
        sample_dir = Path("sample_images")
        if sample_dir.exists():
            sample_files = list(sample_dir.glob("*.jpg")) + list(sample_dir.glob("*.png"))
            if sample_files:
                st.markdown("**Or try a sample image:**")
                cols = st.columns(min(3, len(sample_files)))
                for i, sf in enumerate(sample_files[:3]):
                    with cols[i]:
                        img = Image.open(sf)
                        if st.button(sf.name, key=f"sample_{i}"):
                            uploaded = sf   # use sample
                        st.image(img, use_container_width=True)
        return

    # ---- Load uploaded image ----
    if isinstance(uploaded, Path):
        image = Image.open(uploaded)
        img_name = uploaded.name
    else:
        image = Image.open(uploaded)
        img_name = uploaded.name

    # ---- Run inference ----
    with st.spinner("Running segmentation…"):
        if use_demo:
            results = demo_predict(image)
        else:
            results = predictor.predict(image)

        # Re-blend with user's alpha choice
        from label_map import colorize_label_map
        resized_orig = np.array(
            image.convert("RGB").resize((512, 256), Image.BILINEAR),
            dtype=np.float32,
        )
        pred_color = results["pred_color"]
        overlay = (
            (1 - overlay_alpha) * resized_orig
            + overlay_alpha * pred_color.astype(np.float32)
        ).clip(0, 255).astype(np.uint8)

    # ---- Display results ----
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📷 Input Image")
        st.image(image, use_container_width=True, caption=img_name)

    with col2:
        st.subheader("🎨 Segmentation Overlay")
        st.image(overlay, use_container_width=True, caption="Color-coded prediction")

    st.divider()

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("🗺️ Segmentation Map")
        st.image(pred_color, use_container_width=True, caption="Pure class colors")

    with col4:
        st.subheader("📊 Class Coverage")
        render_coverage_chart(results["class_coverage"])

    # ---- Download buttons ----
    st.divider()
    st.subheader("⬇️ Download Results")

    dl_col1, dl_col2, dl_col3 = st.columns(3)

    with dl_col1:
        overlay_pil = Image.fromarray(overlay)
        from io import BytesIO
        buf_overlay = BytesIO()
        overlay_pil.save(buf_overlay, format="PNG")
        st.download_button(
            "Download Overlay",
            data=buf_overlay.getvalue(),
            file_name=f"{Path(img_name).stem}_overlay.png",
            mime="image/png",
        )

    with dl_col2:
        buf_color = BytesIO()
        Image.fromarray(pred_color).save(buf_color, format="PNG")
        st.download_button(
            "Download Color Map",
            data=buf_color.getvalue(),
            file_name=f"{Path(img_name).stem}_segmentation.png",
            mime="image/png",
        )

    with dl_col3:
        import json
        coverage_json = json.dumps(results["class_coverage"], indent=2)
        st.download_button(
            "Download Coverage JSON",
            data=coverage_json,
            file_name=f"{Path(img_name).stem}_coverage.json",
            mime="application/json",
        )

    # ---- Model info expander ----
    with st.expander("ℹ️ About this model"):
        st.markdown("""
**Architecture:** DeepLabV3 with ResNet-50 backbone  
**Training data:** Cityscapes (2975 training images)  
**Input resolution:** 512×256 pixels (preserves 2:1 aspect ratio)  
**Classes:** 19 Cityscapes trainId classes  
**Loss:** Weighted Cross-Entropy with median-frequency class weighting  
**Optimizer:** SGD + Polynomial LR decay (power=0.9)

**Why DeepLabV3?**  
Atrous (dilated) convolutions and Atrous Spatial Pyramid Pooling (ASPP) let the model
capture context at multiple scales without losing resolution — critical for distinguishing
small objects (traffic lights) from large regions (sky, road) in the same image.
        """)


if __name__ == "__main__":
    main()
