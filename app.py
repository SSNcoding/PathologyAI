"""PathologyAI research preview.

Run: streamlit run app.py

This interface performs image preview and simple tissue-region detection only.
Attach a validated, versioned inference service at `run_validated_model` before
displaying clinical model outputs.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import cv2
import numpy as np
import traceback
import streamlit as st
from PIL import Image, ImageEnhance, ImageOps

try:
    from groq import Groq
except ImportError:
    Groq = None


st.set_page_config(
    page_title="PathologyAI | Research Preview",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
      .block-container { max-width: 1440px; padding-top: 2rem; padding-bottom: 2rem; }
      [data-testid="stMetric"] { background: #111827; border: 1px solid #263344; border-radius: 12px; padding: 14px; }
      .eyebrow { color: #62d3be; font-weight: 700; letter-spacing: .08em; font-size: .78rem; text-transform: uppercase; }
      .panel { border: 1px solid #263344; border-radius: 14px; padding: 1.1rem; background: #101827; }
      .muted { color: #9aa8b8; }
    </style>
    """,
    unsafe_allow_html=True,
)


@dataclass(frozen=True)
class Region:
    x: int
    y: int
    width: int
    height: int
    area: int


def load_preview(uploaded_file, max_size: tuple[int, int] = (2200, 2200)) -> Image.Image:
    """
    Safely load uploaded pathology image.
    """

    try:
        if uploaded_file is None:
            raise ValueError("No file uploaded")
        image = Image.open(uploaded_file)
        image.seek(0)
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")
        image.thumbnail(max_size)
        return image
    
    except Exception as e:

        raise ValueError(
            f"Could not read image file: {e}"
        )

def preprocess_slide(image: Image.Image) -> np.ndarray:
    
    try: 
        if image is None:
            raise ValueError(
                "No image was loaded"
            )
        image = image.convert(
            "RGB"
        )

        image = ImageEnhance.Contrast(
            image
        ).enhance(1.15)

        image_array = np.array(
            image,
            dtype=np.uint8
        )

        if image_array.ndim != 3:
            raise ValueError(
                "Invalid image dimensions"
            )

        if image_array.shape[2] != 3:
            raise ValueError(
                "Image is not RGB"
            )
        
        return image_array
    except Exception as e:
        raise ValueError(
            f"Image preprocessing failed: {e}"
        )

def tissue_mask(image_rgb: np.ndarray) -> np.ndarray:
    """
    Creates a tissue mask from RGB image.
    """
    if not isinstance(image_rgb, np.ndarray):
        image_rgb = np.array(image_rgb)

    if image_rgb.dtype != np.uint8:
        image_rgb = image_rgb.astype(np.uint8)

    if len(image_rgb.shape) != 3 or image_rgb.shape[2] != 3:
        raise ValueError(
            f"Expected RGB image but received shape {image_rgb.shape}"
        )

    image_rgb = np.ascontiguousarray(
        image_rgb,
        dtype=np.uint8
    )
    
    hsv = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2HSV
    )
    saturation = hsv[:, :, 1]

    _, mask = cv2.threshold(
        saturation,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    return mask
def detect_tissue_regions(mask: np.ndarray, min_area: int = 500) -> list[Region]:
    if mask is None:
        return []
    label_count, _, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8
    )
    return [
        Region(int(x), int(y), int(width), int(height), int(area))
        for x, y, width, height, area in stats[1:label_count]
        if area >= min_area
    ]


def region_overlay(image_rgb: np.ndarray, regions: list[Region]) -> np.ndarray:
    overlay = image_rgb.copy()
    for region in regions:
        cv2.rectangle(
            overlay,
            (region.x, region.y),
            (region.x + region.width, region.y + region.height),
            (255, 92, 92),
            2,
        )
    return overlay


def density_overlay(image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Image-derived tissue-density overlay; explicitly not model attention."""
    density = cv2.GaussianBlur(mask, (0, 0), sigmaX=25)
    density = cv2.normalize(density, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap = cv2.applyColorMap(density, cv2.COLORMAP_TURBO)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(image_rgb, 0.74, heatmap, 0.26, 0)


def get_groq_key() -> str | None:
    key = os.getenv("GROQ_API_KEY")
    if key:
        return key
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return None


@st.cache_resource
def groq_client(api_key: str):
    return Groq(api_key=api_key)


def chat_reply(question: str) -> str:
    api_key = get_groq_key()
    if not api_key or Groq is None:
        return "Chat is not configured. Add GROQ_API_KEY to .streamlit/secrets.toml."
    try:
        response = groq_client(api_key).chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Provide concise, general educational pathology information. "
                        "Do not diagnose, recommend treatment, or infer patient-specific results. "
                        "Remind users not to share patient-identifiable information."
                    ),
                },
                {"role": "user", "content": question},
            ],
        )
        return response.choices[0].message.content
    except Exception:
        return "The chat service is unavailable. Check the configured key and connection."


# ---- Sidebar ----
with st.sidebar:
    st.header("System parameters")
    model_version = st.selectbox("Model backbone", ["EfficientNet-B7", "Vision Transformer (ViT)"])
    confidence_threshold = st.slider("Review threshold", 0.50, 0.99, 0.85, 0.01)
    analysis_mode = st.radio("Analysis mode", ["Triage", "Diagnostic review"])
    st.caption("These settings are saved for the future validated inference service; this preview does not diagnose.")

    st.divider()
    st.subheader("💬 Educational assistant")
    st.caption("Do not enter patient-identifiable information.")
    question = st.text_area("Ask a general pathology question", key="chat_question", height=90)
    if st.button("Send question", use_container_width=True):
        if question.strip():
            st.session_state["last_chat_reply"] = chat_reply(question.strip())
        else:
            st.warning("Enter a question first.")
    if answer := st.session_state.get("last_chat_reply"):
        st.info(answer)


# ---- Main header ----
st.markdown('<div class="eyebrow">Digital pathology research workspace</div>', unsafe_allow_html=True)
st.title("🔬 PathologyAI")
st.markdown(
    "A whole-slide image review experience for image quality checks, tissue-region visualization, "
    "and future model-assisted pathologist review."
)
st.warning(
    "Research preview only — no validated diagnostic model is connected. "
    "Do not use this interface for patient-care decisions.",
    icon="⚠️",
)

left, right = st.columns([1.3, 0.7], gap="large")

with left:
    st.subheader("Whole-slide image preview")
    uploaded_file = st.file_uploader(
        "Upload a slide image",
        type=[
            "jpg",
            "jpeg",
            "png",
        ],
    )

with right:
    st.subheader("Analysis status")
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("**Selected workflow**")
    st.write(f"{analysis_mode} · {model_version}")
    st.markdown("**Review threshold**")
    st.write(f"{confidence_threshold:.0%}")
    st.markdown("**Clinical model**")
    st.error("Not connected — no prediction is shown", icon="ℹ️")
    st.markdown("</div>", unsafe_allow_html=True)


if uploaded_file is None:
    st.info("Upload a jpg, jpeg, or png file to begin image-preview analysis.", icon="🔬")
else:
    try:
        if uploaded_file.size == 0:
            st.error("Uploaded file is empty. ")
            st.stop()
        if uploaded_file.size > 50_000_000:
            st.warning(
                "Large image detected. Processing may take longer."
            )
        loaded_image = load_preview(
            uploaded_file
        )
        st.success("Image loaded successfully!")
        image_rgb = preprocess_slide(
            loaded_image
        )
        mask = tissue_mask(
            image_rgb
        )
        regions = detect_tissue_regions(mask)
        area_fraction = float((mask > 0).mean())

        st.divider()
        metric_1, metric_2, metric_3, metric_4 = st.columns(4)
        metric_1.metric("Preview dimensions", f"{image_rgb.shape[1]} × {image_rgb.shape[0]}")
        metric_2.metric("Tissue coverage", f"{area_fraction:.1%}")
        metric_3.metric("Tissue regions", len(regions))
        metric_4.metric("Model status", "Not connected")

        original_col, regions_col = st.columns(2, gap="large")
        with original_col:
            st.markdown("#### Processed tissue preview")
            st.image(image_rgb, use_container_width=True)
        with regions_col:
            st.markdown("#### Tissue-region overlay")
            st.image(region_overlay(image_rgb, regions), use_container_width=True)
            st.caption("Bounding boxes indicate tissue-like connected regions, not cells or tumors.")

        st.divider()
        st.subheader("Image-derived visualization")
        viz_col, detail_col = st.columns([1.3, 0.7], gap="large")
        with viz_col:
            st.image(density_overlay(image_rgb, mask), use_container_width=True)
            st.caption("Tissue-density overlay — not Grad-CAM, model attention, or a malignancy heatmap.")
        with detail_col:
            st.markdown("#### Next model integration")
            st.write(
                "Connect a versioned model service here to produce calibrated, tile-level "
                "probabilities and Grad-CAM artifacts. Persist model/version, preprocessing "
                "configuration, tile coordinates, and reviewer actions with each run."
            )
            st.code(
                "# prediction = validated_service.infer(slide_id)\n"
                "# route_case(prediction.calibrated_probability, ...)\n"
                "# Require pathologist review before sign-out",
                language="python",
            )

        with st.expander("📝 Research review notes"):
            st.text_area("Notes for the current session", placeholder="Add observations for human review.")
            st.caption("A production report must be signed out through the approved pathology workflow.")

    except Exception as error:
        st.error(
            "Image processing failed."
        )

        st.exception(
            error
        )

st.divider()
st.caption("PathologyAI · Research preview · Human expert review required · No patient data should be sent to the chat assistant")
