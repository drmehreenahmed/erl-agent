# app.py
import streamlit as st
import os
import cv2
import numpy as np
from PIL import Image

# Local modules
from config import YOLO_MODEL_PATH
from detection import detect_scale_bars, load_yolo_model
from ocr_utils import ScaleExtractor
from memory import HeuristicMemory
from agent import run_agent
from ui import inject_css, render_trajectory, render_heuristic, create_sidebar

# Session state initialisation
if "heuristic_memory" not in st.session_state:
    st.session_state.heuristic_memory = None
if "clear_memory_flag" not in st.session_state:
    st.session_state.clear_memory_flag = False

st.set_page_config(
    page_title="SEM Scale Bar Analyst (ERL)",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

inject_css()

@st.cache_resource
def load_models():
    """Load YOLO, OCR extractor and heuristic memory."""
    if not os.path.exists(YOLO_MODEL_PATH):
        st.error(f"YOLO model '{YOLO_MODEL_PATH}' not found.")
        st.stop()
    yolo = load_yolo_model(YOLO_MODEL_PATH)
    ocr = ScaleExtractor()
    memory = HeuristicMemory()
    return {"yolo": yolo, "ocr": ocr, "memory": memory}

def main():
    st.title("🔬 ERL-Style Self‑Learning SEM Scale Bar Agent")
    st.markdown("**Heuristic learning with Trigger→Action rules **")

    custom_query = create_sidebar()

    # Handle memory clearing
    if st.session_state.clear_memory_flag:
        st.session_state.heuristic_memory = HeuristicMemory()
        st.session_state.clear_memory_flag = False
        st.success("Heuristic memory cleared!")

    models = load_models()
    if st.session_state.heuristic_memory is None:
        st.session_state.heuristic_memory = models["memory"]

    uploaded_file = st.file_uploader("📂 Upload an SEM image", type=["png", "jpg", "jpeg", "tif", "bmp"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)

        # Detection
        with st.spinner("🔍 Detecting scale bar..."):
            detections = detect_scale_bars(models["yolo"], image_np)

        if detections is not None and len(detections) > 0:
            best = max(detections, key=lambda d: d[4])
            x1, y1, x2, y2, conf, _ = best
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            polygon = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

            # OCR
            with st.spinner("📝 Reading scale text..."):
                ocr_result = models["ocr"].extract_scale_value(image_np, polygon)

            det_info = {"confidence": float(conf)}
            ocr_info = {
                "value": ocr_result.get("value"),
                "unit": ocr_result.get("unit"),
                "confidence": ocr_result.get("confidence", 0.0)
            }

            # Visualisation
            vis_image = image_np.copy()
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
            if ocr_info["value"] and ocr_info["unit"]:
                overlay_text = f"{ocr_info['value']} {ocr_info['unit']}"
                cv2.putText(vis_image, overlay_text, (x1, max(y1-10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)

            # Run agent
            with st.spinner("🧠 Running self‑learning agent (ERL style)..."):
                agent_result = run_agent(
                    image_np, ocr_info, det_info,
                    custom_query if custom_query.strip() else "Extract scale bar correctly",
                    st.session_state.heuristic_memory
                )

            # Tabs
            tab1, tab2, tab3 = st.tabs(["🖼️ Visual Analysis", "🧠 Self‑Learning Evolution", "📚 Heuristic Memory"])

            with tab1:
                col_img, col_metrics = st.columns([2, 1])
                with col_img:
                    st.image(vis_image, caption="Detected scale bar (green box) with OCR overlay", use_container_width=True)
                with col_metrics:
                    st.metric("Detection Confidence", f"{det_info['confidence']:.1%}")
                    if ocr_info["value"] and ocr_info["unit"]:
                        st.markdown(f"""
                        <div class="ocr-highlight-card">
                            <span style="color:#aaa; font-weight:600;">📐 RAW OCR SCALE</span><br>
                            <span class="ocr-value">{ocr_info['value']}</span>
                            <span class="ocr-unit">{ocr_info['unit']}</span>
                            <br><span style="color:#ccc;">OCR confidence: {ocr_info['confidence']:.1%}</span>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.warning("OCR could not extract a valid value/unit.")

            with tab2:
                st.markdown("### 🔄 Self‑Learning Evolution")
                col_base, col_refined = st.columns(2)
                with col_base:
                    render_trajectory(agent_result["base"], "🧠 Base Reasoning")
                with col_refined:
                    render_trajectory(agent_result["refined"], "🔄 Refined (with Heuristics)")
                st.metric("Outcome", agent_result["outcome"])
                with st.expander("📄 Raw LLM outputs"):
                    st.code("BASE:\n" + agent_result["base"])
                    st.code("REFINED:\n" + agent_result["refined"])

            with tab3:
                st.markdown("### 📚 Learned Heuristics (Trigger → Action)")
                st.markdown("**Newly learned from this image:**")
                render_heuristic(agent_result["heuristic"])
                if agent_result["retrieved_heuristics"]:
                    st.markdown("**Retrieved and applied heuristics:**")
                    for i, h in enumerate(agent_result["retrieved_heuristics"]):
                        with st.expander(f"Heuristic {i+1}"):
                            render_heuristic(h["heuristic"])
                st.markdown("---")
                st.markdown("**All stored heuristics in memory:**")
                for i, h in enumerate(st.session_state.heuristic_memory.heuristics):
                    with st.expander(f"Heuristic {i+1}: {h['task'][:80]}..."):
                        render_heuristic(h["heuristic"])
        else:
            st.warning("No scale bar detected. Try a different image.")
    else:
        st.info("👈 Upload an SEM image to begin analysis.")

if __name__ == "__main__":
    main()