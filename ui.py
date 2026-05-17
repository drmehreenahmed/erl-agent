# ui.py
import streamlit as st
import re
import cv2
from PIL import Image
import numpy as np

# CSS (original styling, kept exactly as in the monolithic script)
CSS = """
<style>
    .stApp { background-color: #fdfcdc; }
    html, body, [class*="css"] { color: #f0f0f0; font-weight: 500; }
    .stButton button {
        background: linear-gradient(135deg, #1e2a3a, #0f1722);
        color: #e0e0e0;
        border-radius: 10px;
        border: 1px solid #2d3e50;
        transition: all 0.3s ease;
        font-weight: 600;
    }
    .stButton button:hover {
        background: linear-gradient(135deg, #2d3e50, #1a2a3a);
        border-color: #4ecdc4;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,173,181,0.3);
        color: white;
    }
    div[data-testid="metric-container"] {
        background: linear-gradient(145deg, #161c24, #0f1419);
        border-radius: 20px;
        padding: 18px;
        border-left: 4px solid #00adb5;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(0,173,181,0.2);
    }
    div[data-testid="metric-container"] label { color: #b0b3c0 !important; font-weight: 600; letter-spacing: 0.5px; }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        color: #4ecdc4 !important;
        font-size: 2.2rem !important;
        font-weight: 800 !important;
        text-shadow: 0 0 2px rgba(0,0,0,0.5);
    }
    .ocr-highlight-card {
        background: linear-gradient(135deg, #1e2a2f, #0f1722);
        border-radius: 20px;
        padding: 20px;
        margin: 10px 0;
        border-left: 6px solid #ffaa33;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        transition: all 0.2s;
    }
    .ocr-highlight-card:hover { transform: scale(1.01); border-left-color: #ffcc66; }
    .ocr-value { font-size: 2.8rem; font-weight: 900; color: #ffaa33; font-family: monospace; }
    .ocr-unit { font-size: 1.6rem; font-weight: 700; color: #e0e0e0; }
    .final-answer { background: #ffd60a; border-radius: 16px; padding: 20px; border-left: 5px solid #4ecdc4; margin-top: 20px; }
    .stTabs [data-baseweb="tab-list"] button p { font-size: 1.1rem; font-weight: 700; color: #cfd8dc; }
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] p { color: #4ecdc4; border-bottom: 3px solid #4ecdc4; }
    .streamlit-expanderHeader { font-size: 1.1rem; font-weight: 700; color: #4ecdc4 !important; background-color: #11141c; border-radius: 10px; }
    .stCodeBlock { background-color: #0d0f14; border-left: 3px solid #4ecdc4; font-weight: 500; }
    .css-1d391kg, .css-1633tjr { background-color: #0a0c10; }
    hr { border-color: #2d3e50; }
    .reasoning-card {
        background: #0f1419;
        border-radius: 16px;
        padding: 12px 16px;
        margin-bottom: 12px;
        border-left: 4px solid;
        box-shadow: 0 2px 6px rgba(0,0,0,0.2);
    }
    .reasoning-thought { border-left-color: #4ecdc4; }
    .reasoning-action { border-left-color: #ffaa33; }
    .reasoning-observation { border-left-color: #a8e6cf; }
    .reasoning-header { font-weight: 700; margin-bottom: 6px; display: flex; align-items: center; gap: 8px; }
    .reasoning-content { color: #e0e0e0; font-size: 0.95rem; line-height: 1.4; }
    .heuristic-card { background: #FF653F; border-radius: 12px; padding: 12px; margin: 8px 0; border-left: 3px solid #ffd60a; }
</style>
"""

def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)

def render_trajectory(text: str, title: str):
    """Render a reasoning trajectory with Thought/Action/Observation blocks."""
    st.markdown(f"**{title}**")
    if not re.search(r'Thought:|Action:|Observation:', text, re.I):
        st.code(text)
        return
    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        if line.lower().startswith('thought:'):
            st.markdown(f'<div class="reasoning-card reasoning-thought"><div class="reasoning-header">💭 Thought</div><div class="reasoning-content">{line[8:]}</div></div>', unsafe_allow_html=True)
        elif line.lower().startswith('action:'):
            st.markdown(f'<div class="reasoning-card reasoning-action"><div class="reasoning-header">⚙️ Action</div><div class="reasoning-content">{line[7:]}</div></div>', unsafe_allow_html=True)
        elif line.lower().startswith('observation:'):
            st.markdown(f'<div class="reasoning-card reasoning-observation"><div class="reasoning-header">👁️ Observation</div><div class="reasoning-content">{line[12:]}</div></div>', unsafe_allow_html=True)

def render_heuristic(heuristic_text: str):
    """Extract Trigger and Action from heuristic text and display as a card."""
    trigger_match = re.search(r'Trigger:\s*(.*?)(?=\nAction:|\n\n|$)', heuristic_text, re.DOTALL)
    action_match = re.search(r'Action:\s*(.*?)$', heuristic_text, re.DOTALL)
    trigger = trigger_match.group(1).strip() if trigger_match else "Not specified"
    action = action_match.group(1).strip() if action_match else "Not specified"
    st.markdown(f"""
    <div class="heuristic-card">
        <b>🎯 Trigger:</b> {trigger}<br>
        <b>⚡ Action:</b> {action}
    </div>
    """, unsafe_allow_html=True)

def create_sidebar():
    """Build the Streamlit sidebar with settings and memory controls."""
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/microscope.png", width=80)
        st.markdown("## ⚙️ Settings")
        custom_query = st.text_area(
            "📝 Agent Task",
            value="Extract the scale bar value correctly, verify units, and detect anomalies.",
            help="Edit the instruction sent to the LLM."
        )
        st.markdown("---")
        st.markdown("### 🧠 Heuristic Memory")
        if st.button("🗑️ Clear Memory"):
            # This will be handled in app.py; we return a flag
            st.session_state.clear_memory_flag = True
        st.markdown("---")
        st.markdown("**API Status:**")
        from config import API_KEY
        if API_KEY:
            st.success("✅ API key loaded")
        else:
            st.error("❌ Missing SJTU_API_KEY")
        return custom_query