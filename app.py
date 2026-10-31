from datetime import datetime
import re
import time
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from openai import OpenAI
from ultralytics import YOLO
from fpdf import FPDF
from pypdf import PdfReader
from docx import Document
import torch
import io
import json

# -------------------------------------------------------------------
# HELPER FUNCTION: STRIP UNICODE EMOJIS FOR FPDF COMPATIBILITY
# -------------------------------------------------------------------
def strip_emojis(text: str) -> str:
    """Removes unicode emoji characters that standard FPDF fonts (Helvetica/Arial) cannot encode."""
    if not text:
        return ""
    emoji_pattern = re.compile(
        "[\U00010000-\U0010ffff"  # Extended symbols/emojis
        "\u2600-\u27BF"          # Miscellaneous symbols
        "\u2300-\u23FF"          # Miscellaneous technical
        "\u2B00-\u2BFF"          # Various symbols & arrows
        "\u200D"                 # Zero-width joiner
        "\uFE0F]+",              # Variation selector
        flags=re.UNICODE
    )
    cleaned_text = emoji_pattern.sub("", text)
    return cleaned_text.encode("latin-1", "ignore").decode("latin-1")

# 1. Page Configuration
st.set_page_config(
    page_title="AISIMULATOR x Twitch Hackathon",
    page_icon="🎮",
    layout="wide",
)

# --- INITIALIZE PERSISTENT LOGS, CUSTOM TRAINING & AUDIO STATE ---
if "simulation_logs" not in st.session_state:
    st.session_state.simulation_logs = []

if "custom_trained_classes" not in st.session_state:
    st.session_state.custom_trained_classes = {}

if "active_hazard_object" not in st.session_state:
    st.session_state.active_hazard_object = None

if "last_alert_time" not in st.session_state:
    st.session_state.last_alert_time = 0.0

if "twitch_chat_actions" not in st.session_state:
    st.session_state.twitch_chat_actions = [
        {"user": "StreamerBot", "action": "!status", "response": "System nominal. Warden Billu on duty!"}
    ]

if "twitch_redemptions" not in st.session_state:
    st.session_state.twitch_redemptions = [
        {"user": "Gamer123", "reward": "Trigger Emergency Brake", "bits_points": "500 Bits / Points", "time": "18:45:00"}
    ]

if "twitch_poll_votes" not in st.session_state:
    st.session_state.twitch_poll_votes = {"Safe": 42, "Hazardous": 18, "Emergency": 5}

# --- UNIVERSAL 5-SECOND AUTO-CLOSE WELCOME POPUP ---
@st.dialog("🚗 Welcome to AISIMULATOR (Twitch Hackathon Edition)", width="medium")
def show_welcome_popup():
    st.markdown(
        """
        <style>
        @keyframes headlight-blink {
            0% { opacity: 0.2; transform: scale(0.9); }
            50% { opacity: 1.0; transform: scale(1.1); filter: drop-shadow(0 0 10px #9146FF); }
            100% { opacity: 0.2; transform: scale(0.9); }
        }
        .blinking-light {
            animation: headlight-blink 0.8s infinite ease-in-out;
            font-size: 32px;
            text-align: center;
        }
        .welcome-banner {
            text-align: center;
            font-family: sans-serif;
        }
        </style>
        
        <div class="welcome-banner">
            <h2>🟣 Codex x TwitchCon Hackathon Suite</h2>
            <p>Powered by <b>YOLO26</b>, OpenCV, <b>GPT Luna</b>, and Twitch Interactive Chat 🎮</p>
            <div class="blinking-light">🟣 🤖 🟣</div>
            <p style="font-size: 13px; color: gray;">Twitch EventSub & Streamer Mode Active!</p>
            <p style="font-size: 12px; color: #9146FF; font-weight: bold;">⏱️ Auto-closing & entering dashboard in exactly 5 seconds...</p>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    st.markdown("---")

    # Strict 5-second JS auto-close trigger for welcome modal
    st.components.v1.html(
        """
        <script>
            setTimeout(function() {
                const autoEnterBtn = window.parent.document.querySelector('button[kind="primary"]');
                if (autoEnterBtn) {
                    autoEnterBtn.click();
                }
            }, 5000);
        </script>
        """, 
        height=0
    )

    if st.button("🚀 Enter AISIMULATOR Dashboard", type="primary", use_container_width=True, key="welcome_enter_btn"):
        st.session_state.welcomed = True
        st.rerun()

if "welcomed" not in st.session_state or not st.session_state.welcomed:
    st.session_state.welcomed = False
    show_welcome_popup()

# --- TRAFFIC HEADER BANNER ---
st.markdown(
    """
    <div style="background: linear-gradient(90deg, #1f4068, #9146FF); padding: 20px; border-radius: 10px; color: white; text-align: center; margin-bottom: 20px;">
        <h1 style="margin: 0; font-size: 32px;">🚥 AISIMULATOR: TWITCH HACKATHON HAZARD SUITE</h1>
        <p style="margin: 5px 0 0 0; font-size: 15px; color: #e4e4e4;">Multi-Format Auditing, Entity Detection & Live Twitch Chat Commands</p>
    </div>
    """, 
    unsafe_allow_html=True
)

# 2. Sidebar Control Panel
st.sidebar.header("Control Panel")
selected_model = st.sidebar.selectbox("YOLO Model Engine", ["yolo26n.pt", "yolo26s.pt"])
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 0.9, 0.45, 0.05)
proximity_limit = st.sidebar.slider("Collision Proximity Limit (px)", 30, 250, 80, 5)

voice_alerts_enabled = st.sidebar.checkbox(
    "🔊 Enable spoken hazard alerts",
    value=False,
    help="Turn this on once to allow your browser to speak hazard counts. Browser audio permissions may apply."
)

source_option = st.sidebar.radio(
    "Select Input Source / Mode", 
    [
        "Media / Document Upload (Up to 400MB)", 
        "Live Webcam", 
        "🧠 Custom Object / Face Training Studio", 
        "📚 Road Safety 100-Question Test",
        "🎮 Twitch Hackathon Chat & Overlay Hub"
    ]
)

# 3. OpenAI & GPT Luna Configuration
st.sidebar.markdown("---")
st.sidebar.subheader("👮‍♂️ Warden Billu (GPT Luna Assistant)")

if "OPENAI_API_KEY" in st.secrets:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
else:
    client = None
