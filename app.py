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
    st.markdown("""
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
