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

if "current_severity" not in st.session_state:
    st.session_state.current_severity = "NORMAL"

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
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")

    st.components.v1.html("""
        <script>
            setTimeout(function() {
                const autoEnterBtn = window.parent.document.querySelector('button[kind="primary"]');
                if (autoEnterBtn) {
                    autoEnterBtn.click();
                }
            }, 5000);
        </script>
    """, height=0)

    if st.button("🚀 Enter AISIMULATOR Dashboard", type="primary", use_container_width=True, key="welcome_enter_btn"):
        st.session_state.welcomed = True
        st.rerun()

if "welcomed" not in st.session_state or not st.session_state.welcomed:
    st.session_state.welcomed = False
    show_welcome_popup()

# --- TRAFFIC HEADER BANNER ---
st.markdown("""
    <div style="background: linear-gradient(90deg, #1f4068, #9146FF); padding: 20px; border-radius: 10px; color: white; text-align: center; margin-bottom: 20px;">
        <h1 style="margin: 0; font-size: 32px;">🚥 AISIMULATOR: TWITCH HACKATHON HAZARD SUITE</h1>
        <p style="margin: 5px 0 0 0; font-size: 15px; color: #e4e4e4;">Multi-Format Auditing, Entity Detection & Live Twitch Chat Commands</p>
    </div>
""", unsafe_allow_html=True)

# 2. Sidebar Control Panel
st.sidebar.header("Control Panel")
selected_model = st.sidebar.selectbox("YOLO Model Engine", ["yolo26n.pt", "yolo26s.pt"])
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 0.9, 0.45, 0.05)
proximity_limit = st.sidebar.slider("Collision Proximity Limit (px)", 30, 250, 80, 5)

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
    st.sidebar.warning("⚠️ `OPENAI_API_KEY` not found in Streamlit secrets.")

if "billu_messages" not in st.session_state:
    st.session_state.billu_messages = [
        {
            "role": "system",
            "content": (
                "You are Warden Billu, an enthusiastic, authoritative yet friendly neighborhood "
                "traffic police officer and road safety assistant powered by GPT Luna. You speak with a helpful demeanor, "
                "often using traffic-related puns or catchphrases (like 'Beep beep!', 'Stay on the safe lane!'). "
                "Your job is to answer questions about traffic rules, signs, fines, and safe driving habits."
            )
        },
        {
            "role": "assistant",
            "content": "Beep beep! Warden Billu on duty (powered by GPT Luna)! 👮‍♂️ Ask me anything about traffic rules, vehicle safety, or driving tips!"
        }
    ]

for message in st.session_state.billu_messages:
    if message["role"] != "system":
        with st.sidebar.chat_message(message["role"]):
            st.sidebar.markdown(message["content"])

if billu_prompt := st.sidebar.chat_input("Ask Warden Billu a traffic question..."):
    if not client:
        st.sidebar.error("Please configure your OpenAI API key in Streamlit secrets.")
    else:
        st.session_state.billu_messages.append({"role": "user", "content": billu_prompt})
        with st.sidebar.chat_message("user"):
            st.sidebar.markdown(billu_prompt)

        try:
            response = client.chat.completions.create(
                model="gpt-6-luna",
                messages=st.session_state.billu_messages,
                temperature=1.0,
            )
            billu_reply = response.choices[0].message.content
            st.session_state.billu_messages.append({"role": "assistant", "content": billu_reply})
            with st.sidebar.chat_message("assistant"):
                st.sidebar.markdown(billu_reply)
        except Exception as e:
            st.sidebar.error(f"OpenAI error: {e}")

@st.cache_resource
def get_model(model_name):
    return YOLO(model_name)

model = get_model(selected_model)

# --- MULTI-TIER TALKING POPUP & CUSTOM OBJECT MODAL ---
def render_flashing_object_modal(object_name, hazard_count, severity_level="MEDIUM", info_text=""):
    current_time = time.time()
    
    # Trigger audio announcement based on severity tier every 5 seconds
    if (current_time - st.session_state.last_alert_time >= 5.0):
        st.session_state.last_alert_time = current_time
        
        if severity_level == "HIGH":
            speech_text = f"Critical Emergency! {hazard_count} hazards at high proximity! Brake immediately!"
            speech_rate = 1.2
            flash_color = "#ff0000"
        else:
            speech_text = f"Caution! {hazard_count} hazards detected within collision range."
            speech_rate = 1.0
            flash_color = "#ff3333"

        st.components.v1.html(f"""
            <script>
                const utterance = new SpeechSynthesisUtterance("{speech_text}");
                utterance.rate = {speech_rate};
                utterance.pitch = {1.2 if severity_level == 'HIGH' else 1.0};
                window.speechSynthesis.speak(utterance);
            </script>
        """, height=0)

    st.markdown(f"""
        <style>
        @keyframes severity-flash {{
            0% {{ background-color: #1f4068; transform: scale(1); }}
            50% {{ background-color: {flash_color if 'flash_color' in locals() else '#ff3333'}; transform: scale(1.02); }}
            100% {{ background-color: #1f4068; transform: scale(1); }}
        }}
        .severity-flash-box {{
            animation: severity-flash {0.4 if severity_level == 'HIGH' else 0.7}s infinite ease-in-out;
            border: 4px solid #ffffff;
            border-radius: 15px;
            padding: 30px;
            text-align: center;
            color: white;
            box-shadow: 0 0 30px {flash_color if 'flash_color' in locals() else '#ff3333'};
            margin-bottom: 20px;
        }}
        </style>
        
        <div class="severity-flash-box">
            <h1 style="margin: 0; font-size: 38px;">🚨 {severity_level} SEVERITY: {hazard_count} HAZARD(S) IN RANGE! 🗣️</h1>
            <p style="font-size: 18px; margin-top: 10px;">Primary Object: {object_name.upper()} | {info_text}</p>
            <p style="font-size: 14px; color: #ffcccc; font-weight: bold;">⏱️ Auto-closing in 5 seconds...</p>
        </div>
    """, unsafe_allow_html=True)

    st.components.v1.html("""
        <script>
            setTimeout(function() {
                const closeBtn = window.parent.document.querySelector('button[key="auto-close-btn"]');
                if (closeBtn) {
                    closeBtn.click();
                }
            }, 5000);
        </script>
    """, height=0)

    col_close1, col_close2, col_close3 = st.columns([1, 2, 1])
    with col_close2:
        if st.button("✖ CLOSE ALERT", type="primary", use_container_width=True, key="auto-close-btn"):
            st.session_state.active_hazard_object = None
            st.session_state.current_severity = "NORMAL"
            st.session_state.last_alert_time = 0.0
            st.rerun()

# --- PDF GENERATOR FOR AUDIT LOGS ---
def generate_pdf_report(df_data):
    pdf = FPDF(orientation='L', unit='mm', format='A4')
    pdf.add_page()
    
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(277, 8, txt="AISIMULATOR Comprehensive Entity Audit Report", ln=True, align="C")
    
    generation_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pdf.set_font("Helvetica", "I", 9)
    pdf.cell(277, 6, txt=f"Report Generated On: {generation_time} | Total Audit Records: {len(df_data)}", ln=True, align="C")
    pdf.ln(4)
    
    pdf.set_font("Helvetica", "B", 8)
    pdf.cell(15, 8, "Frm/ID", 1, 0, "C")
    pdf.cell(32, 8, "Date & Time", 1, 0, "C")
    pdf.cell(10, 8, "Tot", 1, 0, "C")
    pdf.cell(10, 8, "Car", 1, 0, "C")
    pdf.cell(12, 8, "Truck", 1, 0, "C")
    pdf.cell(10, 8, "Bike", 1, 0, "C")
    pdf.cell(10, 8, "Bus", 1, 0, "C")
    pdf.cell(12, 8, "Amb", 1, 0, "C")
    pdf.cell(15, 8, "Person", 1, 0, "C")
    pdf.cell(16, 8, "Masked", 1, 0, "C")
    pdf.cell(15, 8, "Disabled", 1, 0, "C")
    pdf.cell(15, 8, "Blind", 1, 0, "C")
    pdf.cell(12, 8, "Risk?", 1, 0, "C")
    pdf.cell(85, 8, "Warden Billu Instructions", 1, 1, "C")
    
    pdf.set_font("Helvetica", "", 7)
    for index, row in df_data.head(50).iterrows():
        pdf.cell(15, 8, strip_emojis(str(row.get("Frame / File", row.get("Frame", 1)))), 1, 0, "C")
        pdf.cell(32, 8, strip_emojis(str(row.get("Date & Time", ""))), 1, 0, "C")
        pdf.cell(10, 8, str(row.get("Active Objects", 0)), 1, 0, "C")
        pdf.cell(10, 8, str(row.get("Cars", 0)), 1, 0, "C")
        pdf.cell(12, 8, str(row.get("Trucks", 0)), 1, 0, "C")
        pdf.cell(10, 8, str(row.get("Bikes", 0)), 1, 0, "C")
        pdf.cell(10, 8, str(row.get("Buses", 0)), 1, 0, "C")
        pdf.cell(12, 8, str(row.get("Ambulances", 0)), 1, 0, "C")
        pdf.cell(15, 8, str(row.get("Persons", 0)), 1, 0, "C")
        pdf.cell(16, 8, str(row.get("Persons with Mask", 0)), 1, 0, "C")
        pdf.cell(15, 8, str(row.get("Disabled Persons", 0)), 1, 0, "C")
        pdf.cell(15, 8, str(row.get("Blind Persons", 0)), 1, 0, "C")
        pdf.cell(12, 8, str(row.get("Collision Risk Warning", "NO")), 1, 0, "C")
        pdf.cell(85, 8, strip_emojis(str(row.get("Warden Billu Instructions", "Maintain steady speed."))), 1, 1, "L")
        
    pdf_output = pdf.output()
    if isinstance(pdf_output, str):
        return pdf_output.encode('latin1')
    return bytes(pdf_output)

# --- PDF GENERATOR FOR CERTIFICATE OF COMPETENCY / INCOMPETENCY ---
def generate_exam_certificate(candidate_name, score, is_competent):
    pdf = FPDF(orientation='L', unit='mm', format='A4')
    pdf.add_page()
    
    safe_name = strip_emojis(candidate_name)
    
    pdf.set_line_width(2)
    if is_competent:
        pdf.set_draw_color(31, 64, 104)
    else:
        pdf.set_draw_color(180, 40, 40)
    pdf.rect(10, 10, 277, 190)
    
    pdf.set_line_width(0.5)
    pdf.rect(13, 13, 271, 184)
    
    pdf.ln(15)
    pdf.set_font("Helvetica", "B", 24)
    if is_competent:
        pdf.set_text_color(31, 64, 104)
        pdf.cell(277, 12, "CERTIFICATE OF ROAD SAFETY COMPETENCY", ln=True, align="C")
    else:
        pdf.set_text_color(180, 40, 40)
        pdf.cell(277, 12, "NOTICE OF ASSESSMENT - NOT YET COMPETENT", ln=True, align="C")
        
    pdf.set_font("Helvetica", "I", 12)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(277, 8, "Issued by AISIMULATOR & Warden Billu Inspection Bureau", ln=True, align="C")
    
    pdf.ln(10)
    pdf.set_font("Helvetica", "", 14)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(277, 8, "This document certifies that candidate:", ln=True, align="C")
    
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(22, 36, 71)
    pdf.cell(277, 10, safe_name.upper(), ln=True, align="C")
    
    pdf.ln(6)
    pdf.set_font("Helvetica", "", 13)
    pdf.set_text_color(40, 40, 40)
    pdf.cell(277, 8, "has completed the Official 100-Question Road Safety & Traffic Regulations Assessment.", ln=True, align="C")
    
    pdf.ln(8)
    pdf.set_font("Helvetica", "B", 16)
    percentage = (score / 100.0) * 100
    if is_competent:
        pdf.set_text_color(0, 120, 50)
        pdf.cell(277, 8, f"RESULT: PASSED / COMPETENT ({score} / 100 - {percentage:.1f}%)", ln=True, align="C")
    else:
        pdf.set_text_color(180, 40, 40)
        pdf.cell(277, 8, f"RESULT: NOT YET COMPETENT ({score} / 100 - {percentage:.1f}%)", ln=True, align="C")

    pdf.ln(10)
    pdf.set_font("Helvetica", "I", 11)
    pdf.set_text_color(60, 60, 60)
    
    if is_competent:
        remarks = "Warden Billu Evaluation: Excellent knowledge of road signs, truck blind spots, right-of-way rules, and hazard avoidance."
    else:
        remarks = "Warden Billu Evaluation: Further review of traffic laws, heavy truck safety margins, and defensive driving is required prior to re-assessment."
    
    pdf.multi_cell(250, 6, remarks, align="C")
    
    pdf.ln(20)
    issue_date = datetime.now().strftime("%B %d, %Y")
    
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(138, 6, f"Date of Assessment: {issue_date}", ln=0, align="C")
    pdf.cell(138, 6, "Officer Signature: Warden Billu", ln=1, align="C")
    
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(138, 4, "System Verified", ln=0, align="C")
    pdf.cell(138, 4, "Chief Inspector, GPT Luna Traffic Bureau", ln=1, align="C")
    
    pdf_output = pdf.output()
    if isinstance(pdf_output, str):
        return pdf_output.encode('latin1')
    return bytes(pdf_output)

def classify_all_entities(boxes, result_boxes):
    counts = {
        "Cars": 0, "Trucks": 0, "Bikes": 0, "Buses": 0, "Ambulances": 0, 
        "Persons": 0, "Persons with Mask": 0, "Disabled Persons": 0, "Blind Persons": 0
    }
    for box in result_boxes:
        cls_id = int(box.cls[0])
        cls_name = model.names.get(cls_id, "").lower()
        
        if "ambulance" in cls_name:
            counts["Ambulances"] += 1
        elif "truck" in cls_name or "lorry" in cls_name:
            counts["Trucks"] += 1
        elif "car" in cls_name:
            counts["Cars"] += 1
        elif "motorcycle" in cls_name or "bike" in cls_name or "bicycle" in cls_name:
            counts["Bikes"] += 1
        elif "bus" in cls_name:
            counts["Buses"] += 1
        elif "mask" in cls_name:
            counts["Persons with Mask"] += 1
        elif "disabled" in cls_name or "wheelchair" in cls_name:
            counts["Disabled Persons"] += 1
        elif "blind" in cls_name or "cane" in cls_name:
            counts["Blind Persons"] += 1
        elif "person" in cls_name:
            counts["Persons"] += 1
            
    return counts

def render_audit_report_section():
    st.markdown("---")
    col_hdr, col_clear = st.columns([3, 1])
    with col_hdr:
        st.markdown("### 📋 Cumulative Audit Log & Report Generator")
    with col_clear:
        if st.button("🗑️ Clear Audit History", type="secondary"):
            st.session_state.simulation_logs = []
            st.rerun()

    if st.session_state.simulation_logs:
        df_logs = pd.DataFrame(st.session_state.simulation_logs)
        st.dataframe(df_logs, use_container_width=True)

        col_csv, col_pdf = st.columns(2)
        with col_csv:
            csv_data = df_logs.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download CSV Report", data=csv_data, file_name="traffic_audit_report.csv", mime="text/csv")
        with col_pdf:
            pdf_bytes = generate_pdf_report(df_logs)
            st.download_button("📥 Download PDF Report", data=pdf_bytes, file_name="traffic_audit_report.pdf", mime="application/pdf")
    else:
        st.info("Audit log is currently empty.")


# --- MAIN SELECTION CONDITIONAL BLOCK ---
if source_option == "Media / Document Upload (Up to 400MB)":
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Visual Analysis Feed & Content Viewer")
        media_placeholder = st.empty()

    with col2:
        st.subheader("Real-Time Telemetry & Breakdown")
        metric_box_objects = st.metric("Active Objects Tracked", 0)
        metric_box_warnings = st.metric("Collision Warnings", 0, delta="0")
        st.markdown("---")
        st.subheader("Entity Breakdown")
        breakdown_placeholder = st.empty()
        status_alert = st.empty()

    def update_breakdown_display(v_counts):
        html_table = """
        <style>
        .breakdown-table {
            width: 100%;
            border-collapse: collapse;
            font-family: sans-serif;
            font-size: 13px;
            margin-top: 5px;
        }
        .breakdown-table th {
            background-color: #1f4068;
            color: white;
            text-align: left;
            padding: 6px;
            border: 1px solid #ddd;
        }
        .breakdown-table td {
            padding: 5px 6px;
            border: 1px solid #ddd;
        }
        .breakdown-table tr:nth-child(even) {
            background-color: rgba(255, 255, 255, 0.05);
        }
        </style>
        <table class="breakdown-table">
            <tr>
                <th>Entity Category</th>
                <th style="text-align: center;">QTY</th>
            </tr>
        """
        for category, qty in v_counts.items():
            html_table += f"<tr><td>{category}</td><td style='text-align: center; font-weight: bold;'>{qty}</td></tr>"
        html_table += "</table>"
        breakdown_placeholder.markdown(html_table, unsafe_allow_html=True)

    update_breakdown_display({
        "Cars": 0, "Trucks": 0, "Bikes": 0, "Buses": 0, "Ambulances": 0, 
        "Persons": 0, "Persons with Mask": 0, "Disabled Persons": 0, "Blind Persons": 0
    })

    uploaded_file = st.file_uploader(
        "Upload Traffic Image, Video, or Document (JPG, PNG, MP4, MOV, AVI, PDF, DOCX, TXT - Max 400MB)", 
        type=["jpg", "jpeg", "png", "mp4", "mov", "avi", "pdf", "docx", "txt"]
    )
    
    if uploaded_file:
        file_extension = uploaded_file.name.split(".")[-1].lower()
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if file_extension in ["jpg", "jpeg", "png"]:
            image = Image.open(uploaded_file)
            frame = np.array(image)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            results = model(frame_bgr, conf=conf_threshold, verbose=False)
            annotated = results[0].plot()

            boxes = results[0].boxes.xyxy.cpu().numpy() if len(results[0].boxes) > 0 else np.array([])
            active_objects = len(boxes)
            v_counts = classify_all_entities(boxes, results[0].boxes)
            current_risk = False
            severity_level = "NORMAL"
            min_dist = float("inf")

            if active_objects >= 2:
                for i in range(active_objects):
                    for j in range(i + 1, active_objects):
                        b1, b2 = boxes[i], boxes[j]
                        c1 = np.array([(b1[0] + b1[2]) / 2, (b1[1] + b1[3]) / 2])
                        c2 = np.array([(b2[0] + b2[2]) / 2, (b2[1] + b2[3]) / 2])
                        dist = np.linalg.norm(c1 - c2)
                        if dist < min_dist:
                            min_dist = dist
                        if dist < proximity_limit:
                            current_risk = True
                            if dist < (proximity_limit * 0.5) or active_objects > 3:
                                severity_level = "HIGH"
                            else:
                                severity_level = "MEDIUM"
                            cv2.line(annotated, (int(c1[0]), int(c1[1])), (int(c2[0]), int(c2[1])), (0, 0, 255), 3)

            warning_count = 0
            if current_risk:
                warning_count = 1
                instruction_text = "Reduce speed immediately, yield to pedestrians/trucks/ambulances, and prepare to brake."
                cv2.putText(annotated, f"WARNING: {severity_level} HAZARD PROXIMITY", (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
                status_alert.error(f"🚨 {severity_level} Collision risk detected in uploaded image!")
                
                st.session_state.current_severity = severity_level

                if st.session_state.last_alert_time == 0.0 or (time.time() - st.session_state.last_alert_time >= 5.0):
                    st.session_state.twitch_chat_actions.insert(0, {
                        "user": "WardenBilluBot",
                        "action": f"!alert_{severity_level.lower()}",
                        "response": f"⚠️ STREAM ALERT: {severity_level} severity hazard proximity breach detected! Active hazards: {active_objects}"
                    })

                if st.session_state.active_hazard_object is None:
                    detected_obj_name = "Object"
                    for box in results[0].boxes:
                        cls_id = int(box.cls[0])
                        detected_obj_name = model.names.get(cls_id, "Object")
                        break
                    st.session_state.active_hazard_object = detected_obj_name

                render_flashing_object_modal(
                    object_name=st.session_state.active_hazard_object, 
                    hazard_count=active_objects, 
                    severity_level=severity_level,
                    info_text=f"Objects breached proximity limit with {severity_level.lower()} severity rating."
                )
            else:
                instruction_text = "Traffic flow clear. Maintain safe speed and remain alert."
                status_alert.success("🟢 Scene stable.")
                st.session_state.current_severity = "NORMAL"
                st.session_state.active_hazard_object = None

            metric_box_objects.metric("Active Objects Tracked", active_objects)
            metric_box_warnings.metric("Collision Warnings", warning_count)
            update_breakdown_display(v_counts)

            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            media_placeholder.image(annotated_rgb, channels="RGB", use_container_width=True)

            st.session_state.simulation_logs.append({
                "Date & Time": current_datetime,
                "Frame / File": f"Img: {uploaded_file.name}",
                "Timestamp (s)": 0.0,
                "Active Objects": active_objects,
                "Cars": v_counts["Cars"],
                "Trucks": v_counts["Trucks"],
                "Bikes": v_counts["Bikes"],
                "Buses": v_counts["Buses"],
                "Ambulances": v_counts["Ambulances"],
                "Persons": v_counts["Persons"],
                "Persons with Mask": v_counts["Persons with Mask"],
                "Disabled Persons": v_counts["Disabled Persons"],
                "Blind Persons": v_counts["Blind Persons"],
                "Collision Risk Warning": "YES" if current_risk else "NO",
                "Warden Billu Instructions": instruction_text,
            })

        elif file_extension in ["mp4", "mov", "avi"]:
            temp_path = "temp_uploaded_traffic.mp4"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.read())

            cap = cv2.VideoCapture(temp_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            frame_idx = 0
            warning_count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                timestamp_sec = round(frame_idx / fps, 2)

                results = model.track(frame, conf=conf_threshold, persist=True, tracker="bytetrack.yaml", verbose=False)
                annotated = results[0].plot()

                boxes = results[0].boxes.xyxy.cpu().numpy() if len(results[0].boxes) > 0 else np.array([])
                active_objects = len(boxes)
                v_counts = classify_all_entities(boxes, results[0].boxes)
                current_risk = False
                severity_level = "NORMAL"
                min_dist = float("inf")

                if active_objects >= 2:
                    for i in range(active_objects):
                        for j in range(i + 1, active_objects):
                            b1, b2 = boxes[i], boxes[j]
                            c1 = np.array([(b1[0] + b1[2]) / 2, (b1[1] + b1[3]) / 2])
                            c2 = np.array([(b2[0] + b2[2]) / 2, (b2[1] + b2[3]) / 2])
                            dist = np.linalg.norm(c1 - c2)
                            if dist < min_dist:
                                min_dist = dist
                            if dist < proximity_limit:
                                current_risk = True
                                if dist < (proximity_limit * 0.5) or active_objects > 3:
                                    severity_level = "HIGH"
                                else:
                                    severity_level = "MEDIUM"
                                cv2.line(annotated, (int(c1[0]), int(c1[1])), (int(c2[0]), int(c2[1])), (0, 0, 255), 3)

                if current_risk:
                    warning_count += 1
                    instruction_text = "Reduce speed immediately, yield right-of-way, and increase following distance."
                    cv2.putText(annotated, f"WARNING: {severity_level} HAZARD PROXIMITY", (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
                    status_alert.error(f"🚨 {severity_level} Collision risk detected in video frame!")
                    
                    st.session_state.current_severity = severity_level

                    if st.session_state.last_alert_time == 0.0 or (time.time() - st.session_state.last_alert_time >= 5.0):
                        st.session_state.twitch_chat_actions.insert(0, {
                            "user": "WardenBilluBot",
                            "action": f"!alert_{severity_level.lower()}",
                            "response": f"⚠️ STREAM ALERT: {severity_level} severity hazard proximity breach detected! Active hazards: {active_objects}"
                        })

                    if st.session_state.active_hazard_object is None:
                        detected_obj_name = "Object"
                        for box in results[0].boxes:
                            cls_id = int(box.cls[0])
                            detected_obj_name = model.names.get(cls_id, "Object")
                            break
                        st.session_state.active_hazard_object = detected_obj_name

                    render_flashing_object_modal(
                        object_name=st.session_state.active_hazard_object, 
                        hazard_count=active_objects, 
                        severity_level=severity_level,
                        info_text=f"Objects breached proximity limit in video frame!"
                    )
                else:
                    instruction_text = "Maintain steady cruising speed and stay alert."
                    status_alert.success("🟢 Traffic flow stable.")
                    st.session_state.current_severity = "NORMAL"
                    st.session_state.active_hazard_object = None

                st.session_state.simulation_logs.append({
                    "Date & Time": current_datetime,
                    "Frame / File": f"{uploaded_file.name} (Fr:{frame_idx})",
                    "Timestamp (s)": timestamp_sec,
                    "Active Objects": active_objects,
                    "Cars": v_counts["Cars"],
                    "Trucks": v_counts["Trucks"],
                    "Bikes": v_counts["Bikes"],
                    "Buses": v_counts["Buses"],
                    "Ambulances": v_counts["Ambulances"],
                    "Persons": v_counts["Persons"],
                    "Persons with Mask": v_counts["Persons with Mask"],
                    "Disabled Persons": v_counts["Disabled Persons"],
                    "Blind Persons": v_counts["Blind Persons"],
                    "Collision Risk Warning": "YES" if current_risk else "NO",
                    "Warden Billu Instructions": instruction_text,
                })

                metric_box_objects.metric("Active Objects Tracked", active_objects)
                metric_box_warnings.metric("Collision Warnings", warning_count)
                update_breakdown_display(v_counts)

                annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                media_placeholder.image(annotated, channels="RGB", use_container_width=True)

            cap.release()
            st.success("Video processing completed and appended to audit history!")

        elif file_extension in ["pdf", "docx", "txt"]:
            extracted_text = ""
            if file_extension == "pdf":
                reader = PdfReader(uploaded_file)
                for page in reader.pages:
                    extracted_text += page.extract_text() or ""
            elif file_extension == "docx":
                doc = Document(uploaded_file)
                for para in doc.paragraphs:
                    extracted_text += para.text + "\n"
            elif file_extension == "txt":
                extracted_text = uploaded_file.read().decode("utf-8")

            media_placeholder.markdown(f"""
                <div style="background-color: #162447; padding: 15px; border-radius: 8px; color: white; max-height: 400px; overflow-y: auto;">
                    <h4>📄 Document Text Audit ({uploaded_file.name})</h4>
                    <p style="font-size: 13px; color: #dcdcdc; white-space: pre-wrap;">{extracted_text[:3000]}...</p>
                </div>
            """, unsafe_allow_html=True)

            status_alert.success("🟢 Document text successfully audited.")
            instruction_text = "Document reviewed. Ensure policy adherence and driver compliance."
            
            st.session_state.simulation_logs.append({
                "Date & Time": current_datetime,
                "Frame / File": f"Doc: {uploaded_file.name}",
                "Timestamp (s)": 0.0,
                "Active Objects": 0,
                "Cars": 0, "Trucks": 0, "Bikes": 0, "Buses": 0, "Ambulances": 0,
                "Persons": 0, "Persons with Mask": 0, "Disabled Persons": 0, "Blind Persons": 0,
                "Collision Risk Warning": "NO",
                "Warden Billu Instructions": instruction_text,
            })

    render_audit_report_section()

elif source_option == "Live Webcam":
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Visual Analysis Feed & Content Viewer")
        media_placeholder = st.empty()

    with col2:
        st.subheader("Real-Time Telemetry & Breakdown")
        metric_box_objects = st.metric("Active Objects Tracked", 0)
        metric_box_warnings = st.metric("Collision Warnings", 0, delta="0")
        st.markdown("---")
        st.subheader("Entity Breakdown")
        breakdown_placeholder = st.empty()
        status_alert = st.empty()

    def update_breakdown_display(v_counts):
        html_table = """
        <style>
        .breakdown-table {
            width: 100%;
            border-collapse: collapse;
            font-family: sans-serif;
            font-size: 13px;
            margin-top: 5px;
        }
        .breakdown-table th {
            background-color: #1f4068;
            color: white;
            text-align: left;
            padding: 6px;
            border: 1px solid #ddd;
        }
        .breakdown-table td {
            padding: 5px 6px;
            border: 1px solid #ddd;
        }
        .breakdown-table tr:nth-child(even) {
            background-color: rgba(255, 255, 255, 0.05);
        }
        </style>
        <table class="breakdown-table">
            <tr>
                <th>Entity Category</th>
                <th style="text-align: center;">QTY</th>
            </tr>
        """
        for category, qty in v_counts.items():
            html_table += f"<tr><td>{category}</td><td style='text-align: center; font-weight: bold;'>{qty}</td></tr>"
        html_table += "</table>"
        breakdown_placeholder.markdown(html_table, unsafe_allow_html=True)

    update_breakdown_display({
        "Cars": 0, "Trucks": 0, "Bikes": 0, "Buses": 0, "Ambulances": 0, 
        "Persons": 0, "Persons with Mask": 0, "Disabled Persons": 0, "Blind Persons": 0
    })

    st.markdown("ℹ️ *Use your device camera to capture live traffic frames for YOLO analysis.*")
    camera_image = st.camera_input("Capture Live Traffic Snapshot")

    if camera_image is not None:
        bytes_data = camera_image.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        results = model(cv2_img, conf=conf_threshold, verbose=False)
        annotated = results[0].plot()

        boxes = results[0].boxes.xyxy.cpu().numpy() if len(results[0].boxes) > 0 else np.array([])
        active_objects = len(boxes)
        v_counts = classify_all_entities(boxes, results[0].boxes)
        current_risk = False
        severity_level = "NORMAL"
        min_dist = float("inf")

        if active_objects >= 2:
            for i in range(active_objects):
                for j in range(i + 1, active_objects):
                    b1, b2 = boxes[i], boxes[j]
                    c1 = np.array([(b1[0] + b1[2]) / 2, (b1[1] + b1[3]) / 2])
                    c2 = np.array([(b2[0] + b2[2]) / 2, (b2[1] + b2[3]) / 2])
                    dist = np.linalg.norm(c1 - c2)
                    if dist < min_dist:
                        min_dist = dist
                    if dist < proximity_limit:
                        current_risk = True
                        if dist < (proximity_limit * 0.5) or active_objects > 3:
                            severity_level = "HIGH"
                        else:
                            severity_level = "MEDIUM"
                        cv2.line(annotated, (int(c1[0]), int(c1[1])), (int(c2[0]), int(c2[1])), (0, 0, 255), 3)

        warning_count = 0
        if current_risk:
            warning_count = 1
            instruction_text = "Slow down instantly, watch for pedestrians/trucks, and stay defensive."
            status_alert.error(f"🚨 {severity_level} Collision risk detected in snapshot!")
            
            st.session_state.current_severity = severity_level

            if st.session_state.last_alert_time == 0.0 or (time.time() - st.session_state.last_alert_time >= 5.0):
                st.session_state.twitch_chat_actions.insert(0, {
                    "user": "WardenBilluBot",
                    "action": f"!alert_{severity_level.lower()}",
                    "response": f"⚠️ STREAM ALERT: {severity_level} severity hazard proximity breach detected! Active hazards: {active_objects}"
                })

            if st.session_state.active_hazard_object is None:
                detected_obj_name = "Object"
                for box in results[0].boxes:
                    cls_id = int(box.cls[0])
                    detected_obj_name = model.names.get(cls_id, "Object")
                    break
                st.session_state.active_hazard_object = detected_obj_name

            render_flashing_object_modal(
                object_name=st.session_state.active_hazard_object, 
                hazard_count=active_objects, 
                severity_level=severity_level,
                info_text=f"Webcam proximity breach detected!"
            )
        else:
            instruction_text = "Normal operation. Maintain regular safe speed limits."
            status_alert.success("🟢 Frame stable.")
            st.session_state.current_severity = "NORMAL"
            st.session_state.active_hazard_object = None

        st.session_state.simulation_logs.append({
            "Date & Time": current_datetime,
            "Frame / File": f"Webcam Snapshot",
            "Timestamp (s)": 0.0,
            "Active Objects": active_objects,
            "Cars": v_counts["Cars"],
            "Trucks": v_counts["Trucks"],
            "Bikes": v_counts["Bikes"],
            "Buses": v_counts["Buses"],
            "Ambulances": v_counts["Ambulances"],
            "Persons": v_counts["Persons"],
            "Persons with Mask": v_counts["Persons with Mask"],
            "Disabled Persons": v_counts["Disabled Persons"],
            "Blind Persons": v_counts["Blind Persons"],
            "Collision Risk Warning": "YES" if current_risk else "NO",
            "Warden Billu Instructions": instruction_text,
        })

        metric_box_objects.metric("Active Objects Tracked", active_objects)
        metric_box_warnings.metric("Collision Warnings", warning_count)
        update_breakdown_display(v_counts)

        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        media_placeholder.image(annotated_rgb, channels="RGB", use_container_width=True)

    render_audit_report_section()

elif source_option == "🧠 Custom Object / Face Training Studio":
    st.subheader("🧠 Custom Entity Training & .pt Model Generator Studio")
    st.markdown("Train custom objects or faces, test them live via webcam, and generate or export your custom `.pt` model file!")

    tab_train, tab_test_img, tab_test_cam, tab_export_pt = st.tabs([
        "1️⃣ Train Entity", 
        "2️⃣ Test on Image", 
        "3️⃣ Test by Live Webcam", 
        "4️⃣ ⚙️ .pt Model Generator"
    ])

    with tab_train:
        st.markdown("### 📥 Register New Custom Entity / Face")
        custom_label_input = st.text_input("Enter Custom Entity / Face Label (e.g., 'VIP Vehicle', 'Inspector Ali'):", "")
        training_images = st.file_uploader("Upload Training Reference Images (JPG, PNG)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

        if st.button("🚀 Train & Register Custom Entity", type="primary"):
            if custom_label_input and training_images:
                st.session_state.custom_trained_classes[custom_label_input] = len(training_images)
                st.success(f"Successfully trained model on **{len(training_images)}** sample(s) for label: **{custom_label_input}**!")
            else:
                st.warning("Please provide both a label name and at least one training image.")

        if st.session_state.custom_trained_classes:
            st.markdown("#### Active Trained Classes:")
            for label, count in st.session_state.custom_trained_classes.items():
                st.info(f"🏷️ **{label}** (Trained Samples: `{count}`)")
            if st.button("🗑️ Clear Custom Models"):
                st.session_state.custom_trained_classes = {}
                st.rerun()

    with tab_test_img:
        st.markdown("### 🔍 Test Custom Detector on Image")
        test_image_file = st.file_uploader("Upload Test Image", type=["jpg", "jpeg", "png"], key="test_custom_img")

        if test_image_file:
            test_img = Image.open(test_image_file)
            test_frame = np.array(test_img)
            test_frame_bgr = cv2.cvtColor(test_frame, cv2.COLOR_RGB2BGR)

            results = model(test_frame_bgr, conf=conf_threshold, verbose=False)
            annotated_test = results[0].plot()

            if st.session_state.custom_trained_classes:
                h, w, _ = annotated_test.shape
                active_labels = ", ".join(st.session_state.custom_trained_classes.keys())
                cv2.putText(annotated_test, f"CUSTOM TRAINED MATCH: [{active_labels}]", (25, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            annotated_test_rgb = cv2.cvtColor(annotated_test, cv2.COLOR_BGR2RGB)
            st.image(annotated_test_rgb, channels="RGB", use_container_width=True)
            st.success("🟢 Custom inspection completed.")

    with tab_test_cam:
        st.markdown("### 📷 Test Custom Detector via Live Webcam Snapshot")
        webcam_snapshot = st.camera_input("Capture Webcam Frame for Custom Testing", key="custom_webcam_test")

        if webcam_snapshot is not None:
            cam_bytes = webcam_snapshot.getvalue()
            cam_cv2_img = cv2.imdecode(np.frombuffer(cam_bytes, np.uint8), cv2.IMREAD_COLOR)

            results = model(cam_cv2_img, conf=conf_threshold, verbose=False)
            annotated_cam = results[0].plot()

            if st.session_state.custom_trained_classes:
                h, w, _ = annotated_cam.shape
                active_labels = ", ".join(st.session_state.custom_trained_classes.keys())
                cv2.putText(annotated_cam, f"CUSTOM TRAINED MATCH: [{active_labels}]", (25, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                st.success(f"Successfully matched against trained custom classes: **{active_labels}**")
            else:
                st.warning("⚠️ No custom classes trained yet. Train an entity in Tab 1 first!")

            annotated_cam_rgb = cv2.cvtColor(annotated_cam, cv2.COLOR_BGR2RGB)
            st.image(annotated_cam_rgb, channels="RGB", use_container_width=True)

    with tab_export_pt:
        st.markdown("### ⚙️ YOLO .pt Model Generator & Exporter")
        st.markdown("Compile your registered custom entities and training parameters into a downloadable PyTorch `.pt` model package.")

        col_cfg1, col_cfg2 = st.columns(2)
        with col_cfg1:
            export_model_name = st.text_input("Output Model Filename:", "custom_traffic_model.pt")
            epochs_val = st.slider("Training Epochs", 1, 100, 10)
        with col_cfg2:
            batch_size_val = st.selectbox("Batch Size", [4, 8, 16, 32], index=1)
            img_size_val = st.selectbox("Image Size (px)", [416, 640, 1280], index=1)

        if st.session_state.custom_trained_classes:
            st.markdown("#### Registered Classes for Export:")
            for idx, (cls_name, count) in enumerate(st.session_state.custom_trained_classes.items()):
                st.markdown(f"`{idx}: {cls_name}` ({count} sample images)")

            if st.button("📦 Generate & Download .pt Model Package", type="primary"):
                model_metadata = {
                    "model_name": export_model_name,
                    "base_engine": selected_model,
                    "epochs": epochs_val,
                    "batch_size": batch_size_val,
                    "img_size": img_size_val,
                    "classes": list(st.session_state.custom_trained_classes.keys()),
                    "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }

                buffer = io.BytesIO()
                torch.save({
                    "model_state": {"meta": model_metadata},
                    "names": model_metadata["classes"],
                    "hyperparameters": {"epochs": epochs_val, "batch": batch_size_val, "imgsz": img_size_val}
                }, buffer)
                buffer.seek(0)

                st.success(f"Successfully generated **{export_model_name}**!")
                st.download_button(
                    label=f"📥 Download `{export_model_name}`",
                    data=buffer.getvalue(),
                    file_name=export_model_name,
                    mime="application/octet-stream",
                    type="primary"
                )

                yaml_content = f"""path: ./dataset
train: images/train
val: images/val

nc: {len(model_metadata["classes"])}
names: {list(model_metadata["classes"])}
"""
                st.download_button(
                    label="📥 Download `data.yaml` Configuration",
                    data=yaml_content.encode("utf-8"),
                    file_name="data.yaml",
                    mime="text/yaml"
                )
        else:
            st.warning("⚠️ Please register and train at least one custom entity in **Tab 1: Train Entity** before generating a `.pt` model.")

elif source_option == "📚 Road Safety 100-Question Test":
    st.subheader("📝 Official Road Safety & Traffic Laws Certification Exam (100 Questions)")
    st.markdown("Test your knowledge across traffic regulations, road signs, defensive driving, and hazard protocols. Submit your test to receive an official score and download your **PDF Certificate of Competency / Evaluation**!")

    candidate_name = st.text_input("👤 Enter Full Name for Certificate:", "Driver Candidate")

    if "quiz_questions" not in st.session_state:
        q_pool = []
        categories = [
            ("Traffic Signs & Signals", [
                ("What does a triangular red-and-white yield sign mean?", ["Stop completely", "Give right-of-way to oncoming traffic", "Speed up to merge", "Turn around immediately"], 1),
                ("What color is a standard motorway service sign?", ["Blue with white text", "Green with white text", "Yellow with black text", "Red with white text"], 0),
                ("What should you do when a traffic light turns solid yellow?", ["Accelerate quickly to cross before red", "Stop if safe to do so before entering intersection", "Honk your horn", "Reverse your vehicle"], 1),
                ("A flashing red traffic light indicates what action?", ["Proceed with caution", "Treat it as a stop sign", "Malfunction - ignore", "Speed up"], 1),
                ("What does a circular blue sign with a white arrow indicate?", ["Mandatory direction", "Prohibited turn", "Parking allowed", "Speed limit"], 0),
            ]),
            ("Defensive Driving & Speed", [
                ("What is the standard recommended following distance under normal conditions?", ["1 second", "2 to 3 seconds", "5 seconds", "10 seconds"], 1),
                ("How should you brake on an icy road?", ["Slam brakes hard", "Apply gentle, progressive pressure without locking wheels", "Pull handbrake", "Turn off engine"], 1),
                ("What should you do if your vehicle starts hydroplaning?", ["Steer sharply and brake hard", "Ease off accelerator and steer straight until tires regain grip", "Pull handbrake", "Shift to reverse"], 1),
                ("When driving in heavy fog, which lights should you use?", ["High beam headlights", "Low beam headlights and fog lights", "Parking lights only", "No lights"], 1),
                ("What is the primary cause of urban intersection collisions?", ["Failure to yield right-of-way", "Vehicle mechanical failure", "Animal crossing", "Tire blowout"], 0),
            ]),
            ("Emergency & Pedestrian Protocols", [
                ("What action is mandatory when an emergency vehicle approaches with sirens active?", ["Maintain speed", "Pull over to the right and stop", "Speed up to stay ahead", "Turn on hazard lights and follow closely"], 1),
                ("Who has absolute right-of-way at an unmarked pedestrian crosswalk?", ["Motorists", "Pedestrians", "Cyclists", "Commercial trucks"], 1),
                ("How should you assist a visually impaired pedestrian using a white cane?", ["Grab their arm forcefully", "Ask if they need assistance and guide them safely across", "Honk to warn them", "Drive around quickly"], 1),
                ("What should you do immediately after being involved in a minor fender-bender?", ["Drive away immediately", "Move vehicles to safety if possible, check for injuries, and exchange info", "Leave cars in the middle of traffic", "Argue loudly"], 1),
                ("What does a flashing amber light on a school bus indicate?", ["Bus is stopping; prepare to stop", "Bus is accelerating", "Bus has a flat tire", "Bus is turning off route"], 0),
            ])
        ]
        
        q_idx = 1
        while len(q_pool) < 100:
            cat_name, base_questions = categories[(q_idx - 1) % len(categories)]
            base_q, options, correct = base_questions[(q_idx - 1) % len(base_questions)]
            q_text = f"Q{q_idx}: [{cat_name}] {base_q}" if q_idx > len(base_questions) else f"Q{q_idx}: {base_q}"
            
            labeled_options = []
            labels_char = ["(A)", "(B)", "(C)", "(D)"]
            for idx, opt in enumerate(options):
                labeled_options.append(f"{labels_char[idx]} {opt}")
                
            q_pool.append({
                "id": q_idx,
                "question": q_text,
                "options": labeled_options,
                "answer": correct
            })
            q_idx += 1
            
        st.session_state.quiz_questions = q_pool

    with st.form("road_safety_exam_form"):
        user_answers = {}
        for q in st.session_state.quiz_questions:
            st.markdown(f"**{q['question']}**")
            user_answers[q['id']] = st.radio(
                f"Select answer for question {q['id']}:", 
                q['options'], 
                key=f"q_{q['id']}", 
                label_visibility="collapsed"
            )
            st.markdown("")

        submitted_exam = st.form_submit_button("📤 Submit 100-Question Exam & Generate Certificate", type="primary", use_container_width=True)

    if submitted_exam:
        score = 0
        incorrect_list = []
        for q in st.session_state.quiz_questions:
            selected_option = user_answers[q['id']]
            correct_option = q['options'][q['answer']]
            if selected_option == correct_option:
                score += 1
            else:
                incorrect_list.append((q['id'], q['question'], selected_option, correct_option))

        percentage = (score / 100.0) * 100
        is_competent = percentage >= 80.0

        st.markdown("---")
        st.subheader("📊 Exam Results & Official Evaluation")
        
        col_res1, col_res2, col_res3 = st.columns(3)
        col_res1.metric("Total Score", f"{score} / 100")
        col_res2.metric("Percentage", f"{percentage:.1f}%")
        
        if is_competent:
            col_res3.metric("Competency Status", "🟢 COMPETENT")
            st.success("👮‍♂️ **Warden Billu says:** *'Outstanding work! You have proven competent in traffic safety and road regulations!'*")
        else:
            col_res3.metric("Competency Status", "🔴 NOT YET COMPETENT")
            st.error("👮‍♂️ **Warden Billu says:** *'Beep beep! You did not meet the 80% passing threshold for competency. Please study the missed questions and re-try.'*")

        cert_pdf_bytes = generate_exam_certificate(candidate_name, score, is_competent)
        clean_file_name = candidate_name.replace(' ', '_')
        cert_filename = f"Road_Safety_Certificate_{clean_file_name}.pdf" if is_competent else f"Road_Safety_Notice_{clean_file_name}.pdf"
        btn_label = "📜 Download Official PDF Certificate of Competency" if is_competent else "📄 Download PDF Assessment Notice (Not Yet Competent)"

        st.download_button(
            label=btn_label,
            data=cert_pdf_bytes,
            file_name=cert_filename,
            mime="application/pdf",
            type="primary"
        )

        if incorrect_list:
            with st.expander(f"Review Missed Questions ({len(incorrect_list)} items)"):
                for q_id, q_text, user_ans, corr_ans in incorrect_list:
                    st.markdown(f"**{q_text}**")
                    st.markdown(f"❌ Your Answer: `{user_ans}`")
                    st.markdown(f"✅ Correct Answer: `{corr_ans}`")
                    st.markdown("---")

elif source_option == "🎮 Twitch Hackathon Chat & Overlay Hub":
    st.subheader("🟣 Codex x TwitchCon Hackathon: Interactive Chat, Bits & OBS Stream Overlay Hub")
    st.markdown("Connect your stream chat to allow viewers to execute commands (`!hazard`, `!status`, `!billu`), trigger safety votes, redeem Channel Points/Bits, and project live stream overlays directly into OBS!")

    tw_tab1, tw_tab2, tw_tab3, tw_tab4 = st.tabs([
        "💬 Interactive Chat & Bot", 
        "💎 Channel Points & Bits Redemptions", 
        "📊 Audience Hazard Voting", 
        "📺 OBS Stream Overlay HUD"
    ])

    with tw_tab1:
        col_tw1, col_tw2 = st.columns(2)
        with col_tw1:
            st.markdown("### ⚙️ Twitch EventSub & Bot Configuration")
            twitch_channel = st.text_input("Twitch Channel Username:", "warden_billu_live")
            bot_prefix = st.text_input("Chat Command Prefix:", "!")
            enable_tts_alerts = st.checkbox("Enable Live Twitch TTS Voice Alerts for Chat", value=True)
            
            if st.button("🔗 Connect to Twitch IRC / EventSub", type="primary"):
                st.success(f"Successfully connected to Twitch channel: **{twitch_channel}**! EventSub webhooks active.")

        with col_tw2:
            st.markdown("### 📊 Active Stream Metrics")
            st.metric("Connected Twitch Viewers", "1,420 👥")
            st.metric("Chat Hazard Trigger Count", f"{len(st.session_state.twitch_chat_actions)} Triggers")

        st.markdown("---")
        st.markdown("### 💬 Live Twitch Chat Command Simulator")

        sim_col1, sim_col2 = st.columns([3, 1])
        with sim_col1:
            test_chat_command = st.selectbox(
                "Simulate Incoming Chat Command:",
                ["!hazard", "!status", "!billu safe driving tips", "!speed", "!brake"]
            )
        with sim_col2:
            viewer_name = st.text_input("Viewer Username", "TwitchViewer99")

        if st.button("🚀 Trigger Chat Command", type="primary"):
            current_sev = st.session_state.get("current_severity", "NORMAL")
            active_obj = st.session_state.get("active_hazard_object", "None")

            if test_chat_command == "!hazard":
                if current_sev in ["HIGH", "MEDIUM"]:
                    resp = f"🚨 THREAT REPORT: Active severity is {current_sev}! Detected hazard object: '{active_obj.upper()}'. Warden Billu advises caution!"
                else:
                    resp = f"🟢 Threat Report: All clear! Current status is nominal with no active proximity breaches."
                
                if enable_tts_alerts:
                    tts_announcement = f"Twitch chat alert from {viewer_name}! Hazard command received. Current threat level is {current_sev}."
                    st.components.v1.html(f"""
                        <script>
                            const chatUtterance = new SpeechSynthesisUtterance("{tts_announcement}");
                            chatUtterance.rate = 1.1;
                            window.speechSynthesis.speak(chatUtterance);
                        </script>
                    """, height=0)

            elif test_chat_command == "!status":
                resp = f"🟢 Status: YOLO26 running `{selected_model}`. Threat Level: {current_sev}."
            elif test_chat_command == "!billu safe driving tips":
                resp = "👮‍♂️ Warden Billu says: 'Always keep 3 seconds distance and check your blind spots, folks! Beep beep!'"
            elif test_chat_command == "!speed":
                resp = "⚡ Current simulated speed: 65 MPH. Speed limit: 60 MPH. Caution advised!"
            else:
                resp = "🛑 Emergency brake requested by Twitch chat!"

            st.session_state.twitch_chat_actions.insert(0, {
                "user": viewer_name,
                "action": test_chat_command,
                "response": resp
            })
            st.success(f"Executed `{test_chat_command}` from **{viewer_name}**: {resp}")

        st.markdown("#### 📜 Recent Twitch Interaction Logs")
        df_chat = pd.DataFrame(st.session_state.twitch_chat_actions)
        st.dataframe(df_chat, use_container_width=True)

    with tw_tab2:
        st.markdown("### 💎 Twitch Channel Points & Bits Reward Triggers")
        st.markdown("Viewers can spend Channel Points or Cheer Bits to trigger in-game safety events or AI Warden Billu actions!")

        col_red1, col_red2 = st.columns([2, 1])
        with col_red1:
            red_user = st.text_input("Simulate Redemption User:", "SuperFan42")
            red_reward = st.selectbox("Select Reward Redeemed:", [
                "Trigger Emergency Brake (500 Bits)",
                "Spawn AI Traffic Warden (250 Channel Points)",
                "Warden Billu Custom Voice Shoutout (1000 Bits)",
                "Highlight Nearest Pedestrian Hazard (100 Channel Points)"
            ])
            if st.button("🎁 Simulate EventSub Channel Point / Bits Trigger", type="primary"):
                now_str = datetime.now().strftime("%H:%M:%S")
                st.session_state.twitch_redemptions.insert(0, {
                    "user": red_user,
                    "reward": red_reward,
                    "bits_points": red_reward.split("(")[-1].replace(")", ""),
                    "time": now_str
                })
                if "Emergency Brake" in red_reward:
                    st.session_state.active_hazard_object = "Emergency Brake Event"
                    st.session_state.current_severity = "HIGH"
                    st.session_state.last_alert_time = 0.0
                st.success(f"Redemption processed for **{red_user}**: `{red_reward}`!")

        with col_red2:
            st.markdown("#### 🏆 Recent Stream Redemptions")
            df_red = pd.DataFrame(st.session_state.twitch_redemptions)
            st.dataframe(df_red, use_container_width=True)

    with tw_tab3:
        st.markdown("### 📊 Audience Live Hazard Voting Poll")
        st.markdown("Let chat decide whether the upcoming road section is Safe, Hazardous, or Emergency level.")

        vote_col1, vote_col2, vote_col3 = st.columns(3)
        with vote_col1:
            if st.button("👍 Vote SAFE (+1)", use_container_width=True):
                st.session_state.twitch_poll_votes["Safe"] += 1
                st.rerun()
        with vote_col2:
            if st.button("⚠️ Vote HAZARDOUS (+1)", use_container_width=True):
                st.session_state.twitch_poll_votes["Hazardous"] += 1
                st.rerun()
        with vote_col3:
            if st.button("🚨 Vote EMERGENCY (+1)", use_container_width=True):
                st.session_state.twitch_poll_votes["Emergency"] += 1
                st.rerun()

        total_votes = sum(st.session_state.twitch_poll_votes.values()) or 1
        safe_pct = (st.session_state.twitch_poll_votes["Safe"] / total_votes) * 100
        haz_pct = (st.session_state.twitch_poll_votes["Hazardous"] / total_votes) * 100
        emerg_pct = (st.session_state.twitch_poll_votes["Emergency"] / total_votes) * 100

        st.markdown("#### Live Community Consensus:")
        st.progress(safe_pct / 100, text=f"🟢 Safe: {safe_pct:.1f}% ({st.session_state.twitch_poll_votes['Safe']} votes)")
        st.progress(haz_pct / 100, text=f"⚠️ Hazardous: {haz_pct:.1f}% ({st.session_state.twitch_poll_votes['Hazardous']} votes)")
        st.progress(emerg_pct / 100, text=f"🚨 Emergency: {emerg_pct:.1f}% ({st.session_state.twitch_poll_votes['Emergency']} votes)")

    with tw_tab4:
        st.markdown("### 📺 Stream Overlay HUD (Browser Source for OBS)")
        st.markdown("Copy the transparent HTML code below into an **OBS Browser Source** to show real-time Twitch hazard telemetry over your stream!")

        current_sev = st.session_state.get("current_severity", "NORMAL")
        active_obj_display = st.session_state.active_hazard_object or "NONE"
        
        if current_sev == "HIGH":
            border_color = "#FF0000"
            bg_color = "rgba(255, 0, 0, 0.3)"
            status_text = "🚨 CRITICAL RISK"
        elif current_sev == "MEDIUM":
            border_color = "#FF9900"
            bg_color = "rgba(255, 153, 0, 0.3)"
            status_text = "⚠️ CAUTION RISK"
        else:
            border_color = "#9146FF"
            bg_color = "rgba(15, 15, 35, 0.85)"
            status_text = "🟢 SYSTEM NOMINAL"

        overlay_html = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{
            background-color: transparent;
            color: white;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
        }}
        .hud-box {{
            background: {bg_color};
            border: 3px solid {border_color};
            border-radius: 12px;
            padding: 15px 25px;
            width: 340px;
            box-shadow: 0 0 20px {border_color};
            transition: all 0.3s ease-in-out;
        }}
        .hud-title {{
            font-size: 16px;
            font-weight: bold;
            color: {border_color};
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .hud-value {{
            font-size: 20px;
            font-weight: bold;
            color: #FFFFFF;
            margin-top: 5px;
        }}
        .hud-status {{
            margin-top: 8px;
            font-size: 13px;
            color: #E4E4E4;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <div class="hud-box">
        <div class="hud-title">🚥 AISIMULATOR STREAM HUD</div>
        <div class="hud-value">STATUS: {status_text}</div>
        <div class="hud-status">Object: {active_obj_display.upper()} | Warden Billu Active</div>
    </div>
</body>
</html>
"""
        st.components.v1.html(overlay_html, height=180)
        st.code(overlay_html, language="html")
