# 🚥 AISIMULATOR x Twitch Hackathon Edition

**AISIMULATOR** is an advanced AI-powered traffic simulation, entity auditing, and interactive Twitch streaming suite. Powered by **YOLO26**, OpenCV, **GPT Luna (Warden Billu)**, and Twitch Interactive Chat, this platform enables real-time hazard detection, custom model generation, automated road safety examinations, and live stream overlays.

---

## 🚀 Key Features

### 1. 👁️ Visual Analysis Feed & Content Viewer
* **Multi-Format Support:** Upload images (`.jpg`, `.png`), videos (`.mp4`, `.mov`, `.avi`), or documents (`.pdf`, `.docx`, `.txt`) up to 400MB.
* **Proximity-Only Talking Alerts:** Dynamic browser speech synthesis (`SpeechSynthesisUtterance`) that triggers **only when collision proximity limits are breached**.
* **Exact Hazard Counting & Severity Tiers:** Announces the precise number of active hazards in view and classifies threats into **Medium** or **High** severity levels with customized pitch, rate, and visual flash styling.

### 2. 🎮 Twitch Hackathon Chat & OBS Overlay Hub
* **Interactive Chat Bot:** Viewers can type `!hazard`, `!status`, or `!billu safe driving tips` to receive real-time threat reports and Warden Billu commentary.
* **Stream Window TTS Alerts:** Automatically broadcasts incoming chat hazard commands aloud in the main dashboard window.
* **Channel Points & Bits Redemptions:** Simulate redemptions like triggering emergency brakes or spawning AI traffic wardens.
* **Live Hazard Voting Polls:** Real-time community consensus tracking (Safe vs. Hazardous vs. Emergency).
* **OBS Stream Overlay HUD:** Transparent HTML/CSS browser source snippet that dynamically shifts its border and background colors (Purple ➡️ Orange ➡️ Red) based on live threat severity.

### 3. 🧠 Custom Object / Face Training Studio
* Register and train custom entities or faces with reference images.
* Test custom detectors live via webcam snapshots.
* Generate and export custom PyTorch (`.pt`) model packages and `data.yaml` configurations.

### 4. 📚 Road Safety 100-Question Test & PDF Certification
* Complete an automated 100-question road safety and traffic regulations exam.
* Receive an official score, percentage evaluation, and customized Warden Billu feedback.
* Instantly generate and download an official **PDF Certificate of Competency** (if passed ≥ 80%) or **Assessment Notice**.

---

## 🛠️ Installation & Local Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-username/aisimulator-twitch.git
   cd aisimulator-twitch
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Secrets:**
   Create a `.streamlit/secrets.toml` file to add your OpenAI API key for Warden Billu (GPT Luna):
   ```toml
   OPENAI_API_KEY = "your-openai-api-key-here"
   ```

4. **Run the Streamlit App:**
   ```bash
   streamlit run aisimulator_app.py
   ```

---

## 📂 Project Structure
* `aisimulator_app.py` — Main Streamlit application containing YOLO inference, telemetry, Twitch integration, and audio alerts.
* `requirements.txt` — Python dependencies (Ultralytics, OpenCV, Streamlit, FPDF, PyPDF, python-docx, etc.).
* `README.md` — Project documentation and setup guide.

---

## 🏆 Hackathon Credits
Developed for the **Codex x TwitchCon Hackathon Suite**, combining cutting-edge computer vision safety auditing with interactive live-streaming engagement!