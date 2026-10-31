# 🚥 AISIMULATOR: Twitch Hackathon Hazard Suite

**AISIMULATOR** is an interactive, real-time traffic safety auditing, entity detection, and hazard simulation dashboard built for the **Codex x TwitchCon Hackathon**. Powered by **YOLO26**, OpenCV, **GPT Luna (Warden Billu)**, and Streamlit, the application offers multi-format media auditing, custom object training, automated PDF reporting, and live Twitch chat integration.

---

## ✨ Features

- **📷 Multi-Format Media Auditing**: Process images (`.jpg`, `.png`), video feeds (`.mp4`, `.avi`, `.mov`), and document policies (`.pdf`, `.docx`, `.txt`) up to 400MB.
- **🚨 Real-Time Hazard Proximity Alerts**: Calculates spatial Euclidean distances between detected entities to trigger live visual flashing banners and Web Speech API audio announcements with a 10-second auto-close timer.
- **👮‍♂️ Warden Billu (GPT Luna Assistant)**: An interactive, friendly traffic police assistant in the sidebar providing driving advice, fines breakdown, and road safety guidance.
- **🧠 Custom Entity & Face Training Studio**: Register custom entity classes or faces with uploaded sample images, test detection on live webcam feeds, and generate/export custom PyTorch (`.pt`) model weights alongside `data.yaml` files.
- **📚 100-Question Road Safety Exam**: Interactive assessment covering traffic regulations, signs, and emergency protocols, generating a downloadable **PDF Certificate of Competency** or Assessment Notice.
- **🎮 Twitch Interactive Chat Hub**: Simulates and receives Twitch chat commands (`!hazard`, `!status`, `!billu`, `!speed`, `!brake`) to integrate directly into live stream overlays.
- **📋 Cumulative Audit Logging & PDF Export**: Automatically tracks frame telemetry, entity counts (cars, trucks, buses, ambulances, persons with masks/disabilities), and exports comprehensive PDF/CSV reports.

---

## 🛠️ Prerequisites & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/aisimulator.git
cd aisimulator
```

### 2. Set Up a Virtual Environment
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

### 3. Install Requirements
Create a `requirements.txt` file (or install directly) with the following dependencies:

```bash
pip install streamlit opencv-python-headless numpy pandas pillow openai ultralytics fpdf pypdf python-docx torch
```

---

## 🔑 Configuration

To enable **Warden Billu (GPT Luna)** assistant, set up your OpenAI API key using Streamlit's secrets management:

1. Create a `.streamlit` folder in the project root.
2. Create a `secrets.toml` file inside `.streamlit/`:

```toml
OPENAI_API_KEY = "your-openai-api-key-here"
```

---

## 🚀 Running the Application

Launch the Streamlit app:

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501` to view the dashboard.

---

## 📁 Project Structure

```text
├── app.py                   # Main Streamlit application entry point
├── README.md                # Project documentation
├── .streamlit/
│   └── secrets.toml         # Secret API keys configuration
└── temp_uploaded_traffic.mp4# Temporary video cache for file processing
```

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.