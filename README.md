# MAGNORA | OMNI-CORE: Acoustic Pipeline Digital Twin 🌊⚡

A high-performance Digital Twin and SCADA dashboard for high-pressure gas pipelines. This project leverages an advanced **Multi-Task 1D Convolutional Neural Network (CNN)** to analyze acoustic telemetry data in real-time, instantly detecting both the **exact location (meters)** and the **severity** of pipeline leaks.

![Magnora SCADA System](https://img.shields.io/badge/MAGNORA-OMNI__CORE-00ffcc?style=for-the-badge)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Dash](https://img.shields.io/badge/Dash-008DE4?style=for-the-badge&logo=dash&logoColor=white)

## 🚀 Key Features

* **Multi-Task Deep Learning:** A custom PyTorch 1D CNN that simultaneously performs regression (leak distance prediction) and classification (leak severity).
* **Self-Adaptive Architecture:** Dynamically adjusts its fully connected layers based on the acoustic dataset's feature size.
* **3D Digital Twin Interface:** A cyberpunk-themed, real-time interactive 3D pipeline visualization built with Plotly & Dash.
* **Live Acoustic Signatures:** Real-time waveform rendering of frequency signatures with dynamic noise generation based on leak severity.
* **Interactive SCADA Controls:** Includes localized zoom, pan retention (`uirevision`), and a live data stream Play/Pause functionality for engineer inspection.

## 📁 Project Structure

```text
magnora-acoustic-twin/
├── app/
│   └── dashboard.py          # Real-time Dash SCADA Web Server
├── data/
│   └── pipeline_dataset.csv  # Acoustic sensor & telemetry dataset (23 features)
├── models/
│   ├── acoustic_dnn.pth      # Extracted Neural Network weights
│   └── acoustic_scaler.pkl   # Fitted StandardScaler for inference
├── notebooks/
│   └── 01_acoustic_dnn_twin.ipynb # AI Training, Evaluation, and Export
├── requirements.txt
└── README.md
```

## 🛠️ Installation & Setup

Clone the repository:

```
git clone [https://github.com/YourUsername/magnora-acoustic-twin.git](https://github.com/YourUsername/magnora-acoustic-twin.git)
cd magnora-acoustic-twin
```
Create a virtual environment (optional but recommended):

```
python3 -m venv venv
source venv/bin/activate
Install the dependencies:
pip install -r requirements.txt
```

## 🧠 Usage
### Phase 1: Train the Core AI
- Navigate to the notebooks/ directory and run 01_acoustic_dnn_twin.ipynb using Jupyter. This will train the CNN on your dataset and export the optimized weights to the models/ directory.

Phase 2: Launch the SCADA Dashboard
Run the dashboard script from the root directory to start the local server:
```
python3 app/dashboard.py
```
Open your web browser and navigate to http://127.0.0.1:8050 to interact with the live Digital Twin.

## 🤝 Developed By
Arya Azimi | Co-Founders at Magnora.tech

Bridging Complex AI Algorithms with High-Performance Web Interfaces.