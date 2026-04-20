# 🚀 Nexus MLOps: Production-Ready Iris Pipeline

This repository demonstrates a complete, end-to-end MLOps lifecycle for a Machine Learning model. It transitions a standard Iris classification task into a scalable, monitored, and automated production system.

## 🌟 Key Features

- **Automated Training (CI/CD)**: Integrated with GitHub Actions to retrain the model and run tests on every push.
- **Experiment Tracking**: Uses **MLflow** to log metrics, parameters, and model versions.
- **Production API**: A high-performance **FastAPI** backend for real-time inference.
- **Premium Dashboard**: A **Streamlit**-based monitoring nexus with glassmorphism aesthetics and real-time telemetry.
- **Robust Testing**: Comprehensive test suite using **Pytest**.

## 🛠️ Tech Stack

- **Core**: Python 3.10+, Scikit-learn
- **API**: FastAPI, Uvicorn
- **UI/Monitoring**: Streamlit, Plotly
- **MLOps**: MLflow, GitHub Actions
- **Data**: Pandas, NumPy

## 📂 Project Structure

```bash
├── .github/workflows/   # CI/CD Pipeline (Automated Training)
├── api/                 # FastAPI Serving Layer
├── dashboard/           # Streamlit Monitoring Layer
├── src/                 # Logic Layer (Data Loading & Training)
├── models/              # Serialized Model Artifacts
├── tests/               # Quality Assurance (Unit Tests)
└── mlruns/              # MLflow Experiment History
```

## 🚀 Getting Started

### 1. Clone & Install
```bash
git clone https://github.com/bisma1206/MLops.git
cd MLops
pip install -r requirements.txt
```

### 2. Train the Model
```bash
python3 src/train.py
```

### 3. Run the Services
- **Backend API**: `uvicorn api.main:app --reload`
- **Interactive Dashboard**: `streamlit run dashboard/app.py`

## 📊 Monitoring
The dashboard provides real-time insights into model fidelity, system latency, and feature contribution analysis, ensuring your model remains accurate in a production environment.

---
Built with ❤️ for Scalable AI.
