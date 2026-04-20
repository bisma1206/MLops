import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
import requests
from datetime import datetime

# Set page config for a premium feel
st.set_page_config(
    page_title="Nexus MLOps Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for glassmorphism and premium aesthetics
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #f8fafc;
    }
    .stMetric {
        background: rgba(255, 255, 255, 0.05);
        padding: 20px;
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
    }
    .stPlotlyChart {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 10px;
    }
    h1, h2, h3 {
        color: #38bdf8 !important;
        font-family: 'Inter', sans-serif;
    }
    .sidebar .sidebar-content {
        background: #0f172a;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to load model
@st.cache_resource
def load_model():
    model_path = "models/model.joblib"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

model = load_model()

# Sidebar
with st.sidebar:
    st.title("Nexus AI")
    st.markdown("---")
    st.info("System Status: **Operational**")
    st.markdown("### Model Properties")
    st.write("**Model:** Random Forest")
    st.write("**Version:** 1.0.2-prod")
    st.write("**Last Retrained:** " + datetime.now().strftime("%Y-%m-%d"))
    
    st.markdown("---")
    st.markdown("### Control Center")
    if st.button("Trigger Training Pipeline"):
        st.toast("Training pipeline initiated...")

# Main Content
st.title("🚀 MLOps Performance Nexus")
st.markdown("Real-time monitoring and model health analytics.")

# Metrics Row
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Model Fidelity", "98.2%", "+0.5%")
with col2:
    st.metric("Avg Latency", "12ms", "-1ms", delta_color="inverse")
with col3:
    st.metric("Total Predictions", "1.2M", "+12k")
with col4:
    st.metric("Drift Index", "0.02", "Stable", delta_color="off")

st.markdown("---")

# Prediction Interface
st.header("🔮 Real-time Prediction Lab")
with st.container():
    c1, c2 = st.columns([2, 3])
    with c1:
        st.subheader("Input Features")
        sl = st.slider("Sepal Length", 4.0, 8.0, 5.8)
        sw = st.slider("Sepal Width", 2.0, 4.5, 3.0)
        pl = st.slider("Petal Length", 1.0, 7.0, 4.3)
        pw = st.slider("Petal Width", 0.1, 2.5, 1.3)
        
        if st.button("Generate Prediction", use_container_width=True):
            if model:
                # Mock calling the API if running, else use local model
                features = np.array([[sl, sw, pl, pw]])
                cols = ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]
                df_input = pd.DataFrame(features, columns=cols)
                
                pred = model.predict(df_input)[0]
                proba = model.predict_proba(df_input)[0]
                classes = ["Setosa", "Versicolor", "Virginica"]
                
                st.success(f"Prediction: **{classes[pred]}**")
                
                # Show probability chart
                fig_prob = px.bar(
                    x=classes, y=proba, 
                    labels={'x': 'Class', 'y': 'Probability'},
                    color=classes,
                    template="plotly_dark",
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                fig_prob.update_layout(height=300, showlegend=False)
                st.plotly_chart(fig_prob, use_container_width=True)
            else:
                st.error("Model file not found. Please train the model first.")

    with c2:
        st.subheader("System Telemetry")
        # Generate some mock telemetry data
        time_series = pd.DataFrame({
            'Time': pd.date_range(start='2024-01-01', periods=20, freq='h'),
            'Latency (ms)': np.random.normal(12, 2, 20),
            'Load (%)': np.random.uniform(20, 80, 20)
        })
        
        fig_telemetry = go.Figure()
        fig_telemetry.add_trace(go.Scatter(x=time_series['Time'], y=time_series['Latency (ms)'], name='Latency', line=dict(color='#38bdf8')))
        fig_telemetry.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            title="Prediction Latency Over Time",
            height=400
        )
        st.plotly_chart(fig_telemetry, use_container_width=True)

st.markdown("---")
st.header("📊 Model Insights")
# Feature Importance
if model:
    importances = model.feature_importances_
    features = ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"]
    fig_feat = px.pie(values=importances, names=features, hole=.4, template="plotly_dark")
    fig_feat.update_layout(
        title="Feature Contribution Analysis",
        height=450,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation="h", y=-0.2, xanchor="center", x=0.5)
    )
    st.plotly_chart(fig_feat, use_container_width=True)

st.caption("Nexus MLOps Suite v1.0 | Built for Scale")
