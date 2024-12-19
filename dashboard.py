# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import plotly.express as px
import json

# Load datasets
raw_data_path = "https://raw.githubusercontent.com/Iamvideo123/fds/refs/heads/main/alien_mission_data.csv"
preprocessed_data_path = "https://raw.githubusercontent.com/Iamvideo123/fds/refs/heads/main/alien_mission_data_preprocessed.csv"

# Read raw and preprocessed data
try:
    df_raw = pd.read_csv(raw_data_path)
    df_preprocessed = pd.read_csv(preprocessed_data_path)
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Title and Sidebar
st.title("Interactive Dashboard for Alien Mission Data")
st.sidebar.title("Options")

# Section 1: Data Distribution Visualization
st.header("1. Data Distribution Visualization")

data_selection = st.sidebar.radio("Choose data to display:", ("Raw Data", "Preprocessed Data"))
selected_data = df_raw if data_selection == "Raw Data" else df_preprocessed

st.write(f"### {data_selection} Distribution")
for col in selected_data.select_dtypes(include=["object", "int64", "float64"]).columns:
    fig = px.histogram(selected_data, x=col, title=f"Distribution of {col}")
    st.plotly_chart(fig, use_container_width=True)

# Section 2: Model Performance Metrics
st.header("2. Model Performance Metrics")

# Classification report for raw data
raw_report_json = """
{
    "0": {"precision": 1.0, "recall": 1.0, "f1-score": 1.0, "support": 419},
    "1": {"precision": 1.0, "recall": 1.0, "f1-score": 1.0, "support": 381},
    "accuracy": 0.9975,
    "macro avg": {"precision": 1.0, "recall": 1.0, "f1-score": 1.0, "support": 800},
    "weighted avg": {"precision": 1.0, "recall": 1.0, "f1-score": 1.0, "support": 800}
}
"""

# Classification report for preprocessed data
preprocessed_report_json = """
{
    "0": {"precision": 0.8555, "recall": 0.8902, "f1-score": 0.8725, "support": 419},
    "1": {"precision": 0.8736, "recall": 0.8346, "f1-score": 0.8537, "support": 381},
    "accuracy": 0.86375,
    "macro avg": {"precision": 0.8646, "recall": 0.8624, "f1-score": 0.8631, "support": 800},
    "weighted avg": {"precision": 0.8641, "recall": 0.8638, "f1-score": 0.8636, "support": 800}
}
"""

# Load JSON reports from strings
try:
    raw_report = json.loads(raw_report_json)
    preprocessed_report = json.loads(preprocessed_report_json)
except Exception as e:
    st.error(f"Error parsing classification reports: {e}")
    st.stop()

# Display metrics for raw data
st.write("### Training Metrics (Raw Data)")
raw_metrics = ["precision", "recall", "f1-score"]
raw_metric_data = {metric: [raw_report[str(i)][metric] for i in range(2)] for metric in raw_metrics}
raw_metric_data["Class"] = ["Class 0", "Class 1"]

raw_df_metrics = pd.DataFrame(raw_metric_data)
fig = px.bar(raw_df_metrics, x="Class", y=raw_metrics, barmode="group", title="Classification Metrics for Raw Data")
st.plotly_chart(fig, use_container_width=True)

# Display metrics for preprocessed data
st.write("### Training Metrics (Preprocessed Data)")
preprocessed_metrics = ["precision", "recall", "f1-score"]
preprocessed_metric_data = {metric: [preprocessed_report[str(i)][metric] for i in range(2)] for metric in preprocessed_metrics}
preprocessed_metric_data["Class"] = ["Class 0", "Class 1"]

preprocessed_df_metrics = pd.DataFrame(preprocessed_metric_data)
fig = px.bar(preprocessed_df_metrics, x="Class", y=preprocessed_metrics, barmode="group", title="Classification Metrics for Preprocessed Data")
st.plotly_chart(fig, use_container_width=True)

# Section 3: Model Comparison
st.header("3. Model Comparison")

# Model comparison table
comparison_data = {
    "Model": ["Raw Data", "Preprocessed Data"],
    "Accuracy": [0.9975, 0.86375],
    "Precision": [1.0, 0.8641],
    "Recall": [1.0, 0.8638],
    "F1-Score": [1.0, 0.8636]
}
df_comparison = pd.DataFrame(comparison_data)

# Visualization of model comparison
fig = px.bar(df_comparison, x="Model", y=["Accuracy", "Precision", "Recall", "F1-Score"],
             barmode="group", title="Performance Comparison Between Raw and Preprocessed Data")
st.plotly_chart(fig, use_container_width=True)

# Display comparison table
st.write("### Comparison Table")
st.dataframe(df_comparison)

# Section 4: Insights
st.header("4. Insights")
st.markdown("""
- **Raw Data**: The model performs exceptionally well with an accuracy of 99.75%, but this could indicate overfitting.
- **Preprocessed Data**: After preprocessing, the accuracy drops to 86.38%, with more realistic precision, recall, and F1-score values.
- **Significance**: Preprocessing reduces overfitting and results in a more generalizable model.
- Use the bar charts and tables above to analyze and compare performance metrics interactively.
""")
