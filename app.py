# app.py

import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# --- Page layout ---
st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    layout="wide"
)

# --- Sidebar controls ---
st.sidebar.title("Settings")
uploaded_file = st.sidebar.file_uploader("Upload PM_test.txt", type="txt")
threshold     = st.sidebar.slider("RUL threshold for 'Failure Imminent'", 1, 100, 20)
test_size     = st.sidebar.slider("Test set proportion", 0.1, 0.5, 0.2, step=0.05)
random_state  = st.sidebar.number_input("Random seed", min_value=0, value=42, step=1)

# --- Helpers ---
@st.cache_data
def load_data(file) -> pd.DataFrame:
    cols = ['engine_id','cycle','setting1','setting2','setting3'] + [f"s{i}" for i in range(1,22)]
    return pd.read_csv(file, delim_whitespace=True, header=None, names=cols)

def compute_rul_and_label(df: pd.DataFrame, thresh: int) -> pd.DataFrame:
    max_cycle = df.groupby('engine_id')['cycle'].transform('max')
    df['RUL']     = max_cycle - df['cycle']
    df['failure'] = (df['RUL'] < thresh).astype(int)
    return df

@st.cache_resource
def train_evaluate(df: pd.DataFrame, test_sz: float, seed: int):
    features = ['setting1','setting2','setting3'] + [f"s{i}" for i in range(1,22)]
    X, y = df[features], df['failure']
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_sz, random_state=seed, shuffle=True
    )
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    model = RandomForestClassifier(n_estimators=100, random_state=seed)
    model.fit(X_tr_s, y_tr)

    y_pred = model.predict(X_te_s)
    report = classification_report(y_te, y_pred,
                                   target_names=["No Failure","Failure"],
                                   output_dict=True)
    cm = confusion_matrix(y_te, y_pred)
    return model, scaler, report, cm

# --- Main ---
st.title("🔧 Predictive Maintenance Dashboard")

if not uploaded_file:
    st.info("👉 Upload your `PM_test.txt` on the left to get started.")
    st.stop()

# 1) Load & label
df = load_data(uploaded_file)
df = compute_rul_and_label(df, threshold)

# 2) Data preview & stats
st.subheader("1. Raw Data Preview & Summary")
col1, col2 = st.columns(2)
with col1:
    st.write("**Sample rows**")
    st.dataframe(df.head(10), use_container_width=True)
with col2:
    st.write("**Summary statistics**")
    st.dataframe(df.describe(), use_container_width=True)

# 3) RUL & failure distribution
st.subheader("2. RUL & Failure Distributions")
col1, col2 = st.columns(2)
with col1:
    fig = plt.figure()
    sns.histplot(df["RUL"], bins=50, kde=True)
    plt.xlabel("RUL"); plt.title("Remaining Useful Life")
    st.pyplot(fig)
with col2:
    counts = df['failure'].value_counts().rename({0:"No Failure",1:"Failure"})
    st.bar_chart(counts)

# 4) Sensor correlation heatmap
st.subheader("3. Sensor Correlation Heatmap")
with st.expander("Show/Hide"):
    corr = df[[f"s{i}" for i in range(1,22)]].corr()
    fig = plt.figure(figsize=(10,8))
    sns.heatmap(corr, cmap="coolwarm", center=0, vmin=-1, vmax=1)
    plt.title("Sensors s1–s21 Correlation")
    st.pyplot(fig)

# 5) Model training & evaluation
st.subheader("4. Train & Evaluate Model")
model, scaler, report, cm = train_evaluate(df, test_size, random_state)

st.write("**Classification Report**")
st.json(report)

fig = plt.figure()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Fail","Fail"],
            yticklabels=["No Fail","Fail"])
plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("Confusion Matrix")
st.pyplot(fig)

# 6) Engine‐level RUL trends
st.subheader("5. Remaining Useful Life by Engine")
engine_ids = sorted(df["engine_id"].unique())
selected = st.selectbox("Select engine_id", engine_ids)
sub = df[df["engine_id"] == selected]
st.line_chart(sub.set_index("cycle")["RUL"])

# 7) Download processed data
st.subheader("6. Download Processed Data")
csv = df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="📥 Download full table as CSV",
    data=csv,
    file_name="processed_PM_test.csv",
    mime="text/csv"
)
