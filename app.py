import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Parkinson's Detector", layout="centered")

df = pd.read_csv("parkinsons.csv")
if 'name' in df.columns:
    df = df.drop(columns=['name'])

X = df.drop(columns=['status'])
y = df['status']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test) * 100

st.title("🧠 Parkinson's Disease Detection")
st.write("Upload a CSV file with matching features to predict Parkinson's disease.")

uploaded_file = st.file_uploader("📁 Upload your CSV file here", type=["csv"])

def plot_prediction_chart(probabilities):
    labels = [f"S{i+1}" for i in range(len(probabilities))]
    park_probs = [p[1]*100 for p in probabilities]
    healthy_probs = [p[0]*100 for p in probabilities]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels, park_probs, label="Parkinson's 🟥", color='red')
    ax.bar(labels, healthy_probs, bottom=park_probs, label="Healthy 💚", color='green')
    ax.set_ylabel('Probability (%)')
    ax.set_title("Prediction Probabilities for Each Sample")
    ax.legend()
    st.pyplot(fig)

def plot_overall_distribution(predictions):
    labels = ["Parkinson's 🟥", "Healthy 💚"]
    values = [sum(predictions), len(predictions) - sum(predictions)]
    fig, ax = plt.subplots()
    ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90, colors=['red', 'green'])
    ax.axis('equal')
    st.pyplot(fig)

def plot_confidence_trend(probabilities):
    park_probs = [p[1]*100 for p in probabilities]
    healthy_probs = [p[0]*100 for p in probabilities]
    x = [f"S{i+1}" for i in range(len(probabilities))]
    fig, ax = plt.subplots()
    ax.plot(x, park_probs, marker='o', label="Parkinson's 🟥", color='red')
    ax.plot(x, healthy_probs, marker='o', label="Healthy 💚", color='green')
    ax.set_ylabel("Probability (%)")
    ax.set_title("Model Confidence per Sample")
    ax.legend()
    st.pyplot(fig)

if uploaded_file is not None:
    try:
        input_df = pd.read_csv(uploaded_file)
        if 'name' in input_df.columns:
            input_df = input_df.drop(columns=['name'])
        if 'status' in input_df.columns:
            input_df = input_df.drop(columns=['status'])

        input_scaled = scaler.transform(input_df)
        predictions = model.predict_proba(input_scaled)

        binary_preds = []
        for i, probs in enumerate(predictions):
            parkinsons_prob = probs[1] * 100
            healthy_prob = probs[0] * 100
            if parkinsons_prob > healthy_prob:
                st.markdown(
                    f"**Sample {i+1}:** 🟥 Parkinson's Detected "
                    f"(🧾 {parkinsons_prob:.2f}% chance of Parkinson's, 💚 {healthy_prob:.2f}% chance of being Healthy)"
                )
                binary_preds.append(1)
            else:
                st.markdown(
                    f"**Sample {i+1}:** 🟩 Likely Healthy "
                    f"(💚 {healthy_prob:.2f}% chance of being Healthy, 🧾 {parkinsons_prob:.2f}% chance of Parkinson's)"
                )
                binary_preds.append(0)

        st.success(f"✅ Model Accuracy on Internal Test Data: {accuracy:.2f}%")

        st.markdown("---")
        st.subheader("📊 Prediction Summary Charts")
        plot_prediction_chart(predictions)
        plot_overall_distribution(binary_preds)
        plot_confidence_trend(predictions)

    except Exception as e:
        st.error("⚠️ Error processing the uploaded file.")
        st.exception(e)
