import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Parkinson's Detector", layout="centered")

# Custom CSS for background image and UI enhancements
st.markdown(
    f"""
    <style>
    /* Overall app background */
    .stApp {{
        background-image: url("https://raw.githubusercontent.com/loknadh09/Parkinsons-Disease-Detection-Using-AI/main/images/par.webp"); /* <--- PASTE YOUR RAW GITHUB URL FOR 'par.webp' HERE */
        background-size: cover; /* Ensures the image covers the entire background */
        background-position: center; /* Centers the image */
        background-repeat: no-repeat; /* Prevents the image from repeating */
        background-attachment: fixed; /* Keeps the background fixed when scrolling */
    }}

    /* Makes the header transparent */
    .stApp > header {{
        background-color: rgba(0,0,0,0);
    }}

    /* Adds a subtle dark overlay to the entire app for text readability over the background image */
    .stApp {{
        background-color: rgba(0,0,0,0.5); /* Adjust opacity (0.0 to 1.0) as needed */
    }}

    /* Sets text color to white and adds a slight shadow for better visibility on dark background */
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6, .stApp p, .stApp label, .stApp .stMarkdown {{
        color: #FFFFFF;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.7);
    }}

    /* Ensures file uploader label is visible */
    .stFileUploader label span {{
        color: #FFFFFF !important;
    }}

    /* Styles the main content block to have a semi-transparent background, padding, and rounded corners */
    /* Note: Streamlit's internal CSS class names like .css-1d391kg.e16z5j303 can sometimes change */
    .css-1d391kg.e16z5j303 {{
        background-color: rgba(0, 0, 0, 0.6); /* Semi-transparent black */
        padding: 20px;
        border-radius: 10px;
    }}
    .e1fb0mya1.css-1r6dm1x.exnng7e0 { /* Targeting the inner content block if the above doesn't cover everything */
        background-color: rgba(0, 0, 0, 0.6);
        padding: 20px;
        border-radius: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# Load the dataset
# Ensure 'parkinsons.csv' is in the same directory as 'app.py' or provide its path
df = pd.read_csv("parkinsons.csv")
if 'name' in df.columns:
    df = df.drop(columns=['name'])

X = df.drop(columns=['status'])
y = df['status']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Calculate model accuracy on test data
accuracy = model.score(X_test, y_test) * 100

st.title("🧠 Parkinson's Disease Detection")
st.write("Upload a CSV file with matching features to predict Parkinson's disease.")

uploaded_file = st.file_uploader("📁 Upload your CSV file here", type=["csv"])

# --- Chart Plotting Functions ---
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
    # Enhance chart readability on dark background
    ax.yaxis.label.set_color('white')
    ax.xaxis.label.set_color('white')
    ax.title.set_color('white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.spines['left'].set_color('white')
    ax.spines['bottom'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.set_facecolor('none') # Make plot background transparent
    fig.patch.set_alpha(0.0) # Make figure background transparent
    st.pyplot(fig)

def plot_overall_distribution(predictions):
    labels = ["Parkinson's 🟥", "Healthy 💚"]
    values = [sum(predictions), len(predictions) - sum(predictions)]
    fig, ax = plt.subplots()
    # textprops={'color': 'white'} ensures percentages and labels on slices are white
    ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90, colors=['red', 'green'], textprops={'color': 'white'})
    ax.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.
    ax.set_facecolor('none') # Make plot background transparent
    fig.patch.set_alpha(0.0) # Make figure background transparent
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
    # Enhance chart readability on dark background
    ax.yaxis.label.set_color('white')
    ax.xaxis.label.set_color('white')
    ax.title.set_color('white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.spines['left'].set_color('white')
    ax.spines['bottom'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.set_facecolor('none') # Make plot background transparent
    fig.patch.set_alpha(0.0) # Make figure background transparent
    st.pyplot(fig)

# --- File Uploader and Prediction Logic ---
if uploaded_file is not None:
    try:
        input_df = pd.read_csv(uploaded_file)
        if 'name' in input_df.columns:
            input_df = input_df.drop(columns=['name'])
        if 'status' in input_df.columns: # Remove 'status' column if present in input file
            input_df = input_df.drop(columns=['status'])

        # Ensure input_df has the same columns as X (training features)
        # This is a good practice to prevent errors if the uploaded CSV has different columns
        # You might want to add more robust column validation here
        missing_cols = set(X.columns) - set(input_df.columns)
        if missing_cols:
            st.error(f"⚠️ Uploaded file is missing required features: {', '.join(missing_cols)}. Please ensure your CSV matches the expected format.")
        else:
            # Reorder columns to match training data before scaling
            input_df = input_df[X.columns]
            input_scaled = scaler.transform(input_df)
            predictions = model.predict_proba(input_scaled)

            binary_preds = []
            for i, probs in enumerate(predictions):
                parkinsons_prob = probs[1] * 100
                healthy_prob = probs[0] * 100
                if parkinsons_prob > healthy_prob:
                    st.markdown(
                        f"**Sample {i+1}:** <span style='color:red;'>🟥 Parkinson's Detected</span> "
                        f"(🧾 {parkinsons_prob:.2f}% chance of Parkinson's, 💚 {healthy_prob:.2f}% chance of being Healthy)",
                        unsafe_allow_html=True
                    )
                    binary_preds.append(1)
                else:
                    st.markdown(
                        f"**Sample {i+1}:** <span style='color:green;'>🟩 Likely Healthy</span> "
                        f"(💚 {healthy_prob:.2f}% chance of being Healthy, 🧾 {parkinsons_prob:.2f}% chance of Parkinson's)",
                        unsafe_allow_html=True
                    )
                    binary_preds.append(0)

            st.success(f"✅ Model Accuracy on Internal Test Data: {accuracy:.2f}%")

            st.markdown("---")
            st.subheader("📊 Prediction Summary Charts")
            plot_prediction_chart(predictions)
            plot_overall_distribution(binary_preds)
            plot_confidence_trend(predictions)

    except Exception as e:
        st.error("⚠️ Error processing the uploaded file. Please ensure it is a valid CSV with correct features.")
        st.exception(e)
