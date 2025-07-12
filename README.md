# 🧠 Parkinson's Disease Detection Using AI

A machine learning-powered web app that detects **Parkinson’s Disease** using biomedical voice measurements.

Built using **Python**, **Scikit-learn**, and **Streamlit**. Trained via a Jupyter Notebook and deployed on Streamlit Cloud.

🔗 **Live App**: [Click here to try it](https://parkinsons-disease-detection-using-ai-zxsyxxvunheqf7th8eqhbx.streamlit.app)

---

## 📌 Features

- 📁 Upload your own CSV file containing voice data
- ✅ Predicts whether each entry indicates Parkinson’s or not
- 📊 Shows prediction confidence for both healthy and affected classes
- 📈 Displays visual charts: bar, pie, and line
- ⚙️ Uses Random Forest Classifier and StandardScaler

---

## 📊 Dataset Information

- **Source**: [UCI Parkinson's Dataset](https://archive.ics.uci.edu/ml/datasets/parkinsons)
- **Samples**: 195 voice recordings (individuals with and without Parkinson’s)
- **Label**: `status` (1 = Parkinson’s, 0 = Healthy)

### 🔬 Example Features:
```
MDVP:Fo(Hz), MDVP:Fhi(Hz), MDVP:Flo(Hz), MDVP:Jitter(%), MDVP:Jitter(Abs),
MDVP:RAP, MDVP:PPQ, Jitter:DDP, MDVP:Shimmer, MDVP:Shimmer(dB),
Shimmer:APQ3, Shimmer:APQ5, MDVP:APQ, Shimmer:DDA, NHR, HNR,
RPDE, DFA, spread1, spread2, D2, PPE
```

---

## 🗂️ Project Structure

```
📦 Parkinsons-Disease-Detection-Using-AI
├── app.py                  ← Streamlit web app
├── parkinsons_model.ipynb  ← Model training Jupyter Notebook
├── model.pkl               ← Trained model
├── scaler.pkl              ← Saved StandardScaler
├── parkinsons.csv          ← Training dataset
├── test.csv                ← Sample input data for prediction
└── requirements.txt        ← Required Python libraries
```

---

## ⚙️ Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- Joblib
- Matplotlib
- Streamlit

---

## 💻 How to Run Locally

```bash
# Clone the repository
git clone https://github.com/loknadh09/Parkinsons-Disease-Detection-Using-AI.git
cd Parkinsons-Disease-Detection-Using-AI

# Install dependencies
pip install -r requirements.txt

# Launch the app
streamlit run app.py
```

---

## 🌐 Try the Web App

✅ The app is hosted and accessible via Streamlit Cloud.  
Just upload a properly formatted `.csv` file and the model will provide predictions instantly.

🔗 **Live App**:  
[https://parkinsons-disease-detection-using-ai-zxsyxxvunheqf7th8eqhbx.streamlit.app](https://parkinsons-disease-detection-using-ai-zxsyxxvunheqf7th8eqhbx.streamlit.app)

---

## 🙋‍♂️ Author

**Loknadh**  
📎 GitHub: [https://github.com/loknadh09](https://github.com/loknadh09)

---

## 📄 License

This project is for educational and demonstration purposes only.  
Dataset sourced from the UCI Machine Learning Repository.
