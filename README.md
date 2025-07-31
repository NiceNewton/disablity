

## 🧠 Learning Disability Detector

This is a simple web app built with **Streamlit** that predicts whether a person may have a learning disability based on inputs like reading speed, spelling accuracy, math score, and attention span. It uses a trained machine learning model saved in `.pkl` format.

---

### 🚀 Live Demo

Deploy this app for free on [Streamlit Cloud](https://share.streamlit.io/) or run it locally using the steps below.

---

### 📁 Files Included

* `app.py` – Main Streamlit application.
* `ld_classifier_model.pkl` – Pretrained classification model.
* `label_encoder.pkl` – Label encoder for class names.
* `requirements.txt` – Python dependencies for running the app.
* `README.md` – This file.

---

### 📦 Requirements

Ensure you have Python 3.7+ installed.

Install dependencies using pip:

```bash
pip install -r requirements.txt
```

---

### 💻 Running Locally

Clone the repo or download the files, then run:

```bash
streamlit run app.py
```

This will open the app in your browser.

---

### 📊 How It Works

The model predicts based on 4 inputs:

* **Reading Speed (words per minute)**
* **Spelling Accuracy (%)**
* **Math Score (%)**
* **Attention Span (minutes)**

It outputs whether a learning disability is likely or not.

---

### 🧪 Model Details

The model is trained using **scikit-learn** and saved using **joblib**.

### 📬 Contact

Feel free to open an issue or connect if you have questions!
