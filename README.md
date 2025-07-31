# 🫁 COVID-19 Infection Recognition Application

This project uses deep learning (CNNs) to classify **COVID-positive** and **Non-COVID** chest CT scans.  
The model is built using TensorFlow/Keras and deployed through a Streamlit web interface for real-time image upload and classification.
This project aims to assist medical professionals in rapid, reliable COVID-19 diagnosis from CT scans.

---

## 🔬 Project Overview

COVID-19 diagnosis from CT scan imaging is a critical task in assisting radiologists during the pandemic. This project implements a **lightweight, high-performing Convolutional Neural Network (CNN)** enhanced with **contrastive learning-based self-regularization (SR)** to improve classification accuracy while keeping the model size minimal.

### ✅ Key Features
- Achieves **99.4% accuracy** on COVIDx CT-2A (largest public dataset)
- Uses **DenseNet121-SR** (only ~6.6 MB in size)
- Real-world dataset of **2,482 CT scans** from hospitals in São Paulo, Brazil
- Streamlit interface for easy CT image upload and diagnosis

---

## 📁 Dataset

We use a publicly available SARS-CoV-2 CT scan dataset:
- 🦠 1,252 COVID-positive CT scans
- 🧑‍⚕️ 1,230 Non-COVID CT scans  
- 📦 Total: 2,482 scans

> Data Source: Hospitals in São Paulo, Brazil  
> 📌 For dataset details, visit: [SARS-CoV-2 CT Scan Dataset on Kaggle](https://www.kaggle.com/plameneduardo/sarscov2-ctscan-dataset)

---

## 🧠 Model Architecture

- CNN backbone: **DenseNet121**
- Added contrastive learning-based **Self-Regularization (SR)**
- Trained using:
  - `ImageDataGenerator` for real-time augmentation
  - `categorical_crossentropy` loss
  - `Adam` optimizer

---

## 🚀 Streamlit App

You can upload a CT scan image (`.png`, `.jpg`, `.jpeg`) and the app will classify it as **COVID** or **Non-COVID** in real time.

### ▶️ To Run Locally

```bash
git clone https://github.com/yourusername/covid-ct-classifier.git
cd covid-ct-classifier
pip install -r requirements.txt
streamlit run covid_classifier_app.py

