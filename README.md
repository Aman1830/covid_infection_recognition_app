# 🫁 COVID-19 Infection Recognition Application

This project uses deep learning (CNNs) to classify **COVID-positive** and **Non-COVID** chest CT scans.  
It is built using TensorFlow/Keras and deployed through a Streamlit web interface for real-time image upload and diagnosis.

> 🩺 Aims to support medical professionals by offering a fast, lightweight, and reliable tool for automated CT scan evaluation.

---

## 🔬 Project Overview

COVID-19 diagnosis using CT scan imaging is crucial for timely and accurate treatment planning.  
This project implements a **Convolutional Neural Network (CNN)** optimized with **contrastive learning-based Self-Regularization (SR)**, enhancing classification performance while keeping the model compact.

### ✅ Key Highlights
- 🚀 CNN built from scratch using **TensorFlow and Keras**
- 📈 Achieves up to **99.4% accuracy**
- 🔄 Real-time image augmentation with `ImageDataGenerator`
- 🌐 Streamlit interface for instant CT image prediction
  
---

## 📁 Dataset

We use the publicly available **SARS-CoV-2 CT Scan Dataset**:

- 🦠 1,252 COVID-positive CT scans  
- 🧑‍⚕️ 1,230 Non-COVID CT scans  
- 📦 Total: 2,482 images  
- 🏥 Source: Hospitals in São Paulo, Brazil

📌 **Dataset Link**:  
[Kaggle – SARS-CoV-2 CT Scan Dataset](https://www.kaggle.com/datasets/plameneduardo/sarscov2-ctscan-dataset)

> ⚠️ **Note:** Due to size and licensing, the dataset is not included in this repository. Please download it manually from the link above and place it in a `data/` directory as follows:

