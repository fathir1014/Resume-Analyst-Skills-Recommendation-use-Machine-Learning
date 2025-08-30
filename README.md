# 📄 Resume Analyst • Skills Recommender  

AI-powered tool untuk menganalisis **Resume/CV**, mengklasifikasikan bidang kerja, dan memberikan rekomendasi skill yang relevan.  
Dibangun dengan **Python, Scikit-learn, Streamlit**, dan **NLP (TF-IDF + Logistic Regression baseline)**.  

---

## ✨ Features
- ✅ Upload **Resume** (PDF/TXT) atau copy-paste teks langsung  
- ✅ **Klasifikasi bidang kerja** (Engineering, Finance, Arts, Banking, Designer, dll)  
- ✅ **Skill Recommendations** sesuai bidang kerja  
- ✅ Visualisasi **Top-3 Probabilitas** dalam bentuk chart  
- ✅ Simpan hasil prediksi ke **CSV log** otomatis  
- ✅ Antarmuka interaktif dengan **Streamlit**  

---

## 📷 Demo Screenshots

### 🏠 Page : Home
![Demo1](assets/Demo(1).png)
![Demo2](assets/Demo(2).png)
![Demo3](assets/Demo(3).png)

### 📄 Page : Upload & Classify
![Demo4](assets/Demo(4).png)
![Demo5](assets/Demo(5).png)
![Demo6](assets/Demo(6).png)

---

## ⚡ Installation

Clone repo & install dependencies:
```bash
git clone https://github.com/<username>/<repo-name>.git
cd <repo-name>/Resume Analyst
pip install -r requirements.txt

## How to run
streamlit run app/Home.py

## Project structure

Resume Analyst-Skills recommendation engine/
│
|── Resume Analyst/
   ├── app/                # Streamlit pages
   ├── src/                # Source code (parser, loader, recommender, infer, visualizer)
   ├── config/             # Skill mapping JSON
   ├── models/             # ML models (baseline_pipeline.pkl)
   ├── data/               # Dataset (raw & preprocessed)
   ├── output/             # Hasil prediksi (CSV, log, dsb)
   ├── requirements.txt    # Dependencies
   └── assets/             # Screenshoots demo

📌 Tech Stack

- Python 3.12
- Pandas, Numpy, Scikit-learn
- Streamlit
- pdfminer.six / pypdf / OCR fallback
- Matplotlib / Altair

Project by : Fathir Rizki Fadillah
