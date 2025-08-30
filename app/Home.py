# --- make repo root importable on Streamlit Cloud ---
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]  # repo root (works for app/ & app/pages/)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
# ----------------------------------------------------

import streamlit as st
from datetime import datetime

# ---------- Page Config ----------
st.set_page_config(
    page_title="Resume Analyst • Skills Recommender",
    page_icon="📄",
    layout="wide",
)

# ---------- Small CSS polish ----------
st.markdown("""
<style>
/* Hero card */
.hero {
  padding: 1.25rem 1.5rem;
  border-radius: 18px;
  background: linear-gradient(135deg, #0ea5e9 0%, #6366f1 60%, #22c55e 100%);
  color: white;
  box-shadow: 0 8px 30px rgba(0,0,0,0.15);
}
.hero h1 { margin: 0 0 .25rem 0; font-size: 2.1rem; }
.hero p  { margin: .25rem 0 0 0; opacity: .95; }

.card {
  border: 1px solid rgba(0,0,0,0.06);
  border-radius: 14px;
  padding: 1rem 1.1rem;
  background: white;
}
.small { font-size: 0.92rem; opacity: .9; }
</style>
""", unsafe_allow_html=True)

# ---------- Hero ----------
st.markdown(f"""
<div class="hero">
  <h1>📄 Resume Analyst <span style="opacity:.9">•</span> Skills Recommender</h1>
  <p>Upload CV ➜ klasifikasi pekerjaan ➜ rekomendasi skill yang relevan. Cepat, ringan, dan siap demo.</p>
</div>
""", unsafe_allow_html=True)

st.write("")

# ---------- Top Stats ----------
c1, c2, c3, c4 = st.columns([1.1, 1, 1, 1])
with c1:
    st.metric("Model", "baseline_pipeline.pkl", "v0.1.0")
with c2:
    st.metric("Kelas Terdukung", "10+", help="ENGINEERING, FINANCE, DESIGNER, DATA SCIENCE, dll.")
with c3:
    st.metric("Inference Latency", "~<100 ms", help="Pada teks pendek, lokal CPU")
with c4:
    st.metric("Last Update", datetime.now().strftime("%d %b %Y"))

st.write("")

# ---------- Quick Actions ----------
qa1, qa2 = st.columns([1,1])
with qa1:
    st.markdown("#### 🚀 Mulai Cepat")
    st.page_link("pages/1_Upload_&_Classify.py", label="➡️ Buka Upload & Classify", icon="🧪")
    st.caption("Upload PDF/TXT atau tempel teks. Klik **Analyze** untuk lihat prediksi & skill.")
with qa2:
    st.markdown("#### 🧰 Utilities")
    st.write("• Log prediksi otomatis ke `output/pred_log.csv` (opsional).")
    st.write("• OCR fallback untuk PDF hasil scan.")
    st.write("• Top-3 probabilitas + flag **ambiguous**.")

st.divider()

# ---------- How to Use ----------
st.markdown("### 🔧 Cara Pakai")
st.markdown("""
1. Buka **Upload & Classify** → unggah file **PDF/TXT** atau **paste** teks resume.  
2. Klik **Analyze** → lihat **prediksi kategori**, **Top-3 probability**, & **rekomendasi skill**.  
3. Jika PDF kosong: aktifkan **Paksa OCR** (untuk PDF hasil scan).  
4. (Opsional) **Save result to CSV** untuk portofolio atau laporan ke klien.
""")

# ---------- Tips ----------
colA, colB = st.columns([1,1])
with colA:
    st.markdown("### 🎯 Tips Akurasi")
    st.markdown("""
- Tulis **tech stack** & **tanggung jawab** (bukan cuma jabatan).
- Tambah **capaian kuantitatif** (contoh: naikkan akurasi 12%).
- Sebut **domain** (Finance, Health, E-commerce) bila relevan.
- Hindari resume 1–2 kalimat; minimal ~60 karakter.
""")

with colB:
    st.markdown("### 🗺️ Peta Fitur (MVP)")
    st.markdown("""
- **Classifier**: TF-IDF + model baseline (scikit-learn).  
- **Recommender**: mapping JSON per kategori (editable).  
- **Parser**: PyPDF → pdfminer → **OCR** fallback (Tesseract).  
- **UI**: Streamlit, chart probabilitas, export CSV.
""")

# ---------- Expanders ----------
with st.expander("📝 Contoh teks yang bagus untuk uji coba"):
    st.code("""Experienced software engineer with 3+ years building ML pipelines.
Strong in Python, scikit-learn, SQL, Docker, and REST APIs. 
Led a project improving model accuracy by 12% on e-commerce data.""", language="markdown")

with st.expander("🧪 Sanity Check Cepat"):
    st.markdown("""
- Upload **TXT** dulu untuk memastikan pipeline jalan.  
- Lalu coba **PDF text-based**.  
- Terakhir, tes **PDF scan** dengan **Paksa OCR** aktif.  
""")

st.caption("© 2025 • Resume Analyst — built for speed & clarity")
