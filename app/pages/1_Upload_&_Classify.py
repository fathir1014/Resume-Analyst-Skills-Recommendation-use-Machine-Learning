# --- make repo root importable on Streamlit Cloud ---
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]  # repo root (works for app/ & app/pages/)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
# ----------------------------------------------------

import pandas as pd
import streamlit as st
from src.infer import predict_topk
from src.recommender import recommend_skills
from src.parser import parse_text_bytes, parse_pdf_bytes

st.header("Upload & Classify")

col1, col2 = st.columns([2, 1])
with col1:
    up = st.file_uploader("Upload PDF / TXT", type=["pdf", "txt"])
    sample = st.text_area("Atau tempel teks resume:", height=220)
with col2:
    force_ocr = st.toggle("Paksa OCR (PDF scan)", value=False, help="Aktifkan jika PDF tidak terbaca")

text, err = "", ""
if up:
    raw = up.read()
    if up.type == "text/plain":
        text = parse_text_bytes(raw)
    else:
        text = parse_pdf_bytes(raw, use_ocr=force_ocr)
        if not text:
            err = "Parser PDF tidak menemukan teks. Aktifkan 'Paksa OCR' atau upload versi TXT/copy-paste."

if err:
    st.warning(err)

text = text or sample
analyze = st.button("Analyze", disabled=not text)

if analyze:
    out = predict_topk(text, k=3)
    if out["error"]:
        st.error(out["error"])
    else:
        ambi_msg = "⚠️ Hasil ambigu — tambahkan detail teknis/proyek." if out["ambiguous"] else "✅ Cukup meyakinkan."
        st.subheader(f"Predicted: {out['pred']}  (ambiguous: {out['ambiguous']})")
        st.caption(ambi_msg)

        df = pd.DataFrame(out["topk"])  # kolom: label, conf (0-1), mungkin ada pct
        if "pct" not in df.columns:  # fallback kalau tidak ada
            df["pct"] = (df["conf"] * 100).round(2)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Top-3 Probabilities (%)**")
            st.bar_chart(df.set_index("label")["pct"])
        with c2:
            st.markdown("**Raw Top-3**")
            st.dataframe(df[["label", "pct", "conf"]])

        # === Skill Recommendations ===
        st.markdown("### 🔧 Skill Recommendations")
        rec = recommend_skills(text, out["pred"], out["topk"], n=6)
        need = rec.get("need_to_add", [])
        alts = rec.get("alts_preview", [])

        if not need and not alts:
            st.info("Belum ada rekomendasi skill untuk kategori ini.")
        else:
            if need:
                st.markdown("**Gap skills (tambahkan ini di CV):**")
                st.write(", ".join(need))
            if alts:
                st.markdown("**Alternatif (terkait label kandidat lain):**")
                st.write(", ".join(alts))

        # === Save Result (opsional) ===
        if st.button("Save result to CSV"):
            import time, os
            os.makedirs("output", exist_ok=True)
            row = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "pred": out["pred"],
                "ambiguous": out["ambiguous"],
                "top1_label": df.iloc[0]["label"],
                "top1_pct": df.iloc[0]["pct"],
                "chars": len(text),
            }
            path = "output/pred_log.csv"
            if not os.path.exists(path):
                pd.DataFrame([row]).to_csv(path, index=False)
            else:
                pd.concat([pd.read_csv(path), pd.DataFrame([row])], ignore_index=True).to_csv(path, index=False)
            st.success(f"Hasil tersimpan ke {path}")
