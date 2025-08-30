from __future__ import annotations
from pathlib import Path
from typing import Dict, List
import joblib
import numpy as np

# Model relatif terhadap root project
_DEF_MODEL = "models/baseline_pipeline.pkl"

def _find_root(target_rel: str = _DEF_MODEL) -> Path:
    """Cari root project dari CWD ke atas sampai ketemu file target."""
    here = Path.cwd().resolve()
    for p in [here, *here.parents]:
        if (p / target_rel).exists():
            return p
    raise FileNotFoundError(f"Could not find project root containing {target_rel}")

_ROOT = _find_root()
_PIPE = joblib.load(_ROOT / _DEF_MODEL)

def predict_topk(text: str, k: int = 3, ambig_gap: float = 0.05, min_chars: int = 60,
                 min_conf: float = 0.12) -> Dict:
    text = (text or "").strip()
    if len(text) < min_chars:
        return {
            "pred": None, "topk": [], "ambiguous": True,
            "error": f"Input terlalu pendek (<{min_chars} chars). Tambah detail pengalaman/proyek."
        }

    proba = _PIPE.predict_proba([text])[0]
    classes = _PIPE.classes_
    idx = np.argsort(proba)[::-1][:k]
    top = [{"label": str(classes[i]), "conf": float(proba[i])} for i in idx]
    pred = top[0]["label"]

    # ambiguous bila gap tipis ATAU konfiden top1 rendah
    ambig = (k > 1 and (top[0]["conf"] - top[1]["conf"] < ambig_gap)) or (top[0]["conf"] < min_conf)

    # tambahkan field percent utk UI
    for t in top:
        t["pct"] = round(t["conf"] * 100, 2)

    return {"pred": pred, "topk": top, "ambiguous": ambig, "error": None}

