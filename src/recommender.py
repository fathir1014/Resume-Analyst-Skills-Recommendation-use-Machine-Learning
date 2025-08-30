from __future__ import annotations
import json, re
from pathlib import Path
from typing import Dict, List

_DEF_SKILLS = "config/skill_map.json"

def _load_skill_map() -> Dict[str, List[str]]:
    p = Path.cwd()
    # cari file dari CWD ke atas
    for cand in [p / _DEF_SKILLS, *[pp / _DEF_SKILLS for pp in p.parents]]:
        if cand.exists():
            return json.loads(cand.read_text(encoding="utf-8"))
    return {}

_SKILL_MAP = _load_skill_map()

def recommend_skills(text: str, label: str, topk: List[Dict], n: int = 6) -> Dict:
    """Rekomendasi skill berbasis mapping sederhana + deteksi gap di teks."""
    wanted = set(_SKILL_MAP.get(label, []))
    found = {s for s in wanted if re.search(rf"\b{re.escape(s)}\b", text, re.I)}
    gap = list(wanted - found)[:n]

    # Ambil sedikit skill dari kandidat label lain untuk preview
    alts: List[str] = []
    for t in topk[1:]:
        alts.extend(_SKILL_MAP.get(t["label"], [])[:2])

    # dedupe dengan urutan terjaga
    seen = set()
    alts_unique = [x for x in alts if not (x in seen or seen.add(x))]

    return {"need_to_add": gap, "alts_preview": alts_unique}
