"""
jobs.csv + full-taxonomy-knet.csv  ->  kariyer_set.jsonl  (bench_decisions.py --file ile okunur)

Her satır bench formatında tek bir (ilan, soru) çifti. Etiketler taksonomiden; one-hot probs.
Kullanım:
    python make_kariyer_set.py --jobs jobs.csv --tax full-taxonomy-knet.csv --n 600 --out kariyer_set.jsonl
"""
import argparse
import html
import io
import json
import random
import re

import pandas as pd

# ------------------------------------------------------------------ sorular
KIDEM_LEVELS = ["Giriş seviyesi / işçi / personel", "Stajyer / yeni başlayan", "Yardımcı (uzman yardımcısı, asistan)",
                "Uzman / mühendis / sorumlu", "Kıdemli uzman", "Şef / takım lideri / yönetici",
                "Müdür ve üstü (müdür, direktör, genel müdür)"]


def kidem_bucket(rank):
    if rank <= 9: return 0
    if rank <= 11: return 1
    if rank <= 15: return 2
    if rank <= 16: return 3
    if rank <= 18: return 4
    if rank <= 20: return 5
    return 6


YAKA = ["Mavi Yaka", "Gri Yaka", "Beyaz Yaka"]


def decode_field(s):
    """Dosya latin-1 ile (bayt kaybetmeden) okundu. Metin cp1254, ama içine UTF-8 kodlu noktalama
    (“ ” ’ – … gibi) gömülü. Önce tümü UTF-8 mi dene; değilse cp1254 çöz, sonra kalan 2-4 baytlık
    UTF-8 dizilerini (cp1254'te â€œ, Ã¶ gibi görünür) yerinde onar."""
    if not isinstance(s, str):
        return s
    b = s.encode("latin-1")
    try:
        return b.decode("utf-8")
    except UnicodeDecodeError:
        pass
    out, i, n = [], 0, len(b)
    while i < n:
        c = b[i]
        if c >= 0xC2:  # olası UTF-8 başlangıç baytı
            for ln in (4, 3, 2):
                chunk = b[i:i + ln]
                if len(chunk) == ln and all(0x80 <= x < 0xC0 for x in chunk[1:]):
                    try:
                        out.append(chunk.decode("utf-8")); i += ln; break
                    except UnicodeDecodeError:
                        continue
            else:
                out.append(bytes([c]).decode("cp1254", errors="replace")); i += 1
        else:
            out.append(bytes([c]).decode("cp1254", errors="replace")); i += 1
    return "".join(out)


def strip_html(s):
    s = html.unescape(str(s))
    s = re.sub(r"<br\s*/?>|</li>|</p>|</ul>", "\n", s, flags=re.I)
    s = re.sub(r"<[^>]+>", " ", s)
    s = s.replace("##", ", ")
    return re.sub(r"[ \t]+", " ", re.sub(r"\n\s*\n+", "\n", s)).strip()


def clean_cat(s):
    return re.sub(r"\s*\(P\)\s*$", "", str(s)).strip()


def build(jobs, tax, n, seed):
    m = jobs.merge(tax, left_on="pozisyonId", right_on="position_id", how="inner")
    m = m.dropna(subset=["level_rank", "category_l1_name", "classification_name", "segment_name", "level_set_name"])
    m["aciklama"] = m["IlanAciklama"].map(strip_html).replace("noContent", "")
    m["nitelik"] = m["Nitelikler"].fillna("").map(strip_html).replace("noContent", "")
    m["text"] = ("Pozisyon: " + m["pozisyonAdi"].astype(str) + "\n\n" + m["aciklama"] + "\n\n" + m["nitelik"]).str.strip()
    m = m[m["text"].str.len() >= 120].copy()
    m["kidem"] = m["level_rank"].map(kidem_bucket)
    m["cat"] = m["category_l1_name"].map(clean_cat)
    m["cls"] = m["classification_name"].astype(str).str.strip()
    m["seg"] = m["segment_name"].astype(str).str.strip()
    m["yaka"] = m["level_set_name"].map(YAKA.index)

    # tabakalı örneklem: cat x kidem
    rng = random.Random(seed)
    m["strata"] = m["cat"] + "|" + m["kidem"].astype(str)
    groups = {k: g.index.tolist() for k, g in m.groupby("strata")}
    for v in groups.values(): rng.shuffle(v)
    picked, i = [], 0
    while len(picked) < n and any(groups.values()):
        for k in sorted(groups):
            if groups[k] and len(picked) < n:
                picked.append(groups[k].pop())
    sample = m.loc[picked]

    cats = sorted(m["cat"].unique()); clss = sorted(m["cls"].unique()); segs = sorted(m["seg"].unique())
    questions = {
        "kidem": {"type": "score", "instructions": "Bu ilanın hedeflediği kıdem seviyesi nedir?", "criteria": KIDEM_LEVELS},
        "yaka": {"type": "score", "instructions": "Bu pozisyon hangi çalışan grubuna girer?",
                 "criteria": ["Mavi yaka (saha/üretim işçisi, şoför, usta)", "Gri yaka (teknisyen, tekniker, operatör, formen)",
                              "Beyaz yaka (ofis, uzman, yönetici)"]},
        "alan": {"type": "choice", "instructions": "Bu ilan hangi meslek alanına ait?", "criteria": {c: None for c in cats}},
        "meslek_sinifi": {"type": "choice", "instructions": "Bu ilan hangi meslek sınıfına ait?", "criteria": {c: None for c in clss}},
        "segment": {"type": "choice", "instructions": "Bu pozisyon hangi işgücü segmentine girer?", "criteria": {c: None for c in segs}},
        "yonetici": {"type": "noul", "instructions": "Bu pozisyon yönetsel bir pozisyon mu (ekip/birim yönetimi içeriyor mu)?",
                     "criteria": {"true": "Şef, takım lideri, yönetici, müdür ve üstü", "false": "Bireysel katkı veren pozisyon"}},
        "ingilizce": {"type": "noul", "instructions": "İlan İngilizce bilmeyi şart koşuyor mu?",
                      "criteria": {"true": "İngilizce açıkça istenmiş", "false": "İngilizce istenmiyor ya da belirtilmemiş"}},
    }

    def label(row, q):
        if q == "kidem": return KIDEM_LEVELS, int(row["kidem"])
        if q == "yaka": return questions["yaka"]["criteria"], int(row["yaka"])
        if q == "alan": return cats, cats.index(row["cat"])
        if q == "meslek_sinifi": return clss, clss.index(row["cls"])
        if q == "segment": return segs, segs.index(row["seg"])
        if q == "yonetici": return ["true", "false"], 0 if row["level_rank"] >= 19 else 1
        if q == "ingilizce": return ["true", "false"], 0 if float(row["Lang"]) == 2 else 1

    rows = []
    for _, r in sample.iterrows():
        for q, qd in questions.items():
            opts, gold = label(r, q)
            opts = list(opts)
            rows.append({
                "id": f"{int(r['IlanId'])}:{q}", "document_id": str(int(r["IlanId"])),
                "state": r["text"], "state_format": "text",
                "question_type": qd["type"], "instructions": qd["instructions"], "criteria": json.dumps(qd["criteria"], ensure_ascii=False),
                "options": opts, "probs": [1.0 if i == gold else 0.0 for i in range(len(opts))], "answer": opts[gold],
                "skill": q, "language": "tr", "lang_mode": "tr/tr", "difficulty": "n/a", "has_injection": False,
                "state_tokens": len(r["text"]) // 3, "disagreement": 0.0, "register": "real",
                "meta": {"level_name": r["level_name"], "level_rank": float(r["level_rank"]), "pozisyonAdi": r["pozisyonAdi"]},
            })
    return rows, questions, sample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", default="jobs.csv"); ap.add_argument("--tax", default="full-taxonomy-knet.csv")
    ap.add_argument("--n", type=int, default=600); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="kariyer_set.jsonl")
    a = ap.parse_args()
    jobs = pd.read_csv(a.jobs, encoding="latin-1", low_memory=False)
    for c in ("pozisyonAdi", "IlanAciklama", "Nitelikler"):
        jobs[c] = jobs[c].map(decode_field)
    tax = pd.read_csv(a.tax, low_memory=False)
    rows, questions, sample = build(jobs, tax, a.n, a.seed)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(a.out.replace(".jsonl", "_questions.json"), "w", encoding="utf-8") as f:
        json.dump(questions, f, ensure_ascii=False, indent=1)
    print(f"{len(sample)} ilan x {len(questions)} soru = {len(rows)} satır -> {a.out}")
    print("kıdem dağılımı:", sample["kidem"].value_counts().sort_index().to_dict())
    print("alan dağılımı:", sample["cat"].value_counts().to_dict())
    print("yönetici oranı:", round((sample["level_rank"] >= 19).mean(), 3), " İngilizce oranı:", round((sample["Lang"] == 2).mean(), 3))


if __name__ == "__main__":
    main()
