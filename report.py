"""
results/*.jsonl -> report.md + accuracy_by_skill.png (+ calibration.png)

Metrikler
  acc       argmax(pred) == argmax(gold)   (noul: p(evet) >= 0.5)
  tvd       0.5 * sum|pred - gold|          (soft etiketle uzaklık; LLM için anlamsız, boş bırakılır)
  ece       10 kutulu Expected Calibration Error (top-prob vs. doğruluk)
  err%      model/parse hatası oranı (hatalı satır yanlış sayılır)
  s/q       soru başına saniye

Kullanım: python report.py results/
"""
import json
import sys
from collections import defaultdict
from pathlib import Path


def load(res_dir):
    recs = []
    for p in sorted(Path(res_dir).glob("*.jsonl")):
        with open(p, encoding="utf-8") as f:
            recs += [json.loads(l) for l in f if l.strip()]
    return recs


def argmax(v):
    return max(range(len(v)), key=lambda i: v[i])


def is_correct(r):
    if not r["pred"]: return False
    if r["question_type"] == "noul":
        return (r["pred"][0] >= 0.5) == (r["probs"][0] >= 0.5)
    return argmax(r["pred"]) == argmax(r["probs"])


def tvd(r):
    return 0.5 * sum(abs(a - b) for a, b in zip(r["pred"], r["probs"]))


def mae(recs):
    """score sorularında argmax düzey farkı (ordinal hata)."""
    s = [abs(argmax(r["pred"]) - argmax(r["probs"])) for r in recs if r["question_type"] == "score" and r["pred"]]
    return sum(s) / len(s) if s else None


def is_soft(recs):
    """Sistem gerçek olasılık üretiyor mu (one-hot değil)?"""
    return any(r["pred"] and max(r["pred"]) < 0.999 for r in recs)


def ece(recs, bins=10):
    pairs = [(max(r["pred"]), is_correct(r)) for r in recs if r["pred"]]
    if not pairs: return None
    tot, out = len(pairs), 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        inb = [(c, ok) for c, ok in pairs if lo < c <= hi or (b == 0 and c == 0)]
        if inb:
            conf = sum(c for c, _ in inb) / len(inb)
            acc = sum(ok for _, ok in inb) / len(inb)
            out += len(inb) / tot * abs(conf - acc)
    return out


def summarize(recs):
    n = len(recs)
    soft = is_soft(recs)
    row = {
        "n": n,
        "acc": sum(is_correct(r) for r in recs) / n,
        "tvd": (sum(tvd(r) for r in recs if r["pred"]) / max(1, sum(1 for r in recs if r["pred"]))) if soft else None,
        "ece": ece(recs) if soft else None,
        "err": sum(1 for r in recs if r["error"] or not r["pred"]) / n,
        "sec": sum(r["seconds"] for r in recs) / n,
        "mae": mae(recs),
    }
    return row


def fmt(v, pct=False, nd=3):
    if v is None: return "–"
    return f"{100 * v:.1f}" if pct else f"{v:.{nd}f}"


def table(groups, systems, title):
    lines = [f"\n### {title}\n", "| grup | " + " | ".join(systems) + " |",
             "|---|" + "---|" * len(systems)]
    for g in sorted(groups):
        cells = []
        for s in systems:
            rs = groups[g].get(s)
            cells.append(f"{fmt(summarize(rs)['acc'], pct=True)} (n={len(rs)})" if rs else "–")
        lines.append(f"| {g} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(res_dir="results"):
    recs = load(res_dir)
    systems = sorted({r["system"] for r in recs})
    by_sys = {s: [r for r in recs if r["system"] == s] for s in systems}

    # Ortak kesişim: her sistemde cevaplanmış sorular (adil kıyas)
    ids = None
    for s in systems:
        sid = {r["id"] for r in by_sys[s]}
        ids = sid if ids is None else ids & sid
    common = [r for r in recs if r["id"] in ids]
    by_sys_c = {s: [r for r in common if r["system"] == s] for s in systems}

    md = [f"# Karar modeli kıyaslaması — turkish-decisions-max-19k örneklemi\n",
          f"Sistemler: {', '.join(systems)}. Ortak soru sayısı: {len(ids)}.\n",
          "\n## Özet (ortak sorular)\n",
          "| sistem | n | acc % | score MAE ↓ | TVD ↓ | ECE ↓ | hata % | s/soru |", "|---|---|---|---|---|---|---|---|"]
    for s in systems:
        m = summarize(by_sys_c[s])
        md.append(f"| {s} | {m['n']} | {fmt(m['acc'], pct=True)} | {fmt(m['mae'], nd=2)} | {fmt(m['tvd'])} | {fmt(m['ece'])} | "
                  f"{fmt(m['err'], pct=True)} | {m['sec']:.2f} |")
    md.append("\nTVD ve ECE yalnızca olasılık üreten sistemler için (prompt'lu LLM one-hot verir). "
              "score MAE: ordinal sorularda tahmin ile doğru düzey arasındaki ortalama basamak farkı.")

    def group(key):
        g = defaultdict(dict)
        for s in systems:
            for r in by_sys_c[s]:
                g[r[key]].setdefault(s, []).append(r)
        return g

    # risk-kapsama: güven eşiğinin üstündeki sorular otomatik çözülmüş sayılır
    soft_sys = [s for s in systems if is_soft(by_sys_c[s])]
    if soft_sys:
        md.append("\n### Risk–kapsama (olasılık üreten sistemler)\n")
        md.append("Eşik: top-prob ≥ t olan sorular otomatik çözülür, kalanı inceleme kuyruğuna düşer.\n")
        md.append("| sistem | eşik | kapsama % | kapsamdaki doğruluk % | kalan (kuyruk) doğruluk % |")
        md.append("|---|---|---|---|---|")
        for s in soft_sys:
            rs = [r for r in by_sys_c[s] if r["pred"]]
            for t in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
                inn = [r for r in rs if max(r["pred"]) >= t]; out = [r for r in rs if max(r["pred"]) < t]
                cov = len(inn) / len(rs)
                acc_in = sum(is_correct(r) for r in inn) / len(inn) if inn else None
                acc_out = sum(is_correct(r) for r in out) / len(out) if out else None
                md.append(f"| {s} | {t:.2f} | {100*cov:.1f} | {fmt(acc_in, pct=True)} | {fmt(acc_out, pct=True)} |")
        # soru bazında, eşik 0.8
        md.append("\nEşik 0,80'de soru bazında kapsama / kapsamdaki doğruluk:\n")
        md.append("| soru | " + " | ".join(soft_sys) + " |"); md.append("|---|" + "---|" * len(soft_sys))
        g = group("skill")
        for k in sorted(g):
            cells = []
            for s in soft_sys:
                rs = [r for r in g[k].get(s, []) if r["pred"]]
                inn = [r for r in rs if max(r["pred"]) >= 0.8]
                cells.append(f"{100*len(inn)/len(rs):.0f}% / {fmt(sum(is_correct(r) for r in inn)/len(inn) if inn else None, pct=True)}" if rs else "–")
            md.append(f"| {k} | " + " | ".join(cells) + " |")

    md.append(table(group("skill"), systems, "Görev ailesine göre doğruluk (%)"))
    md.append(table(group("question_type"), systems, "Soru tipine göre"))
    md.append(table(group("language"), systems, "Belge diline göre"))
    md.append(table(group("difficulty"), systems, "Zorluğa göre (setin yazılış netliği)"))
    md.append(table(group("has_injection"), systems, "Prompt injection içeren belgeler (True) vs. değil"))

    Path(res_dir, "report.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))

    # grafikler
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        g = group("skill")
        skills = sorted(g)
        w = 0.8 / max(1, len(systems))
        fig, ax = plt.subplots(figsize=(12, 5))
        for i, s in enumerate(systems):
            vals = [100 * summarize(g[k][s])["acc"] if s in g[k] else 0 for k in skills]
            ax.bar([j + i * w for j in range(len(skills))], vals, w, label=s)
        ax.set_xticks([j + w * (len(systems) - 1) / 2 for j in range(len(skills))])
        ax.set_xticklabels(skills, rotation=45, ha="right")
        ax.set_ylabel("doğruluk (%)"); ax.set_ylim(0, 100); ax.legend(); ax.set_title("Görev ailesine göre doğruluk")
        fig.tight_layout(); fig.savefig(Path(res_dir, "accuracy_by_skill.png"), dpi=150)

        soft = [s for s in systems if is_soft(by_sys_c[s])]
        if soft:
            fig, ax = plt.subplots(figsize=(5, 5))
            for s in soft:
                pairs = [(max(r["pred"]), is_correct(r)) for r in by_sys_c[s] if r["pred"]]
                xs, ys = [], []
                for b in range(10):
                    inb = [(c, ok) for c, ok in pairs if b / 10 < c <= (b + 1) / 10]
                    if inb:
                        xs.append(sum(c for c, _ in inb) / len(inb)); ys.append(sum(ok for _, ok in inb) / len(inb))
                ax.plot(xs, ys, "o-", label=s)
            ax.plot([0, 1], [0, 1], "k--", lw=1)
            ax.set_xlabel("güven (top prob)"); ax.set_ylabel("doğruluk"); ax.set_title("Kalibrasyon"); ax.legend()
            fig.tight_layout(); fig.savefig(Path(res_dir, "calibration.png"), dpi=150)
        print(f"\ngrafikler: {res_dir}/accuracy_by_skill.png, calibration.png")
    except ImportError:
        print("matplotlib yok, grafik atlandı")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results")
