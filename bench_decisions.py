"""
cisimcik/turkish-decisions-max-19k üzerinde karar modeli kıyaslaması.

Sistemler:
  tau        -> cisimcik/tau-4b (System One karar modeli, tek forward pass, kalibre olasılık)
  llm:<id>   -> herhangi bir HF chat modeli, prompt ile zero-shot (örn. llm:cisimcik/cisimcik-4b,
                llm:Qwen/Qwen3.5-4B-Instruct). Sadece argmax üretir, olasılık yok.
  jev        -> TypeSafe System One API (--jev-url, --jev-key). Aynı istek/yanıt şeması varsayılır.

Örnek:
  python bench_decisions.py --system tau --n-per-config 30
  python bench_decisions.py --system llm:cisimcik/cisimcik-4b --n-per-config 30
  python report.py results/

Not: Örneklem seed'e bağlı ve tüm sistemler için aynıdır; sonuçlar results/<sistem>.jsonl'a
satır satır yazılır, kesilen koşu kaldığı yerden devam eder.
"""
import argparse
import json
import os
import random
import re
import time
from pathlib import Path

DATASET = "cisimcik/turkish-decisions-max-19k"
CONFIGS = ["documents", "general", "summarization", "translation", "reading_qa", "nli", "coding",
           "computer_use", "instruction_following", "messy_data", "assistant_judge", "grammar",
           "odd_inputs", "exam_mcqa", "tool_calling", "game_agent", "problem_solving", "pragmatics"]


# ----------------------------------------------------------------------------- veri
def build_sample(n_per_config: int, seed: int, max_state_tokens: int, configs=None):
    """Her görev ailesinden n satır; tüm sistemler aynı seed ile aynı örneği görür."""
    from datasets import load_dataset
    rng = random.Random(seed)
    rows = []
    for cfg in (configs or CONFIGS):
        ds = load_dataset(DATASET, cfg, split="train")
        ds = ds.filter(lambda r: r["state_tokens"] <= max_state_tokens)
        idx = list(range(len(ds)))
        rng.shuffle(idx)
        for i in idx[:n_per_config]:
            r = ds[i]
            rows.append({k: r[k] for k in (
                "id", "document_id", "state", "state_format", "question_type", "instructions",
                "criteria", "options", "probs", "answer", "skill", "language", "lang_mode",
                "difficulty", "has_injection", "state_tokens", "disagreement", "register")})
    return rows


def parse_state(row):
    if row["state_format"] == "json":
        try:
            return json.loads(row["state"])
        except json.JSONDecodeError:
            return row["state"]
    return row["state"]


def build_question(row):
    return {"type": row["question_type"], "instructions": row["instructions"],
            "criteria": json.loads(row["criteria"])}


# ----------------------------------------------------------------------------- sistemler
class TauSystem:
    name = "tau"

    def __init__(self, device=None):
        import torch
        from transformers import AutoModel
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModel.from_pretrained("cisimcik/tau-4b", trust_remote_code=True,
                                               dtype=torch.bfloat16).to(dev).eval()

    def predict(self, row):
        """options sırasına göre olasılık vektörü döndürür."""
        ans = self.model.decide(state=parse_state(row), questions={"q": build_question(row)})["answers"]["q"]
        return answer_to_probs(ans, row)


class JevSystem:
    """TypeSafe POST /v1/systemone (şema tau ile aynı). 429/529'da üstel bekleme, 401/422'de hemen hata."""

    def __init__(self, url, key, model="jev-1.13.0"):
        import requests
        self.s = requests.Session()
        self.s.headers.update({"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        self.url, self.model = url.rstrip("/") + "/v1/systemone", model
        self.name = "jev_" + re.sub(r"[^A-Za-z0-9]+", "-", model)
        self.tokens = 0

    def call(self, state, questions):
        body = {"model": self.model, "state": state, "questions": questions}
        for attempt in range(6):
            r = self.s.post(self.url, json=body, timeout=120)
            if r.status_code in (429, 529, 500, 502, 503):
                time.sleep(min(60, 2 ** attempt) + random.random())
                continue
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
            data = r.json()
            self.tokens += data.get("usage", {}).get("input_tokens", 0)
            return data
        raise RuntimeError("Jev: 6 denemede de 429/5xx")

    def predict(self, row):
        data = self.call(parse_state(row), {"q": build_question(row)})
        return answer_to_probs(data["answers"]["q"], row)


def answer_to_probs(ans, row):
    opts = row["options"]
    t = row["question_type"]
    if t == "noul":
        p = float(ans["noul"])
        return [p, 1.0 - p]                      # options = ["true","false"]
    if t == "choice":
        pr = ans["probabilities"]
        return [float(pr.get(o, 0.0)) for o in opts]
    if t == "score":
        pr = ans["probabilities"]                # anahtarlar "0".."n-1", options sırası = düzey sırası
        return [float(pr.get(str(i), 0.0)) for i in range(len(opts))]
    raise ValueError(t)


class LLMSystem:
    """Prompt'la zero-shot; sadece argmax. Olasılık yok -> TVD/ECE hesaplanmaz."""

    SYS = ("Sen bir sınıflandırma asistanısın. Sana bir durum ve bir soru verilecek. "
           "SADECE istenen cevabı yaz, açıklama ekleme.")

    def __init__(self, model_id, device=None, max_new_tokens=16):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.name = "llm_" + re.sub(r"[^A-Za-z0-9]+", "-", model_id)
        self.model_id = model_id
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).to(dev).eval()
        self.max_new_tokens = max_new_tokens

    def _prompt(self, row):
        state = row["state"]
        q = build_question(row)
        t = row["question_type"]
        if t == "noul":
            crit = q["criteria"] if isinstance(q["criteria"], dict) else {}
            task = (f"Soru: {q['instructions']}\n"
                    f"evet = {crit.get('true', '')}\nhayır = {crit.get('false', '')}\n"
                    "Cevap olarak sadece 'evet' ya da 'hayır' yaz.")
        elif t == "choice":
            crit = q["criteria"]
            opts = "\n".join(f"- {k}: {crit.get(k) or ''}" for k in row["options"])
            task = (f"Soru: {q['instructions']}\nSeçenekler:\n{opts}\n"
                    "Cevap olarak sadece seçeneğin anahtarını (tire başındaki kelimeyi) yaz.")
        else:  # score
            levels = "\n".join(f"{i}: {o}" for i, o in enumerate(row["options"]))
            task = (f"Soru: {q['instructions']}\nDüzeyler (düşükten yükseğe):\n{levels}\n"
                    "Cevap olarak sadece düzeyin numarasını yaz.")
        return [{"role": "system", "content": self.SYS},
                {"role": "user", "content": f"Durum:\n{state}\n\n{task}"}]

    def predict(self, row):
        import torch
        msgs = self._prompt(row)
        kw = {"add_generation_prompt": True, "tokenize": False}
        try:
            prompt = self.tok.apply_chat_template(msgs, enable_thinking=False, **kw)
        except TypeError:
            prompt = self.tok.apply_chat_template(msgs, **kw)
        inputs = self.tok(prompt, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        text = self.tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        text = text.split("</think>")[-1].strip().lower()
        return self._parse(text, row)

    def _parse(self, text, row):
        opts = row["options"]
        t = row["question_type"]
        onehot = lambda i: [1.0 if j == i else 0.0 for j in range(len(opts))]
        if t == "noul":
            if re.search(r"\bevet\b|\byes\b|\btrue\b", text): return onehot(0)
            if re.search(r"\bhay[ıi]r\b|\bno\b|\bfalse\b", text): return onehot(1)
            return None
        if t == "score":
            m = re.search(r"\d+", text)
            if m and int(m.group()) < len(opts): return onehot(int(m.group()))
            return None
        # choice: tam eşleşme, sonra içerme
        for i, o in enumerate(opts):
            if text == str(o).lower(): return onehot(i)
        for i, o in enumerate(opts):
            if str(o).lower() in text: return onehot(i)
        return None


# ----------------------------------------------------------------------------- koşu
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True, help="tau | llm:<hf_model_id> | jev")
    ap.add_argument("--file", default=None, help="hazır JSONL (make_kariyer_set.py çıktısı); verilirse HF seti kullanılmaz")
    ap.add_argument("--n-per-config", type=int, default=30)
    ap.add_argument("--configs", nargs="*", default=None, help="alt küme; varsayılan 18 config")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-state-tokens", type=int, default=2048, help="CPU'da süreyi sınırlar")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--jev-url", default=os.environ.get("JEV_URL", "https://api.typesafe.ai"))
    ap.add_argument("--jev-key", default=os.environ.get("JEV_API_KEY"))
    ap.add_argument("--jev-model", default="jev-1.13.0", help="tekrarlanabilirlik için sürümü sabitle")
    ap.add_argument("--limit", type=int, default=None, help="ilk N satır (deneme koşusu için)")
    args = ap.parse_args()

    if args.system == "tau":
        sysm = TauSystem(args.device)
    elif args.system.startswith("llm:"):
        sysm = LLMSystem(args.system[4:], args.device)
    elif args.system == "jev":
        if not args.jev_key: raise SystemExit("JEV_API_KEY gerekli")
        sysm = JevSystem(args.jev_url, args.jev_key, args.jev_model)
    else:
        raise SystemExit("bilinmeyen sistem")

    if args.file:
        with open(args.file, encoding="utf-8") as f:
            rows = [json.loads(l) for l in f if l.strip()]
    else:
        rows = build_sample(args.n_per_config, args.seed, args.max_state_tokens, args.configs)
    if args.limit:
        rows = rows[:args.limit]
    out_path = Path(args.out_dir) / f"{sysm.name}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            done = {json.loads(l)["id"] for l in f if l.strip()}
    print(f"{len(rows)} soru, {len(done)} hazır -> {out_path}")

    with open(out_path, "a", encoding="utf-8") as f:
        for k, row in enumerate(rows, 1):
            if row["id"] in done: continue
            t0 = time.perf_counter()
            try:
                pred = sysm.predict(row)
                err = None
            except Exception as e:  # modelin kabul etmediği kriterler vb.
                pred, err = None, f"{type(e).__name__}: {e}"[:300]
            rec = {kk: row[kk] for kk in ("id", "document_id", "skill", "question_type", "language",
                                           "lang_mode", "difficulty", "has_injection", "state_tokens",
                                           "disagreement", "register", "options", "probs")}
            rec.update({"system": sysm.name, "pred": pred, "error": err,
                        "seconds": round(time.perf_counter() - t0, 4)})
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if k % 10 == 0:
                print(f"  {k}/{len(rows)}  son: {rec['seconds']:.2f}s  {row['skill']}/{row['question_type']}")
    if hasattr(sysm, "tokens"):
        print(f"Jev input token toplamı (bu koşu): {sysm.tokens:,}")
    print("bitti.")


if __name__ == "__main__":
    main()
