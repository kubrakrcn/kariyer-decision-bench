# kariyer-decision-bench

Türkçe **karar modelleri** (System One: tau-4b, Jev) ile prompt'lu LLM'leri gerçek iş ilanlarında aynı sorular, aynı
metrikler ve aynı örneklemle kıyaslayan düzenek. Kariyer.net Ar-Ge'de ilan metninden kıdem, departman, çalışma şekli
gibi kararları çıkarma işi için yazıldı; başka bir alan için soru setini değiştirmek yeterli.

Yazı: *Üç Karar Modeli, 600 İlan* (Medium, yakında).

## Sonuçlar (600 ilan × 7 soru = 4.200 karar)

| sistem | doğruluk | kıdem MAE ↓ | ECE ↓ | s/soru |
| --- | --- | --- | --- | --- |
| Jev 1.13.0 (kapalı, API) | **71,5** | **0,40** | 0,156 | 0,18 |
| cisimcik-4b (prompt, zero-shot) | 65,7 | 0,44 | – | 0,16 |
| tau-4b (açık karar modeli) | 64,6 | 0,53 | **0,093** | 0,08 |

- tau-4b kendi eğitim verisinde %95,7; gerçek ilanlarda %64,6. Bir modeli yayımladığı veri setinde kıyaslamayın.
- Üç sistem taksonomi sorularında aynı duvara çarpıyor (%40-61): 40 hatanın el incelemesinde 17'si etiket hatası,
  16'sı taksonomi belirsizliği, 7'si gerçek model hatası.
- Jev daha doğru ama daha az kalibre: aynı eşikte çok daha fazla soruya "eminim" diyor, emin olduğunda tau kadar haklı.
- Seçenek açıklaması eklemek tau'ya +2,2 puan (segment +10), Jev'e +0,2 ve daha kötü kalibrasyon getiriyor.

Tam rapor ve grafikler `results/` altında. Ayrıntılar yazıda.

## Kurulum

```bash
pip install -r requirements.txt
pip install flash-linear-attention      # isteğe bağlı, NVIDIA GPU'da tau için
export JEV_API_KEY=...                   # yalnızca Jev koşusu için
```
tau-4b bf16'da ~9 GB; 12 GB+ VRAM ya da 16 GB RAM CPU.

## Dosyalar

| dosya | ne yapar |
| --- | --- |
| `tau_demo_jobs.py` | 3 örnek ilanda tau-4b: 5 tipli soru, biri prompt-injection tuzağı |
| `make_kariyer_set.py` | `jobs.csv` + pozisyon taksonomisi → tabakalı 600 ilan, 7 soru, `kariyer_set.jsonl` |
| `add_descriptions.py` | taksonomi sorularının seçeneklerine bir satır açıklama ekler (ablasyon) |
| `jev_probe.py` | Jev API'ye tek istek: anahtar ve şema kontrolü |
| `bench_decisions.py` | `--system tau \| jev \| llm:<hf-model>`; HF seti ya da `--file` ile kendi setin; kaldığı yerden devam eder |
| `report.py` | `results/*.jsonl` → `report.md` + doğruluk/kalibrasyon grafikleri; acc, MAE, TVD, ECE, risk-kapsama |

## Koşu

```bash
# 1) veri seti (jobs.csv ve taksonomi dosyası elinizde olmalı)
python make_kariyer_set.py --jobs jobs.csv --tax full-taxonomy-knet.csv --n 600
python add_descriptions.py                       # kariyer_set_desc.jsonl

# 2) sistemler, aynı dosya
python bench_decisions.py --system tau --file kariyer_set.jsonl --out-dir results
python bench_decisions.py --system llm:cisimcik/cisimcik-4b --file kariyer_set.jsonl --out-dir results
python bench_decisions.py --system jev --file kariyer_set.jsonl --out-dir results --limit 20   # önce deneme
python bench_decisions.py --system jev --file kariyer_set.jsonl --out-dir results

# 3) rapor (yalnızca tüm sistemlerde cevaplanmış ortak sorular üzerinden)
python report.py results/
```

tau'nun kendi veri setinde koşmak için `--file` yerine `--n-per-config 30` (18 görev ailesi × 30 soru).

## Veri formatı

Her satır bir (ilan, soru) çifti; `bench_decisions.py` ve `report.py` bu alanları bekler:

```json
{"id": "123:kidem", "document_id": "123", "state": "<ilan metni>", "state_format": "text",
 "question_type": "score", "instructions": "Bu ilanın hedeflediği kıdem seviyesi nedir?",
 "criteria": "[\"Giriş seviyesi\", ..., \"Müdür ve üstü\"]",
 "options": ["Giriş seviyesi", "..."], "probs": [0, 0, 1, 0, 0, 0, 0], "answer": "...",
 "skill": "kidem", "language": "tr", "lang_mode": "tr/tr", "difficulty": "n/a",
 "has_injection": false, "state_tokens": 300, "disagreement": 0.0, "register": "real"}
```
`criteria` System One istek şemasındaki haliyle JSON string: noul `{"true": ..., "false": ...}`, choice
`{"<anahtar>": "<açıklama|null>"}`, score `["<en düşük>", ..., "<en yüksek>"]`. `probs` soft etiket olabilir.

## Veri seti

İlan metinleri (`kariyer_set.jsonl`) repoda değil. `results/*.jsonl` dosyaları ilan metni içermez; satır başına id, seçenekler, etiket ve tahmin vardır, rapor bunlardan yeniden üretilebilir.

## Sınırlar

Tek koşu, tek seed, güven aralığı yok. Etiketler işverenin seçtiği pozisyonun taksonomideki yerinden; elle
doğrulanmadı. LLM kolu 4B ve düz prompt. Jev `jev-1.13.0` olarak sabitlendi; her soru ayrı istek (ilan başına tek
istek 7 kat ucuz olurdu, tau ile aynı koşul için ayrı gönderildi).

## Kaynaklar

[cisimcik/tau-4b](https://huggingface.co/cisimcik/tau-4b) ·
[cisimcik/cisimcik-4b](https://huggingface.co/cisimcik/cisimcik-4b) ·
[turkish-decisions-max-19k](https://huggingface.co/datasets/cisimcik/turkish-decisions-max-19k) ·
[TypeSafe Jev API](https://typesafe-jev.com/en/guides/api/)

Lisans: Apache 2.0.
