"""
Jev API'yi tek istekle yoklar: anahtar çalışıyor mu, cevap şeması beklediğimiz gibi mi, Türkçe sorun var mı.
Kullanım:  JEV_API_KEY=... python jev_probe.py
"""
import json
import os
import sys

import requests

KEY = os.environ.get("JEV_API_KEY")
if not KEY:
    sys.exit("JEV_API_KEY ortam değişkeni gerekli")
URL = os.environ.get("JEV_URL", "https://api.typesafe.ai").rstrip("/") + "/v1/systemone"
MODEL = os.environ.get("JEV_MODEL", "jev-1.13.0")

# tau demo'sundaki ilan-1 ile aynı sorular -> iki modeli yan yana görebilirsin
state = ("Kıdemli Backend Geliştirici (Python)\nİstanbul merkezli fintech şirketimiz için en az 6 yıl deneyimli, "
         "Django ve PostgreSQL ile ölçekli sistemler geliştirmiş bir backend geliştirici arıyoruz. Haftada 2 gün ofis, "
         "3 gün uzaktan çalışma modeli uygulanmaktadır. Teknik dokümantasyon İngilizce olduğundan ileri düzey İngilizce "
         "şarttır. Maaş aralığı: 180.000-220.000 TL brüt, özel sağlık sigortası ve yemek kartı sağlanır.")
questions = {
    "kidem": {"type": "score", "instructions": "İlanın aradığı kıdem seviyesi nedir?",
              "criteria": ["Stajyer / yeni mezun", "Junior (0-2 yıl)", "Mid-level (2-5 yıl)", "Senior (5+ yıl)", "Lead / yönetici"]},
    "uzaktan": {"type": "noul", "instructions": "Pozisyon uzaktan (remote) ya da hibrit çalışmaya açık mı?",
                "criteria": {"true": "Uzaktan veya hibrit imkânı var", "false": "Tamamen ofis ya da belirtilmemiş"}},
    "departman": {"type": "choice", "instructions": "Bu pozisyon hangi departmana ait?",
                  "criteria": {"yazilim": "Yazılım, veri, IT", "satis": "Satış, iş geliştirme", "finans": "Muhasebe, finans",
                               "diger": "Yukarıdakilerin hiçbiri"}},
}

r = requests.post(URL, headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
                  json={"model": MODEL, "state": state, "questions": questions}, timeout=60)
print("HTTP", r.status_code)
try:
    data = r.json()
except ValueError:
    sys.exit(r.text[:500])
print(json.dumps(data, ensure_ascii=False, indent=1))

if r.status_code == 200:
    a = data["answers"]
    ok = (a["kidem"]["type"] == "score" and "probabilities" in a["kidem"] and "legend" in a["kidem"]
          and a["uzaktan"]["type"] == "noul" and isinstance(a["uzaktan"]["noul"], (int, float))
          and a["departman"]["type"] == "choice" and "probabilities" in a["departman"])
    print("\nŞema tau ile uyumlu:", ok, "| model:", data.get("model"), "| token:", data.get("usage"))
    print("Beklenen: kidem Senior(3) civarı, uzaktan ~1, departman yazilim")
