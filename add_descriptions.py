"""
kariyer_set.jsonl -> kariyer_set_desc.jsonl : alan / meslek_sinifi / segment sorularına seçenek açıklaması ekler.
Diğer sorular ve etiketler aynen kalır. Kullanım: python add_descriptions.py
"""
import json

ALAN = {
    "Bilim ve Araştırma": "Laboratuvar, akademik araştırma, Ar-Ge bilimcisi",
    "Diğer": "Hiçbir alana net oturmayan pozisyonlar",
    "Enerji": "Elektrik üretimi, santral, akaryakıt, doğalgaz, yenilenebilir enerji",
    "Eğitim ve Öğretim": "Öğretmen, eğitmen, akademisyen, kurs ve okul kadrosu",
    "Finans ve İdari İşleri": "Muhasebe, finans, denetim, insan kaynakları, idari işler, ofis yönetimi",
    "Güvenlik ve Eminiyet": "Güvenlik görevlisi, iş güvenliği, koruma",
    "Hayvancılık, Tarım ve Doğal Kaynaklar": "Ziraat, çiftlik, veterinerlik, ormancılık, madencilik",
    "Hizmet İşleri": "Temizlik, tesis hizmetleri, çağrı merkezi, destek personeli, genel hizmet",
    "Hukuk, Sosyal ve Toplum Hizmetleri": "Avukat, hukuk müşaviri, sosyal hizmet, dernek/vakıf, kamu hizmeti",
    "Konaklama, Turizm ve Eğlence": "Otel, restoran, kafe, turizm, etkinlik ve organizasyon",
    "Mühendislik ve İmalat": "Üretim, fabrika, makine/endüstri/elektrik mühendisliği, kalite, bakım",
    "Sanat, Tasarım ve Medya": "Grafik tasarım, görsel sanatlar, basın, yayın, kurumsal iletişim, reklam",
    "Sağlık ve Tıp": "Doktor, hemşire, tekniker, eczane, hastane ve klinik personeli",
    "Spor ve Fitness": "Antrenör, spor salonu, fitness eğitmeni",
    "Taşımacılık ve Lojistik": "Şoför, kurye, depo, sevkiyat, lojistik operasyonu, havacılık/denizcilik",
    "Teknoloji, Bilişim ve Veri": "Yazılım, IT, veri, sistem/ağ, teknik destek (bilişim)",
    "Web Önyüz Geliştirme": "Front-end web geliştirici",
    "İnşaat, Teknik ve Mimarlık": "İnşaat, şantiye, mimarlık, tesisat, teknik saha işleri",
    "İşletme, Satış ve Pazarlama": "Satış, mağaza, iş geliştirme, pazarlama, müşteri ilişkileri, genel yönetim",
}

SEGMENT = {
    "Nitelik gerektirmeyen meslekler": "Vasıfsız işçi, eleman, personel; özel eğitim gerektirmeyen işler",
    "Nitelikli meslek mensupları ve uzmanlar": "Uzman, mühendis, memur, analist; lisans düzeyi bireysel katkı",
    "Profesyonel meslek mensupları, orta kademe yöneticiler ve kıdemli çalışanlar": "Kıdemli uzman, müdür yardımcısı, orta kademe yönetici",
    "Satış ve hizmet personelleri": "Satış temsilcisi, mağaza personeli, garson, resepsiyonist, çağrı merkezi",
    "Stajyerler": "Stajyer, uzun dönem stajyer, yeni mezun programı",
    "Teknisyen ve operatörler": "Teknisyen, tekniker, makine/üretim operatörü",
    "Usta, formen, sorumlu şef ve amirler": "Usta, formen, vardiya amiri, saha şefi",
    "Yöneticiler": "Müdür, direktör, genel müdür, bölüm/birim yöneticisi",
}

MESLEK = {
    "Acil Durum ve Afet Yönetimi": "Afet, acil durum, itfaiye, arama kurtarma",
    "Arge": "Araştırma geliştirme mühendisi/uzmanı, ürün geliştirme",
    "Basın Medya": "Gazeteci, editör, muhabir, yayıncılık, kurumsal iletişim",
    "Bilgi Teknolojileri": "Yazılım, sistem, ağ, veri, IT destek",
    "Bilim": "Laboratuvar, araştırmacı, bilim insanı",
    "Danışmanlık": "Danışman, müşavir (yönetim, eğitim, finans danışmanlığı)",
    "Denetim": "İç denetim, bağımsız denetim, müfettiş, kontrolör",
    "Denizcilik": "Gemi, liman, denizcilik personeli",
    "Depolama / Dağıtım": "Depo, sevkiyat, dağıtım, forklift",
    "Diğer": "Hiçbir sınıfa net oturmayan pozisyonlar",
    "Elektrik / Elektronik": "Elektrik/elektronik mühendisi, teknisyeni, pano, otomasyon",
    "Endüstri": "Endüstri mühendisi, süreç iyileştirme, planlama",
    "Enerji": "Enerji santrali, elektrik üretim/dağıtım, akaryakıt istasyonu",
    "Eğitim": "Öğretmen, eğitmen, akademisyen, okul idaresi, kurumsal eğitim",
    "Finans": "Muhasebe, finans, bütçe, krediler, hazine, mali işler",
    "Güvenlik": "Güvenlik görevlisi, koruma, güvenlik amiri",
    "Gıda": "Gıda mühendisi, gıda üretimi, aşçı, mutfak",
    "Havacılık": "Havayolu, havalimanı, uçuş, yer hizmetleri",
    "Hizmet": "Temizlik, tesis hizmetleri, genel hizmet elemanı",
    "Hukuk": "Avukat, hukuk müşaviri, sözleşme yönetimi, uyum",
    "Kalite": "Kalite kontrol, kalite güvence, kalite mühendisi",
    "Kamu": "Kamu kurumu, belediye, devlet memurluğu",
    "Kimya": "Kimya mühendisi, kimyager, laboratuvar",
    "Madencilik & Enerji": "Maden, maden mühendisi, ocak işletmesi",
    "Makine": "Makine mühendisi, makine bakım, mekanik",
    "Mühendislik": "Genel mühendislik, teknik proje, diğer mühendislik dalları",
    "Müşteri Hizmetleri": "Çağrı merkezi, müşteri temsilcisi, müşteri deneyimi",
    "Ofis Destek": "Sekreter, asistan, resepsiyon, ofis elemanı",
    "Operasyon": "Operasyon uzmanı/yöneticisi, saha operasyonu",
    "Otomotiv": "Otomotiv üretimi, servis, yedek parça, araç satışı",
    "Pazarlama": "Pazarlama, marka, dijital pazarlama, ürün pazarlama",
    "Perakende": "Mağaza, mağaza müdürü, kasiyer, reyon, satış danışmanı (mağaza)",
    "Proje / İş / Ürün Yönetimi": "Proje yöneticisi, iş analisti, ürün yöneticisi, PMO",
    "Reklamcılık": "Reklam ajansı, kreatif, medya planlama",
    "Sanat": "Grafik tasarım, görsel tasarım, sanat yönetmeni, fotoğraf",
    "Satış": "Satış temsilcisi, satış uzmanı/yöneticisi, iş geliştirme",
    "Satış - Pazarlama": "Satış ve pazarlamayı birlikte yürüten pozisyonlar",
    "Spor": "Antrenör, spor eğitmeni, fitness",
    "Strateji ve Raporlama / Danışmanlık": "Strateji, iş geliştirme analizi, raporlama, yönetim danışmanlığı",
    "Taşımacılık / Lojistik": "Şoför, kurye, lojistik uzmanı, nakliye, ithalat-ihracat operasyonu",
    "Teknik Hizmetler": "Teknik servis, bakım-onarım, tesisat, teknik destek (saha)",
    "Tekstil": "Tekstil, konfeksiyon, moda üretimi",
    "Telekom": "Telekomünikasyon, GSM, şebeke",
    "Ticaret": "Dış ticaret, ithalat-ihracat, satın alma, tedarik",
    "Turizm / Etkinlik / Organizasyon": "Otel, restoran, kafe, tur, etkinlik ve organizasyon",
    "Tıp / Sağlık": "Doktor, hemşire, sağlık teknikeri, eczacı, hastane personeli",
    "Vakıf / Dernek / Sosyal Hizmetler": "STK, dernek, vakıf, sosyal hizmet uzmanı",
    "Ziraat / Tarım / Hayvancılık": "Ziraat mühendisi, çiftlik, veteriner, tarım işçisi",
    "Çevre / İş Güvenliği ve İşçi Sağlığı": "İSG uzmanı, çevre mühendisi, işyeri hekimi",
    "Üretim": "Üretim işçisi, operatör, üretim mühendisi/müdürü, fabrika",
    "Üst Yönetim": "Genel müdür, direktör, C-level, genel müdür yardımcısı",
    "İdari İşler": "İdari işler, ofis yönetimi, satın alma (idari), tesis yönetimi",
    "İnsan Kaynakları": "İK uzmanı, işe alım, bordro, eğitim ve gelişim",
    "İnşaat": "İnşaat mühendisi, şantiye, mimar, inşaat işçisi, proje (inşaat)",
}

DESC = {"alan": ALAN, "meslek_sinifi": MESLEK, "segment": SEGMENT}

n = 0
with open("kariyer_set.jsonl", encoding="utf-8") as f, open("kariyer_set_desc.jsonl", "w", encoding="utf-8") as out:
    for line in f:
        r = json.loads(line)
        if r["skill"] in DESC:
            crit = json.loads(r["criteria"])
            missing = [k for k in crit if k not in DESC[r["skill"]]]
            assert not missing, (r["skill"], missing)
            r["criteria"] = json.dumps({k: DESC[r["skill"]][k] for k in crit}, ensure_ascii=False)
            n += 1
        out.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"{n} satıra açıklama eklendi -> kariyer_set_desc.jsonl")
