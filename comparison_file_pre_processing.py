import pandas as pd
import os

klasor_yolu = r"E:\Muhammed Özel\Bitirme Projesi\test_verileri"
subat_dosyasi = "2026.02.xlsx" 
tam_yol = os.path.join(klasor_yolu, subat_dosyasi)

def urun_siniflandir(urun_adi):
    u = str(urun_adi).lower()
    if any(x in u for x in ["1. sınıf", "2. sınıf", "3. sınıf", "4. sınıf"]): return "İlkokul (1-4)"
    elif any(x in u for x in ["5. sınıf", "6. sınıf", "7. sınıf"]): return "Ortaokul (5-7)"
    elif "8. sınıf" in u or "lgs" in u: return "LGS Grubu"
    elif any(x in u for x in ["9. sınıf", "10. sınıf", "11. sınıf"]): return "Lise (9-11)"
    elif any(x in u for x in ["12. sınıf", "tyt", "ayt", "yks"]): return "YKS/Üniversite Hazırlık"
    elif any(x in u for x in ["kpss", "dgs", "ales"]): return "Sınav Hazırlık (Memurluk/Lisansüstü)"
    return "Diğer"

print(f"{subat_dosyasi} işleniyor...")
df = pd.read_excel(tam_yol, engine='openpyxl')
df['Sinif_Grubu'] = df['Ürün Adı'].apply(urun_siniflandir)

ozet = df.groupby('Sinif_Grubu')['Net Satış Adedi'].sum().reset_index()

cikti_adi = "2026_02_Kategorize_Analiz.xlsx"
ozet.to_excel(cikti_adi, index=False)

print(f"\nİşlem tamam! '{cikti_adi}' dosyası oluşturuldu.")
print("-" * 30)
print(ozet)
