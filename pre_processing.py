import pandas as pd
import glob
import os

# Veri Setinin Okunması ve Sıralanması
# Analize dahil edilecek aylık ham verilerin dosya yolları listelenmekte ve kronolojik tutarlılık için sıralanmaktadır.
path = r"E:\Muhammed Özel\Bitirme Projesi\test_veri"
files = sorted(glob.glob(os.path.join(path, "*.xlsx")))

def urun_siniflandir(urun_adi):
    """
    Kural Tabanlı Kategorizasyon Fonksiyonu:
    Ürün isimlerindeki anahtar kelimeleri (string matching) temel alarak 
    ürünleri akademik ve sınav gruplarına göre homojen sınıflara ayırır.
    """
    u = str(urun_adi).lower()
    if any(x in u for x in ["1. sınıf", "2. sınıf", "3. sınıf", "4. sınıf"]): return "İlkokul (1-4)"
    elif any(x in u for x in ["5. sınıf", "6. sınıf", "7. sınıf"]): return "Ortaokul (5-7)"
    elif "8. sınıf" in u or "lgs" in u: return "LGS Grubu"
    elif any(x in u for x in ["9. sınıf", "10. sınıf", "11. sınıf"]): return "Lise (9-11)"
    elif any(x in u for x in ["12. sınıf", "tyt", "ayt", "yks"]): return "YKS/Üniversite Hazırlık"
    elif any(x in u for x in ["kpss", "dgs", "ales"]): return "Sınav Hazırlık (Memurluk/Lisansüstü)"
    return "Diğer"

# Veri Ön İşleme ve Birleştirme (Data Aggregation)
all_months_data = []
for file in files:
    # Dosya adından dönem (yıl-ay) bilgisi çıkarılmaktadır
    date_str = os.path.basename(file).replace('.xlsx', '')
    df = pd.read_excel(file, engine='openpyxl')
    
    # Sınıflandırma fonksiyonunun uygulanması ve meta-veri eklenmesi
    df['Sinif_Grubu'] = df['Ürün Adı'].apply(urun_siniflandir)
    df['Donem'] = date_str
    
    # Analizde kullanılacak hedef değişkenlerin (feature selection) seçilmesi
    all_months_data.append(df[['Sinif_Grubu', 'Donem', 'Net Satış Adedi']])

# Tüm aylık verilerin tek bir ana veri çerçevesinde (master dataframe) birleştirilmesi
df_master = pd.concat(all_months_data)

# Zaman Serisi Matrisinin Oluşturulması (Pivot Table)
# Veriler sınıf grupları ve dönem bazında optimize edilerek zaman serisi formatına dönüştürülmüştür.
# Eksik gözlemler (NaN değerleri) veri bütünlüğü için 0 ile doldurulmuştur.
ts_data = df_master.pivot_table(index='Sinif_Grubu', columns='Donem', values='Net Satış Adedi', aggfunc='sum').fillna(0)

# Sütunların (Dönemlerin) kronolojik olarak sıralanmasının garanti edilmesi
ts_data = ts_data.reindex(sorted(ts_data.columns), axis=1)

# Hazırlanan nihai veri setinin sonraki analiz aşamaları için dışa aktarılması
ts_data.to_csv("hazir_veri_seti_gruplanmis_tumu.csv")
print("Temiz veri seti hazır!")
