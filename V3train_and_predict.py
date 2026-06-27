import numpy as np
import pandas as pd
import tensorflow as tf
import random
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

# 1. DİZİN YAPILANDIRMASI VE STOKASTİK SİMÜLASYON AYARLARI
test_cikti_klasor_yolu = r"E:\Muhammed Özel\Bitirme Projesi\test_sonuclari"
if not os.path.exists(test_cikti_klasor_yolu):
    os.makedirs(test_cikti_klasor_yolu)

# DETERMINIZM YERİNE ENSEMBLE/STOKASTİK YAKLAŞIM SEÇİMİ
# Modelin tek bir başlangıç ağırlığına (initial weights) bağımlı kalmasını önlemek, 
# yerel minimum (local minima) riskini dağıtmak ve genelleme (generalization) yeteneğini 
# maksimize etmek amacıyla global rastgelelik tohumları (seed) serbest bırakılmıştır.
# Çoklu turların (n_trials) ortalaması alınarak daha kararlı bir regresyon yüzeyi elde edilecektir.

# 2. VERİ YÜKLEME VE YEREL MIN-MAX ÖLÇEKLENDİRME
df = pd.read_csv("hazir_veri_seti_gruplanmis.csv", index_col='Sinif_Grubu')
data = df.values

# Kategoriler arasındaki ölçek farklarının (hacim sapmalarının) derin öğrenme modeli 
# üzerindeki baskınlık etkisini (dominance) nötrlemek adına lokal Min-Max dönüşümü uygulanmıştır.
mins = data.min(axis=1, keepdims=True)
maxs = data.max(axis=1, keepdims=True)
ranges = maxs - mins
ranges[ranges == 0] = 1 
scaled_data = (data - mins) / ranges

# Deney Parametreleri
look_back = 12  # Geriye dönük bakılacak zaman penceresi büyüklüğü (Lag = 12)
n_future = int(input("Kaç ay sonrasını tahmin etmek istersiniz? (Örn: 12): "))
n_trials = int(input("Kaç farklı eğitim turu yapılsın? (Örn: 5): "))

# 3. ZAMAN SERİSİNİN 3 BOYUTLU GİRDİ MATRİSİNE DÖNÜŞTÜRÜLMESİ
X, y = [], []
for group in scaled_data:
    for i in range(len(group) - look_back):
        X.append(group[i:(i + look_back)])
        y.append(group[i + look_back])

X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# 4. PARALEL SIMÜLASYON VE ÖZYİNELEMELİ TAHMİN DÖNGÜSÜ
all_trial_results = []
for trial in range(n_trials):
    # Bellek sızıntısını önlemek ve her simülasyonda ağırlık matrislerinin 
    # bağımsız (stokastik) olarak baştan yaratılmasını sağlamak için oturum temizlenir.
    tf.keras.backend.clear_session()
    
    print(f"\nTur {trial+1}/{n_trials} eğitiliyor... (Rastgele Başlangıç Ağırlıklarıyla)")
    
    # Çok Katmanlı Ardışık LSTM Mimarisi (Kapasitesi artırılmış derin model)
    model = Sequential([
        LSTM(100, return_sequences=True), 
        LSTM(50),
        Dropout(0.25),  # Aşırı öğrenmeyi (overfitting) dizginlemek için regularizasyon kısıtı
        Dense(64),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')
    
    # Modelin karmaşık örüntüleri yakalayabilmesi için sabır (patience) katsayısı 
    # artırılmış ve optimum ağırlıkları geri yükleyen erken durdurma protokolü eklenmiştir.
    callback = EarlyStopping(monitor='loss', patience=20, restore_best_weights=True)
    model.fit(X, y, epochs=100, batch_size=4, verbose=0, callbacks=[callback])

    # Özyinelemeli Çok Adımlı Projeksiyon (Recursive Multi-Step Forecasting)
    trial_preds = {}
    for i, group_name in enumerate(df.index):
        current_batch = scaled_data[i, -look_back:].reshape(1, look_back, 1)
        group_preds = []
        baslangic_ayi = 6  # Tahmin projeksiyonunun başlayacağı indeks (Haziran)

        for j in range(n_future):
            pred = model.predict(current_batch, verbose=0)[0][0]
            
            # --- MEVSİMSEL TALEP ŞOKLARI KONTROLÜ VE ASİMETRİK SINIRLANDIRMA ---
            # Akademik/Eğitim takvimindeki dönemsel talep dalgalanmalarını (Eylül-Ekim / Şubat-Mart) 
            # modele yansıtmak amacıyla dinamik üst sınır kısıtlaması uygulanmıştır.
            su_anki_ay = (baslangic_ayi + j - 1) % 12 + 1
            
            if su_anki_ay in [9, 10, 2, 3]:
                ust_limit = 4.0  # Talep patlaması dönemi kısıtı
            else:
                ust_limit = 1.3  # Stabil dönem kısıtı
            
            pred = max(0, min(pred, ust_limit))
            group_preds.append(pred)
            current_batch = np.append(current_batch[:, 1:, :], [[[pred]]], axis=1)

        # Ölçeklendirme parametrelerinin tersine çevrilerek ham satış verisi ölçeğine dönülmesi
        trial_preds[group_name] = [max(0, round(p * ranges[i][0] + mins[i][0])) for p in group_preds]

    all_trial_results.append(pd.DataFrame(trial_preds).T)

# 5. SİMÜLASYON SONUÇLARININ MERKEZİ EĞİLİM (MEAN) İLE BİRLEŞTİRİLMESİ
# Stokastik eğitim sürecinden elde edilen varyant sonuçlar birleştirilerek, 
# uç sapmalardan arındırılmış (robust) nihai anansembl tahmin matrisi hesaplanmıştır.
final_res = pd.concat(all_trial_results).groupby(level=0).mean().round(0)
final_res.columns = [f"Ay_{i+1}" for i in range(n_future)]

# 6. BULGULARIN DIŞA AKTARILMASI
dosya_adi = "version3_tahmin_sonuclari_ortalama_batch4_lookback12_trials25.LTSM_update.xlsx"
tam_yol = os.path.join(test_cikti_klasor_yolu, dosya_adi)
final_res.to_excel(tam_yol)

print("\n" + "="*50)
print("İşlem Başarıyla Tamamlandı!")
print(f"Stokastik varyasyonlar minimize edildi ve {n_trials} turun ortalaması alındı.")
print(f"Yeni dosya: {dosya_adi}")
print("="*50)
