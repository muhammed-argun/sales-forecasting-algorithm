import numpy as np
import pandas as pd
import tensorflow as tf
import random
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

# 1. DİZİN YAPILANDIRMASI VE ÇEVRESEL AYARLAR
test_cikti_klasor_yolu = r"E:\Muhammed Özel\Bitirme Projesi\test_sonuclari"
if not os.path.exists(test_cikti_klasor_yolu):
    os.makedirs(test_cikti_klasor_yolu)

# Modelin başlangıç ağırlıklarındaki varyansı dağıtmak amacıyla stokastik yaklaşım 
# benimsenmiş, her çalıştırma öncesi Keras arka plan oturumu sıfırlanmıştır.
tf.keras.backend.clear_session()

# 2. VERİ ÖLÇEKLENDİRME VE HİPERPARAMETRE OPTİMİZASYONU
df = pd.read_csv("hazir_veri_seti_gruplanmis.csv", index_col='Sinif_Grubu')
data = df.values

# Grupların hacimsel büyüklüklerinden kaynaklanan yanlılığı (scale bias) ekarte etmek 
# amacıyla satır bazlı lokal Min-Max normalizasyonu uygulanmıştır.
mins = data.min(axis=1, keepdims=True)
maxs = data.max(axis=1, keepdims=True)
ranges = maxs - mins
ranges[ranges == 0] = 1 
scaled_data = (data - mins) / ranges

# AMPİRİK ANALİZLERE DAYALI OPTİMUM MODEL PARAMETRELERİ
look_back = 12       # Zaman penceresi büyüklüğü (Gecikme derecesi / Lag = 12)
n_future = 12        # Projeksiyon ufku (Forecast Horizon: 12 Aylık İleri Tahmin)
n_trials = 20        # Güven aralığını daraltmak ve tahmini kararlı kılmak adına simülasyon sayısı
batch_size = 4       # Gradyan güncellemeleri için optimize edilmiş mini-yığın boyutu

# 3. VERİ MATRİSLERİNİN ARDIŞIK LSTM FORMATINA GETİRİLMESİ
X, y = [], []
for group in scaled_data:
    for i in range(len(group) - look_back):
        X.append(group[i:(i + look_back)])
        y.append(group[i + look_back])

X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# 4. SİMÜLASYON, EĞİTİM VE ASİMETRİK TAHMİN PROSEDÜRÜ
all_trial_results = []
for trial in range(n_trials):
    # Her simülasyon turunun bağımsız parametrelerle başlaması için bellek temizlenir.
    tf.keras.backend.clear_session()
    print(f"Tur {trial+1}/{n_trials} eğitiliyor...")
    
    # Optimizasyon testleri sonucunda yapılandırılmış esnek LSTM regresyon mimarisi
    model = Sequential([
        Input(shape=(look_back, 1)),
        LSTM(64, return_sequences=False), 
        Dropout(0.25),  # Ağın aşırı öğrenmesini (overfitting) sınırlayan regularizasyon oranı
        Dense(32, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')
    callback = EarlyStopping(monitor='loss', patience=20, restore_best_weights=True)
    model.fit(X, y, epochs=120, batch_size=batch_size, verbose=0, callbacks=[callback])

    # Özyinelemeli Çok Adımlı Projeksiyon Döngüsü
    trial_preds = {}
    for i, group_name in enumerate(df.index):
        current_batch = scaled_data[i, -look_back:].reshape(1, look_back, 1)
        group_preds = []
        baslangic_ayi = 6  # Serinin bitiş dönemini takip eden projeksiyon başlangıcı (Haziran)

        for j in range(n_future):
            pred = model.predict(current_batch, verbose=0)[0][0]
            su_anki_ay = (baslangic_ayi + j - 1) % 12 + 1
            
            # RAFİNE ASİMETRİK MEVSİMSEL KISITLAR (REFINED SEASONAL BOUNDS)
            # Önceki iterasyonlardan elde edilen hata analizleri doğrultusunda, eğitim-yayıncılık 
            # takviminin tepe noktaları (Eylül, Şubat) ile geçiş dönemleri (Ekim, Mart) ve stabil aylar 
            # için kademeli ve dinamik bir sınırlandırma maskesi (heuristic constraint) kurgulanmıştır.
            if su_anki_ay in [9, 2]:
                ust_limit = 3.2     # Ana Mevsimsel Şok Sınırı (Eylül / Şubat)
            elif su_anki_ay in [10, 3]:
                ust_limit = 2.0     # İkincil Dalgalanma Sınırı (Ekim / Mart)
            else:
                ust_limit = 1.25    # Stabil Sezon Sınırı (Diğer Aylar)
            
            pred = max(0, min(pred, ust_limit))
            group_preds.append(pred)
            current_batch = np.append(current_batch[:, 1:, :], [[[pred]]], axis=1)

        # Doğrusal ölçeklendirmenin geriye döndürülerek ham veriye dönüştürülmesi
        trial_preds[group_name] = [max(0, round(p * ranges[i][0] + mins[i][0])) for p in group_preds]

    all_trial_results.append(pd.DataFrame(trial_preds).T)

# 5. SIMÜLASYON MATRİSLERİNİN MERKEZİ EĞİLİM METRİĞİ İLE BİRLEŞTİRİLMESİ
# Stokastik varyasyonların etkisini minimize etmek adına 20 farklı bağımsız modelin 
# projeksiyon çıktıları grup düzeyinde merkezi eğilim (ortalama) yöntemiyle birleştirilmiştir.
final_res = pd.concat(all_trial_results).groupby(level=0).mean().round(0)
final_res.columns = [f"Ay_{i+1}" for i in range(n_future)]

# 6. NİHAİ RAPORUN EXCEL FORMATINDA DIŞA AKTARILMASI
dosya_adi = "version4_tahmin_sonuclari_ortalama_batch4_lookback12_trials20.xlsx"
tam_yol = os.path.join(test_cikti_klasor_yolu, dosya_adi)
final_res.to_excel(tam_yol)

print(f"\n{dosya_adi} dosyasına kaydedildi.")
