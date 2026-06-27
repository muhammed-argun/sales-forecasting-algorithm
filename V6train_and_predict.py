import numpy as np
import pandas as pd
import tensorflow as tf
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# 1. DİZİN YAPILANDIRMASI VE AMBİYANS AYARLARI
test_cikti_klasor_yolu = r"E:\Muhammed Özel\Bitirme Projesi\test_sonuclari"
if not os.path.exists(test_cikti_klasor_yolu):
    os.makedirs(test_cikti_klasor_yolu)

# 2. VERİ YÜKLEME VE LOKAL MIN-MAX ÖLÇEKLENDİRME
df = pd.read_csv("hazir_veri_seti_gruplanmis.csv", index_col='Sinif_Grubu')
data = df.values

# Alt kategorilerin hacimsel farklarından kaynaklanan varyans sapmalarını nötrlemek 
# amacıyla satır bazlı lokal Min-Max normalizasyon protokolü sürdürülmüştür.
mins, maxs = data.min(axis=1, keepdims=True), data.max(axis=1, keepdims=True)
ranges = maxs - mins
ranges[ranges == 0] = 1 
scaled_data = (data - mins) / ranges

# Model Hiperparametre Seti
look_back = 12   # Geriye dönük bakılacak zaman penceresi büyüklüğü (Lag = 12)
n_future = 12    # Projeksiyon ufku (Forecast Horizon = 12 Aylık İleri Tahmin)
n_trials = 15    # Stokastik varyansı minimize etmeye yönelik bağımsız simülasyon sayısı
batch_size = 4   # Mini-yığın (mini-batch) boyutu

# 3. ZAMAN SERİSİNİN 3 BOYUTLU TENSÖR FORMATINA GETİRİLMESİ
X, y = [], []
for group in scaled_data:
    for i in range(len(group) - look_back):
        X.append(group[i:(i + look_back)])
        y.append(group[i + look_back])
X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# 4. ROBUST MODELLERİN EĞİTİMİ VE ÖZYİNELEMELİ PROJEKSİYON DÖNGÜSÜ
all_trial_results = []
for trial in range(n_trials):
    # Simülasyonlar arası bağımsızlığı korumak adına Keras oturumu sıfırlanır
    tf.keras.backend.clear_session()
    print(f"Tur {trial+1}/{n_trials} eğitiliyor...")
    
    # Derinleştirilmiş Ardışık LSTM Mimarisi ve Gelişmiş Aktivasyon Stratejisi:
    # Standart ReLU fonksiyonunda görülebilen ölü nöron sorununu (dying ReLU problem) 
    # ekarte etmek ve negatif gradyan akışını korumak için LeakyReLU tercih edilmiştir.
    model = Sequential([
        Input(shape=(look_back, 1)),
        LSTM(100, return_sequences=True), 
        LSTM(50),
        Dropout(0.25),
        Dense(64),
        LeakyReLU(negative_slope=0.1), 
        Dense(1)
    ])

    # Hata karesi yerine mutlak hata tabanlı kayıp fonksiyonu (MAE) entegre edilmiştir.
    # Bu tercih, modelin uç değerlere (outliers/ani sıçramalar) karşı daha dirençli (robust) olmasını sağlar.
    model.compile(optimizer='adam', loss='mae') 
    
    # --- DİNAMİK ÖĞRENME ORANI ZAMANLAMASI VE REGULARİZASYON ---
    # Erken durdurma protokolüne ek olarak, eğitim kaybının duraklama (plateau) evrelerine girmesi
    # durumunda öğrenme oranını dinamik olarak azaltan ReduceLROnPlateau mekanizması kurgulanmıştır.
    early_stop = EarlyStopping(monitor='loss', patience=15, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=7, min_lr=0.0001)
    
    model.fit(X, y, epochs=150, batch_size=batch_size, verbose=0, callbacks=[early_stop, reduce_lr])

    # Özyinelemeli Çok Adımlı Projeksiyon (Recursive Forecasting)
    trial_preds = {}
    for i, group_name in enumerate(df.index):
        current_batch = scaled_data[i, -look_back:].reshape(1, look_back, 1)
        group_preds = []
        for j in range(n_future):
            pred = model.predict(current_batch, verbose=0)[0][0]
            
            # Fiziki gerçeklik kısıtı (Negatif satış adetlerinin engellenmesi)
            pred = max(0, pred) 
            group_preds.append(pred)
            current_batch = np.append(current_batch[:, 1:, :], [[[pred]]], axis=1)

        # Ölçeklendirmenin tersine çevrilerek özgün satış adet ölçeğine dönülmesi
        trial_preds[group_name] = [round(p * ranges[i][0] + mins[i][0]) for p in group_preds]
    all_trial_results.append(pd.DataFrame(trial_preds).T)

# 5. SİMÜLASYON MATRİSLERİNİN MERKEZİ EĞİLİM METRİĞİ İLE BİRLEŞTİRİLMESİ
final_res = pd.concat(all_trial_results).groupby(level=0).mean().round(0)
final_res.columns = [f"Ay_{i+1}" for i in range(n_future)]

# 6. NİHAİ RAPORUN EXCEL FORMATINDA DIŞA AKTARILMASI
final_res.to_excel(os.path.join(test_cikti_klasor_yolu, "version6_tahmin_sonuclari_ortalama_batch4_lookback12_trials15.xlsx"))
print("\nAnalitik olarak güçlendirilmiş V6 sonuçları hazır.")
