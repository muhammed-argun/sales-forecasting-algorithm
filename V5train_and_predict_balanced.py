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

# 2. VERİ SETİNİN OKUNMASI VE LOKAL MIN-MAX ÖLÇEKLENDİRME
df = pd.read_csv("hazir_veri_seti_gruplanmis.csv", index_col='Sinif_Grubu')
data = df.values

# Alt kategorilerin hacimsel farklarından doğan ölçek sapmalarını nötrlemek amacıyla
# satır bazlı lokal Min-Max normalizasyonu tercih edilmiştir.
mins = data.min(axis=1, keepdims=True)
maxs = data.max(axis=1, keepdims=True)
ranges = maxs - mins
ranges[ranges == 0] = 1 
scaled_data = (data - mins) / ranges

# Model Hiperparametreleri
look_back = 12   # Geriye dönük bakılacak zaman penceresi büyüklüğü (Lag = 12)
n_future = 12    # Projeksiyon ufku (Forecast Horizon = 12 Aylık İleri Tahmin)
n_trials = 15    # Stokastik varyansı minimize etmeye yönelik bağımsız simülasyon sayısı
batch_size = 4   # Mini-yığın (mini-batch) boyutu

# 3. ZAMAN SERİSİNİN 3 BOYUTLU GİRDİ MATRİSİNE DÖNÜŞTÜRÜLMESİ
X, y = [], []
for group in scaled_data:
    for i in range(len(group) - look_back):
        X.append(group[i:(i + look_back)])
        y.append(group[i + look_back])

X, y = np.array(X), np.array(y)
y = y.reshape(-1, 1)  # Hedef değişken matris işlemleri için (N, 1) boyutuna getirilmiştir
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# ÖRNEKLEM AĞIRLIKLANDIRMA FONKSİYONU (SAMPLE WEIGHTING FUNCTION)
# Modelin yüksek hacimli talep dönemlerindeki (tepe noktaları/peaks) tahmin başarısını artırmak 
# ve bu kritik dönemlerdeki hataları daha ağır cezalandırmak amacıyla doğrusal bir örneklem 
# ağırlık matrisi (sample_weights) kurgulanmıştır. Bu yaklaşım, modelin asimetrik kayıp duyarlılığını artırır.
sample_weights = 1.0 + (y.flatten() * 5.0) 

# 4. AĞIRLIKLANDIRILMIŞ MODEL EĞİTİMİ VE ÖZYİNELEMELİ TAHMİN DÖNGÜSÜ
all_trial_results = []
for trial in range(n_trials):
    # Simülasyonlar arası bağımsızlığı korumak adına Keras oturumu sıfırlanır
    tf.keras.backend.clear_session()
    print(f"Tur {trial+1}/{n_trials} eğitiliyor... (Yüksek Değer Odaklı Ağırlıklandırma Aktif)")
    
    # LSTM Tabanlı Regresyon Mimarisi
    model = Sequential([
        Input(shape=(look_back, 1)),
        LSTM(64, return_sequences=False), 
        Dropout(0.2),  # Aşırı öğrenmeyi (overfitting) sınırlayan regularizasyon oranı
        Dense(32, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')
    
    # Ağırlıklandırılmış eğitim sürecinde, yüksek hacimli örneklerin gradyan üzerindeki 
    # kararlı etkisini yakalayabilmek adına sabır (patience) katsayısı optimize edilmiştir.
    callback = EarlyStopping(monitor='loss', patience=20, restore_best_weights=True)
    
    # Model eğitimine örneklem ağırlıklarının (sample_weight) entegre edilmesi
    model.fit(X, y, 
              sample_weight=sample_weights, 
              epochs=120, 
              batch_size=batch_size, 
              verbose=0, 
              callbacks=[callback])

    # Özyinelemeli Çok Adımlı Projeksiyon (Recursive Forecasting)
    trial_preds = {}
    for i, group_name in enumerate(df.index):
        current_batch = scaled_data[i, -look_back:].reshape(1, look_back, 1)
        group_preds = []
        baslangic_ayi = 6  # Tahmin döngüsünün başlayacağı ilk indeks (Haziran)

        for j in range(n_future):
            pred = model.predict(current_batch, verbose=0)[0][0]
            
            # Fiziki gerçeklik kısıtı (Satış adetleri alt sınırı = 0)
            pred = max(0, pred) 
            group_preds.append(pred)
            
            # Zaman penceresinin yeni tahmin değeriyle güncellenerek kaydırılması
            current_batch = np.append(current_batch[:, 1:, :], [[[pred]]], axis=1)

        # Ölçeklendirme parametrelerinin tersine çevrilerek özgün satış adet ölçeğine dönülmesi
        trial_preds[group_name] = [max(0, round(p * ranges[i][0] + mins[i][0])) for p in group_preds]

    all_trial_results.append(pd.DataFrame(trial_preds).T)

# 5. SİMÜLASYON MATRİSLERİNİN MERKEZİ EĞİLİM METRİĞİ İLE BİRLEŞTİRİLMESİ
final_res = pd.concat(all_trial_results).groupby(level=0).mean().round(0)
final_res.columns = [f"Ay_{i+1}" for i in range(n_future)]

# 6. NİHAİ RAPORUN EXCEL FORMATINDA DIŞA AKTARILMASI
dosya_adi = "V5_Value_Balanced_Final.xlsx"
tam_yol = os.path.join(test_cikti_klasor_yolu, dosya_adi)
final_res.to_excel(tam_yol)

print("\n" + "="*50)
print("İşlem Başarıyla Tamamlandı!")
print(f"Yeni ağırlıklı tahminler kaydedildi: {dosya_adi}")
print("="*50)
