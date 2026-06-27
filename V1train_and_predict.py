import numpy as np
import pandas as pd
import tensorflow as tf
import random
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

test_cikti_klasor_yolu = r"E:\Muhammed Özel\Bitirme Projesi\test_sonuclari"

# 1. TEKRARLANABİLİRLİK VE DETERMINIZM (REPRODUCIBILITY)
# Derin öğrenme modellerinin stokastik doğasından kaynaklanan farklı sonuçları 
# engellemek ve deneysel tekrarlanabilirliği sağlamak amacıyla tüm rastgelelik tohumları sabitlenmiştir.
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

# 2. VERİ YÜKLEME VE ÖLÇEKLENDİRME (MAX-ABSOLUTE SCALING)
df = pd.read_csv("hazir_veri_seti_gruplanmis.csv", index_col='Sinif_Grubu')
data = df.values

# Her bir sınıf grubunun kendi dinamik yapısını korumak amacıyla satır bazlı maksimum değer 
# normalizasyonu uygulanmıştır. Sıfıra bölme hatası (division by zero) kontrol edilmiştir.
max_values = data.max(axis=1, keepdims=True)
max_values[max_values == 0] = 1 
scaled_data = data / max_values

# Model Parametreleri
look_back = 12  # Geriye dönük zaman penceresi büyüklüğü (12 aylık periyot)
n_future = int(input("Kaç ay sonrasını tahmin etmek istersiniz? (Örn: 11): "))
n_trials = int(input("Kaç farklı eğitim turu yapılsın? (Örn: 5): "))

# 3. VERİ MATRİSİNİN LSTM GİRİŞ FORMATINA DÖNÜŞTÜRÜLMESİ (3D TENSOR PREPARATION)
# Zaman serisi verileri ardışık pencereler halinde ayrıştırılarak modelin girdi (X) ve 
# hedef (y) matrisleri oluşturulmuş ve LSTM mimarisinin beklediği 3 boyutlu tensör yapısına getirilmiştir.
X, y = [], []
for group in scaled_data:
    for i in range(len(group) - look_back):
        X.append(group[i:(i + look_back)])
        y.append(group[i + look_back])

X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# 4. ÇOKLU EĞİTİM VE ANANSAMBL (ENSEMBLING) DÖNGÜSÜ
# Tek bir eğitimin yerel minimuma (local minima) takılma riskini azaltmak amacıyla 
# model n_trials kez bağımsız olarak eğitilmekte ve sonuçların ortalaması alınmaktadır.
all_trial_results = []
for trial in range(n_trials):
    print(f"\nTur {trial+1}/{n_trials} eğitiliyor...")
    
    # Derin LSTM mimarisi: Aşırı öğrenmeyi (overfitting) engellemek amacıyla ardışık 
    # katmanlar arasına Dropout regularizasyonu eklenmiştir.
    model = Sequential([
        Input(shape=(look_back, 1)),
        LSTM(128, return_sequences=True),
        Dropout(0.3), 
        LSTM(64),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')

    # Eğitim kaybının (loss) duraklaması durumunda eğitimi erken kesen optimizasyon mekanizması
    callback = EarlyStopping(monitor='loss', patience=15, restore_best_weights=True)

    model.fit(X, y, epochs=100, batch_size=4, verbose=0, callbacks=[callback])

    # İleriye Yönelik Özyinelemeli Projeksiyon (Recursive Forecasting)
    trial_preds = {}
    for i, group_name in enumerate(df.index):
        current_batch = scaled_data[i, -look_back:].reshape(1, look_back, 1)
        group_preds = []
        for _ in range(n_future):
            pred = model.predict(current_batch, verbose=0)[0][0]

            # Mantıksal ve Fiziki Sınırlama (Domain-Specific Clipping):
            # Modelin gerçeklik dışı negatif değerler veya uç değer sapmaları (outlier) üretmesi engellenmiştir.
            pred = max(0, min(pred, 1.5)) 
            group_preds.append(pred)
            current_batch = np.append(current_batch[:, 1:, :], [[[pred]]], axis=1)

        # Normalizasyonun tersine çevrilerek gerçek satış adet ölçeğine dönülmesi
        trial_preds[group_name] = [round(p * max_values[i][0]) for p in group_preds]

    all_trial_results.append(pd.DataFrame(trial_preds).T)

# 5. AMANSAMBL SONUÇLARININ MERKEZİ EĞİLİM (MEAN) İLE BİRLEŞTİRİLMESİ
# Farklı eğitim turlarından elde edilen matrisler grup bazında normalize edilerek 
# nihai ve varyansı düşürülmüş tahmin değerleri elde edilmiştir.
final_res = pd.concat(all_trial_results).groupby(level=0).mean().round(0)
final_res.columns = [f"Ay_{i+1}" for i in range(n_future)]

# 6. BULGULARIN EXCEL FORMATINDA RAPORLANMASI
dosya_adi = "version1_tahmin_sonuclari_ortalama_batch4_lookback12_trials10.xlsx"
tam_yol = os.path.join(test_cikti_klasor_yolu, dosya_adi)
final_res.to_excel(tam_yol)

print("\n" + "="*50)
print("İşlem Başarıyla Tamamlandı!")
print(f"{dosya_adi}' olarak kaydedildi.")
print("="*50)
