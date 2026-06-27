import numpy as np
import pandas as pd
import tensorflow as tf
import random
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

# 1. DİZİN YAPILANDIRMASI VE DETERMINIZM (REPRODUCIBILITY)
test_cikti_klasor_yolu = r"E:\Muhammed Özel\Bitirme Projesi\test_sonuclari"
if not os.path.exists(test_cikti_klasor_yolu):
    os.makedirs(test_cikti_klasor_yolu)

# Derin öğrenme modelinin stokastik doğasını kontrol altına almak ve deneysel
# tutarlılığı (reproducibility) sağlamak amacıyla tüm rastgelelik tohumları sabitlenmiştir.
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

# 2. VERİ SETİNİN OKUNMASI VE SATIR BAZLI MIN-MAX ÖLÇEKLENDİRME
df = pd.read_csv("hazir_veri_seti_gruplanmis.csv", index_col='Sinif_Grubu')
data = df.values

# Satış hacmi düşük olan alt grupların, yüksek hacimli gruplar arasında baskılanmasını
# (dominance effect) engellemek amacıyla satır bazlı (local) Min-Max ölçeklendirme uygulanmıştır.
mins = data.min(axis=1, keepdims=True)
maxs = data.max(axis=1, keepdims=True)
ranges = maxs - mins
ranges[ranges == 0] = 1  # Sıfıra bölme hatasının (division by zero) ekarte edilmesi
scaled_data = (data - mins) / ranges

# Model Hiperparametreleri
look_back = 12  # Geriye dönük bakılacak zaman penceresi (1 yıllık lag yapısı)
n_future = int(input("Kaç ay sonrasını tahmin etmek istersiniz? (Örn: 12): "))
n_trials = int(input("Kaç farklı eğitim turu yapılsın? (Örn: 5): "))

# 3. KAYDIRMALI PENCERE (SLIDING WINDOW) YÖNTEMİYLE 3D TENSÖR HAZIRLIĞI
X, y = [], []
for group in scaled_data:
    for i in range(len(group) - look_back):
        X.append(group[i:(i + look_back)])
        y.append(group[i + look_back])

X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# 4. ÇOKLU EĞİTİM VE ASİMETRİK ÖZYİNELEMELİ TAHMİN DÖNGÜSÜ (ENSEMBLE FORECAST)
# Modelin yerel minimumlara (local minima) takılmasını önlemek ve genel varyansı 
# düşürmek amacıyla bağımsız eğitim turları (ensemble approach) kurgulanmıştır.
all_trial_results = []
for trial in range(n_trials):
    print(f"\nTur {trial+1}/{n_trials} eğitiliyor...")
    
    # LSTM Tabanlı Regresyon Mimarisi
    model = Sequential([
        Input(shape=(look_back, 1)),
        LSTM(64, return_sequences=False), 
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')
    
    # Overfitting'i engellemek amacıyla eğitim kaybını izleyen erken durdurma mekanizması
    callback = EarlyStopping(monitor='loss', patience=15, restore_best_weights=True)
    model.fit(X, y, epochs=100, batch_size=2, verbose=0, callbacks=[callback])

    # Özyinelemeli Çok Adımlı Projeksiyon (Recursive Multi-Step Forecasting)
    trial_preds = {}
    for i, group_name in enumerate(df.index):
        current_batch = scaled_data[i, -look_back:].reshape(1, look_back, 1)
        group_preds = []
        baslangic_ayi = 6  # Zaman serisinin bitiş noktasını takip eden ilk projeksiyon ayı (Haziran)

        for j in range(n_future):
            pred = model.predict(current_batch, verbose=0)[0][0]
            
            # MEVSİMSEL TALEP ŞOKLARI VE ASİMETRİK SINIRLANDIRMA (ASYMMETRIC CLIPPING)
            # Yayıncılık ve eğitim sektörünün doğası gereği (okul açılış ve sınav dönemleri), 
            # modelin Eylül, Ekim, Şubat ve Mart aylarındaki olası mevsimsel talep şoklarını (peaks)
            # yakalayabilmesi için asimetrik ve dinamik bir üst sınır kısıtı uygulanmıştır.
            su_anki_ay = (baslangic_ayi + j - 1) % 12 + 1
            
            if su_anki_ay in [9, 10, 2, 3]:
                ust_limit = 4.0  # Yüksek sezon / Talep patlaması dönemi üst sınırı
            else:
                ust_limit = 1.3  # Düşük sezon / Stabil dönem üst sınırı
            
            pred = max(0, min(pred, ust_limit))
            group_preds.append(pred)
            
            # Zaman penceresinin yeni tahmin değeriyle güncellenerek kaydırılması
            current_batch = np.append(current_batch[:, 1:, :], [[[pred]]], axis=1)

        # Ölçeklendirmenin tersine çevrilerek özgün satış adet ölçeğine dönülmesi
        trial_preds[group_name] = [max(0, round(p * ranges[i][0] + mins[i][0])) for p in group_preds]

    all_trial_results.append(pd.DataFrame(trial_preds).T)

# 5. DENEYSEL SONUÇLARIN MERKEZİ EĞİLİM (MEAN) İLE BİRLEŞTİRİLMESİ
final_res = pd.concat(all_trial_results).groupby(level=0).mean().round(0)
final_res.columns = [f"Ay_{i+1}" for i in range(n_future)]

# 6. TAHMİN BULGULARININ RAPORLANMASI VE DIŞA AKTARILMASI
dosya_adi = "version2_tahmin_sonuclari_ortalama_batch2_lookback12_trials5.xlsx"
tam_yol = os.path.join(test_cikti_klasor_yolu, dosya_adi)
final_res.to_excel(tam_yol)

print("\n" + "="*50)
print("İşlem Başarıyla Tamamlandı!")
print(f"Sonuçlar şu klasöre kaydedildi:\n{tam_yol}")
print("="*50)
