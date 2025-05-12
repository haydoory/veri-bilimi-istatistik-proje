#/////////////////////////Birinci Bölüm : hayder saad //////////////////////////////
# 1. Gerekli kütüphaneleri içe aktar
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import ttest_ind
from scipy.stats import f_oneway
from scipy.stats import mannwhitneyu, kruskal, spearmanr
import numpy as np 

# --------------------------------------------------------------------
# 2. Veriyi oku
# CSV dosyasını oku ve ayırıcı olarak ; kullan
# Not: Dosya yolu senin bilgisayarına göre güncellenmiş

df = pd.read_csv("C:/Users/hayde/Desktop/archive/Student-Depression-Dataset-görültülü.csv", sep=';')

# --------------------------------------------------------------------
# 3. Veri tipi bilgilerini ve eksik değerleri kontrol et
print(df.info())
print(df.isnull().sum())
# Eksik değer yüzdesini hesapla ve yüzde olarak gösterme
missing_percentage = (df.isnull().sum() / len(df)) * 100
 
# Sonucu daha okunabilir şekilde yazdırma
print("Eksik Değer Yüzdesi (%):")
print(missing_percentage.round(2).sort_values(ascending=False))
# --------------------------------------------------------------------
# 4. Eksik verileri temizleme

# ===================== AÇIKLAMA =====================
# Veri setini analiz etmeye başlamadan önce, eksik değer oranlarını kontrol ettim.
# Yaptığım analizde tüm sütunlardaki eksik değer yüzdesi %0.07'nin altında olduğunu gördüm.
# Bu nedenle, eksik değer içeren satırları güvenle sildim (dropna()).

df_cleaned = df.dropna()
print(df_cleaned.isnull().sum())

# --------------------------------------------------------------------


# ===================== YAŞ VERİSİ TEMİZLİĞİ =====================
# Yaş sütunundaki veriler analiz edilmeden önce filtrelenmiştir.
# Normal üniversite öğrencisi yaş aralığı dikkate alınarak yalnızca 18 yaş ve üzeri, 35 yaş altı bireyler analizde tutulmuştur.
# Bu filtre ile çocuk yaşlar (örneğin 6–17) ve aşırı değerler (40+, 70+) analiz dışı bırakılmıştır.
# Bu sayede daha homojen bir öğrenci kitlesi elde edilmiştir.
# İlk satır temizlenmiş veri kümesinden yaş dağılımını gösterir;
# ikinci satır ise orijinal veri kümesindeki yaşların frekans dağılımını yazdırır.
# ===============================================================


# age control etme
df_cleaned = df_cleaned[(df_cleaned["Age"] >= 18) & (df_cleaned["Age"] < 35)]
print(df_cleaned["Age"].value_counts())
print(df["Age"].value_counts())
# --------------------------------------------------------------------


# ===================== CİNSİYET VERİSİ TEMİZLİĞİ =====================
# Gender sütunundaki değerlerin sayısal analizlerde kullanılabilmesi için 
# kategorik (yazılı) ifadeler sayısal kodlamaya dönüştürülmüştür.
# "Male" → 0, "Female" → 1 olarak yeniden kodlanmıştır.
# Bu işlem, istatistiksel testlerde ve grafiklerde kolaylık sağlar.
# replace() fonksiyonu ile doğrudan dönüşüm yapılmıştır.
# İlk satır dönüşümden önceki değer sayılarını kontrol eder.
# ===============================================================



#gender sutunu temizleme
print(df_cleaned["Gender"].value_counts())
df_cleaned["Gender"] = df_cleaned["Gender"].replace({"Male": 0, "Female": 1})


# --------------------------------------------------------------------


# ===================== ŞEHİR VERİSİ TEMİZLİĞİ =====================
# İlk olarak City sütununda "City" şeklinde hatalı bir değer tespit edilmiştir (muhtemelen başlık satırının yanlışlıkla veri olarak alınması).
# Bu satır filtrelenerek veri kümesinden çıkarılmıştır.

# Ardından şehirlerin frekans dağılımı incelenmiş ve istatistiksel anlamlılık açısından
# sadece 10 ve üzeri sayıda öğrenci içeren şehirler analizde tutulmuştur.

# Bu işlem, çok az temsil edilen şehirlerin neden olabileceği gürültüyü (noise) azaltmak ve
# analiz sonuçlarının daha güvenilir olmasını sağlamak için yapılmıştır.

# Sonuç olarak: yeterli gözlem sayısına sahip şehirler ile daha homojen ve sağlam bir veri kümesi elde edilmiştir.
# ===============================================================



# city sutunu temizleme
print(df_cleaned["City"].value_counts())
df_cleaned = df_cleaned[df_cleaned["City"] != "City"]

# Şehirlerin dağılımına bak
print(df_cleaned["City"].value_counts())

# 10'dan az olanları filtreleme
city_counts = df_cleaned["City"].value_counts()
valid_cities = city_counts[city_counts >= 10].index

# Sadece geçerli şehirleri içeren veri kümesi
df_cleaned = df_cleaned[df_cleaned["City"].isin(valid_cities)]
print(df_cleaned["City"].value_counts())
      
# ----------------------------------------------------------


# ===================== MESLEK (PROFESSION) SÜTUNUNUN KALDIRILMASI =====================
# Profession sütunu incelendiğinde, verinin büyük oranda dağınık, tutarsız veya istatistiksel analiz açısından anlam taşımadığı gözlemlenmiştir.
# Ayrıca öğrencilerin eğitim durumu zaten Degree sütununda temsil edildiği için meslek bilgisi fazlalık oluşturmaktadır.
# Bu nedenle veri setinin sadeleştirilmesi ve analizlerin netleştirilmesi adına bu sütun tamamen kaldırılmıştır.
# ================================================================================



# Profession control etme
print(df_cleaned["Profession"].value_counts())
#bu satır sılıncek cunku onemli degil
df_cleaned.drop("Profession", axis=1, inplace=True)

# ----------------------------------------------------------


# ===================== AKADEMİK BASKI VERİSİ TEMİZLİĞİ =====================
# Academic Pressure sütunu incelendiğinde bazı satırlarda "0.0" değeri olduğu görülmüştür.
# Ancak bu değer, öğrencinin hiç akademik baskı hissetmediğini ifade etse de,
# veri setinin yapısı ve diğer sütunlarla olan tutarlılığı açısından bu tür sıfır değerler güvenilir kabul edilmemiştir.

# Ayrıca sıfır değerlerin çoğunlukla yanlış giriş veya eksik beyan olabileceği düşünülerek,
# sadece pozitif (gerçek) akademik baskı seviyeleri analize dahil edilmiştir.

# Bu nedenle "0.0" olan satırlar veri setinden çıkarılmıştır.
# =========================================================================



# Academic Pressure control etme
print(df["Academic Pressure"].value_counts())
print(df_cleaned["Academic Pressure"].value_counts())
df_cleaned = df_cleaned[df_cleaned["Academic Pressure"] != 0.0]

# ----------------------------------------------------------


# ===================== İŞ BASKISI (WORK PRESSURE) SÜTUNUNUN KALDIRILMASI =====================
# Work Pressure sütunu incelendiğinde, tüm satırların değeri yalnızca 0.0 olduğu görülmüştür.
# Bu da bu sütunun veri setinde herhangi bir farklılık (varyans) içermediğini ve
# istatistiksel analiz için bilgi taşımadığını gösterir.

# Bu nedenle, anlamlı bir katkısı olmayan bu sütun analiz öncesinde veri setinden kaldırılmıştır.
# ================================================================================



# Work Pressure control etme
print(df_cleaned["Work Pressure"].value_counts())
# bu sutun 0 dan baska degere sah'p olmadigindan dolayi bu sutunu silicegim
df_cleaned.drop("Work Pressure", axis=1, inplace=True)

# ----------------------------------------------------------


# ===================== CGPA SÜTUNU TEMİZLİĞİ ve DÖNÜŞÜMÜ =====================
# CGPA sütunundaki bazı değerler metin olarak girilmiş ve ondalık ayırıcı olarak "," kullanılmıştır.
# Bu nedenle önce tüm virgüller noktaya (.) çevrilmiştir.
# Ardından sütun sayısal (float) veri tipine dönüştürülmüştür.

# Son olarak, analiz dışı bırakılması gereken aykırı (mantıksız) değerler filtrelenmiştir:
# Sadece 5.0 ile 10.0 arasında kalan CGPA değerleri tutulmuştur.
# Bu sınırlar, akademik sistemin doğal aralığını temsil eder.
# ================================================================================


# CGPA control etme
df_cleaned["CGPA"] = df_cleaned["CGPA"].str.replace(',', '.')

df_cleaned["CGPA"] = df_cleaned["CGPA"].astype(float)
df_cleaned = df_cleaned[(df_cleaned["CGPA"] >= 5.0) & (df_cleaned["CGPA"] <= 10.0)]
print(df_cleaned["CGPA"].value_counts().to_string())

# ----------------------------------------------------------


# ===================== STUDY SATISFACTION VERİSİ TEMİZLİĞİ =====================
# Study Satisfaction (Çalışma Memnuniyeti) sütununda bazı gözlemler "0.0" değerine sahiptir.
# Bu değerler ya eksik beyanlardan ya da hatalı veri girişinden kaynaklanabilir.
# Çünkü memnuniyet düzeyinin tamamen sıfır olması gerçek dışı veya anket dışı bir durumu işaret edebilir.

# Bu nedenle 0.0 olan satırlar analizden çıkarılmış ve yalnızca geçerli memnuniyet seviyeleri tutulmuştur.
# Bu işlem, istatistiksel analizlerin doğruluğunu ve anlamlılığını artırmak için yapılmıştır.
# ================================================================================


# Study Satisfaction control etme
df_cleaned = df_cleaned[df_cleaned["Study Satisfaction"] != 0.0]
print(df_cleaned["Study Satisfaction"].value_counts())

# ----------------------------------------------------------


# ===================== JOB SATISFACTION SÜTUNUNUN KALDIRILMASI =====================
# Job Satisfaction sütunu incelendiğinde, veri setindeki öğrencilerin büyük çoğunluğunun çalışmadığı
# veya bu sütunun ya sabit değere sahip olduğu ya da analizde anlamlı farklılık taşımadığı anlaşılmıştır.
# Ayrıca, bu sütun proje kapsamındaki ana araştırma konusuyla doğrudan ilişkili değildir.

# Bu nedenle, analizde gereksiz karmaşaya yol açmaması adına sütun tamamen veri setinden çıkarılmıştır.
# ================================================================================



# Job Satisfaction control etme
print(df_cleaned["Job Satisfaction"].value_counts())
df_cleaned.drop("Job Satisfaction", axis=1, inplace=True)

# ----------------------------------------------------------


# ===================== SLEEP DURATION VERİSİ TEMİZLİĞİ ve DÖNÜŞÜMÜ =====================
# "Sleep Duration" sütununda bazı gözlemler "Others" gibi analiz açısından belirsiz ve anlamsız kategoriler içermektedir.
# Bu nedenle, "Others" etiketi içeren satırlar veri setinden çıkarılmıştır.

# Kalan uyku süreleri belirli gruplara ayrılarak sayısal (numeric) değerlere dönüştürülmüştür:
# "Less than 5 hours" → 0, "5-6 hours" → 1, "7-8 hours" → 2, "More than 8 hours" → 3

# Bu dönüşüm sayesinde, uyku süresi değişkeni artık istatistiksel testlerde kullanılabilir hale gelmiştir.
# ================================================================================


# Sleep Duration sutunu temizleme
print(df_cleaned["Sleep Duration"].value_counts())
df_cleaned = df_cleaned[df_cleaned["Sleep Duration"] != "Others"]
sleep_map = {
    "Less than 5 hours": 0,
    "5-6 hours": 1,
    "7-8 hours": 2,
    "More than 8 hours": 3
}
df_cleaned["Sleep Duration"] = df_cleaned["Sleep Duration"].replace(sleep_map)
print(df_cleaned["Sleep Duration"].value_counts())

# ----------------------------------------------------------


# ===================== BESLENME ALIŞKANLIKLARI VERİSİ (DIETARY HABITS) =====================
# "Dietary Habits" sütununda analiz açısından anlam taşımayan "Others" gibi belirsiz kategoriler mevcuttu.
# Bu tür veriler, gruplar arasında net karşılaştırma yapmayı zorlaştırdığı için veri setinden çıkarılmıştır.

# Kalan değerler istatistiksel analizlerde kullanılabilmesi için sayısal olarak yeniden kodlanmıştır:
# "Unhealthy" → 0, "Moderate" → 1, "Healthy" → 2

# Bu işlem, beslenme alışkanlıklarının istatistiksel analizlere uygun hale getirilmesini sağlamıştır.
# ================================================================================



# Dietary Habits control etme
print(df_cleaned["Dietary Habits"].value_counts())
df_cleaned = df_cleaned[df_cleaned["Dietary Habits"] != "Others"]
diet_map = {
    "Unhealthy": 0,
    "Moderate": 1,
    "Healthy": 2
}
df_cleaned["Dietary Habits"] = df_cleaned["Dietary Habits"].replace(diet_map)

# ----------------------------------------------------------


# ===================== MEZUNİYET DERECESİ VERİSİ (Degree) =====================
# "Degree" sütununda bazı değerler "Others" gibi analiz açısından belirsiz kategoriler içermekteydi.
# Bu nedenle, "Others" etiketine sahip satırlar veri setinden çıkarılmıştır.

# Ardından, kalan tüm dereceler anlamlı 3 ana kategoriye indirgenmiştir:
# "Class 12" (lise düzeyi), "Undergraduate" (lisans), "Postgraduate" (yüksek lisans), "Doctorate" (doktora)

# Bu dönüşüm sayesinde, mezuniyet düzeyi analizleri daha sade, tutarlı ve anlamlı hale getirilmiştir.
# ================================================================================



# Degree control etme
print(df_cleaned["Degree"].value_counts())
df_cleaned = df_cleaned[df_cleaned["Degree"] != "Others"]
degree_map = {
    "Class 12": "Class 12",
    "B.Com": "Undergraduate", "B.Tech": "Undergraduate", "B.Arch": "Undergraduate",
    "BCA": "Undergraduate", "BHM": "Undergraduate","B.Ed": "Undergraduate", "BSc": "Undergraduate",
    "BBA": "Undergraduate", "BE": "Undergraduate", "BA": "Undergraduate","MBBS ": "Undergraduate",
    "B.Pharm": "Undergraduate", "LLB": "Undergraduate",
    
    "M.Tech": "Postgraduate", "MBA": "Postgraduate", "MSc": "Postgraduate",
    "MCA": "Postgraduate", "M.Ed": "Postgraduate", "M.Com": "Postgraduate",
    "MA": "Postgraduate", "M.Pharm": "Postgraduate", "LLM": "Postgraduate",
    "MHM": "Postgraduate", "ME": "Postgraduate", "MD": "Postgraduate",
    
    "PhD": "Doctorate"
}
df_cleaned["Degree"] = df_cleaned["Degree"].replace(degree_map)
degree_numeric_map = {
    "Class 12": 0,
    "Undergraduate": 1,
    "Postgraduate": 2,
    "Doctorate": 3
}
df_cleaned["Degree_Numeric"] = df_cleaned["Degree"].replace(degree_numeric_map)


# ----------------------------------------------------------


# ===================== İNTİHAR DÜŞÜNCESİ DEĞİŞKENİNİN DÖNÜŞÜMÜ =====================
# "Have you ever had suicidal thoughts?" sütunu kategorik (evet/hayır) veri içermektedir.
# İstatistiksel analizlerde bu değişkenin sayısal forma dönüştürülmesi gerekmektedir.

# Bu nedenle şu şekilde yeniden kodlanmıştır:
# "No" → 0, "Yes" → 1

# Bu dönüşüm sayesinde bu sütun artık korelasyon hesapları, hipotez testleri ve görselleştirmelerde rahatça kullanılabilir hale gelmiştir.
# ================================================================================



# Have you ever had suicidal thoughts ? control etme
print(df_cleaned["Have you ever had suicidal thoughts ?"].value_counts())
df_cleaned["Have you ever had suicidal thoughts ?"] = df_cleaned["Have you ever had suicidal thoughts ?"].replace({"No": 0, "Yes": 1})

# ----------------------------------------------------------


# ===================== GÜNLÜK ÇALIŞMA/ETÜT SÜRESİ İNCELEMESİ =====================
# "Work/Study Hours" sütunu, öğrencilerin günde kaç saat çalıştıklarını veya ders çalıştıklarını belirtmektedir.
# ================================================================================



# Work/Study Hours control etme
print(df_cleaned["Work/Study Hours"].value_counts())
print(df["Work/Study Hours"].value_counts())
# ----------------------------------------------------------


# ===================== MALİ STRES (FINANCIAL STRESS) VERİSİ DÖNÜŞÜMÜ =====================
# "Financial Stress" sütunu, öğrencilerin mali zorluk yaşayıp yaşamadıklarını belirtir.
# Sütun kategorik değerler (Yes/No) içerdiği için, sayısal analizlerde kullanılabilmesi amacıyla yeniden kodlanmıştır:

# "No" → 0, "Yes" → 1

# Bu dönüşüm, bu değişkenin korelasyon, hipotez testi ve grafik analizlerinde kullanılmasına olanak sağlar.
# ================================================================================



# Financial Stress control etme
print(df_cleaned["Financial Stress"].value_counts())
df_cleaned["Financial Stress"] = df_cleaned["Financial Stress"].replace({"No": 0, "Yes": 1})

# ----------------------------------------------------------


# ===================== AİLESEL RUHSAL HASTALIK GEÇMİŞİ VERİSİ DÖNÜŞÜMÜ =====================
# "Family History of Mental Illness" sütunu, öğrencilerin ailesinde ruhsal hastalık geçmişi olup olmadığını belirtir.
# Bu sütun Yes/No (Evet/Hayır) şeklinde kategorik veriler içerdiğinden, sayısal analizlerde kullanılabilmesi için yeniden kodlanmıştır:

# "No" → 0, "Yes" → 1

# Bu sayısal dönüşüm sayesinde değişken istatistiksel testlerde, korelasyon analizlerinde ve grafiklerde kullanılabilir hale gelmiştir.
# ================================================================================



# Family History of Mental Illness control etme
print(df_cleaned["Family History of Mental Illness"].value_counts())
df_cleaned["Family History of Mental Illness"] = df_cleaned["Family History of Mental Illness"].replace({"No": 0, "Yes": 1})

# ----------------------------------------------------------


# ===================== DEPRESYON VERİSİ KONTROLÜ ve SON KONTROL =====================
# "Depression" sütunu öğrencilerin depresyon durumu hakkında bilgi verir.
# Bu noktada sütundaki değerlerin dağılımı incelenmiştir (Yes/No veya 1/0 olabilir).

# Ardından veri kümesinin genel yapısı (info) ve boyutu (kaç satır/kaynak) yazdırılmıştır.
# Temizleme işlemlerinden önce ve sonra kaç satır kaldığı karşılaştırılmıştır.
# Bu adım, veri temizliği işlemlerinin veri seti üzerindeki etkisini gözlemlemek için yapılır.
# ================================================================================


# Depression control etme
print(df_cleaned["Depression"].value_counts())
data_loss = (1 - len(df_cleaned) / len(df)) * 100
print(f"Veri temizliği sonucu toplam {data_loss:.2f}% veri kaybedildi.")

print(df_cleaned.info())
print("Before:", df.shape)
print("After:", df_cleaned.shape)

#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ikinci bolum :Said Abou Allail\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
# ======================== İkinci Bölüm: İstatistiksel Hesaplamalar ve Görselleştirmeler ========================

# Sayısal değişkenleri liste halinde tanımladım
num_cols = ["Age", "CGPA", "Academic Pressure", "Study Satisfaction", "Work/Study Hours", "Financial Stress"]

# Her sayısal sütun için describe() fonksiyonuyla özet istatistikleri yazdırıyorum
print("\n=== CGPA İstatistikleri ===")
print(df_cleaned["CGPA"].describe())

print("\n=== Age İstatistikleri ===")
print(df_cleaned["Age"].describe())

print("\n=== Academic Pressure İstatistikleri ===")
print(df_cleaned["Academic Pressure"].describe())

print("\n=== Study Satisfaction İstatistikleri ===")
print(df_cleaned["Study Satisfaction"].describe())

print("\n=== Work/Study Hours İstatistikleri ===")
print(df_cleaned["Work/Study Hours"].describe())

print("\n=== Financial Stress İstatistikleri ===")
print(df_cleaned["Financial Stress"].describe())

# Sayısal değişkenler için hem histogram hem de boxplot grafiklerini yan yana çizdiriyorum
for col in num_cols:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    sns.histplot(df_cleaned[col], kde=True, ax=ax1, color='skyblue')
    sns.boxplot(x=df_cleaned[col], ax=ax2, color='lightblue')
    ax1.set_title(f"{col} Histogram")
    ax2.set_title(f"{col} Boxplot")
    plt.tight_layout()
    plt.show()

# Kategorik değişkenler için grafik çizdiriyorum
# Cinsiyet dağılımını gösteren grafik
plt.figure(figsize=(6,4))
sns.countplot(x="Gender", data=df_cleaned, palette="Set2")
plt.xticks([0, 1], ['Male', 'Female'])
plt.title("Cinsiyete Göre Öğrenci Dağılımı")
plt.xlabel("Cinsiyet")
plt.ylabel("Öğrenci Sayısı")
plt.show()

# En çok temsil edilen ilk 10 şehri gösteriyorum
top_cities = df_cleaned["City"].value_counts().head(10)
plt.figure(figsize=(10,6))
sns.barplot(x=top_cities.values, y=top_cities.index, palette="viridis")
plt.title("En Çok Temsil Edilen 10 Şehir")
plt.xlabel("Öğrenci Sayısı")
plt.ylabel("Şehir")
plt.show()

# Uyku süresi dağılım grafiğini çiziyorum
plt.figure(figsize=(8,5))
sns.countplot(x="Sleep Duration", data=df_cleaned, palette="Blues")
plt.xticks(ticks=[0,1,2,3], labels=['<5h', '5-6h', '7-8h', '>8h'])
plt.title("Uyku Süresi Dağılımı")
plt.xlabel("Uyku Süresi")
plt.ylabel("Öğrenci Sayısı")
plt.grid(True)
plt.show()

# Beslenme alışkanlıkları dağılımını gösteren grafik
plt.figure(figsize=(7,5))
sns.countplot(x="Dietary Habits", data=df_cleaned, palette="Greens")
plt.xticks(ticks=[0,1,2], labels=['Unhealthy', 'Moderate', 'Healthy'])
plt.title("Beslenme Alışkanlıklarının Dağılımı")
plt.xlabel("Beslenme Tipi")
plt.ylabel("Öğrenci Sayısı")
plt.grid(True)
plt.show()

# Eğitim derecesine göre öğrenci dağılımını çizdiriyorum
plt.figure(figsize=(7,5))
sns.countplot(x="Degree", data=df_cleaned, order=df_cleaned["Degree"].value_counts().index, palette="pastel")
plt.title("Öğrenci Dağılımı (Derecelere Göre)")
plt.xlabel("Derece")
plt.ylabel("Öğrenci Sayısı")
plt.grid(True)
plt.show()

# İntihar düşüncesi dağılımını ve yüzdesini gösteriyorum
plt.figure(figsize=(6,4))
sns.countplot(x="Have you ever had suicidal thoughts ?", data=df_cleaned, palette="Reds")
plt.xticks([0, 1], ['No', 'Yes'])
plt.title("İntihar Düşüncesi Dağılımı")
plt.xlabel("İntihar Düşüncesi")
plt.ylabel("Öğrenci Sayısı")
plt.grid(True)
plt.show()
percentage_yes = (df_cleaned["Have you ever had suicidal thoughts ?"].value_counts().loc[1] / df_cleaned["Have you ever had suicidal thoughts ?"].value_counts().sum()) * 100
print(f"Yüzde olarak 'Evet' diyen öğrenciler: {percentage_yes:.2f}%")

# Ailede ruhsal hastalık geçmişini gösteren grafik
plt.figure(figsize=(6,4))
sns.countplot(x="Family History of Mental Illness", data=df_cleaned, palette="coolwarm")
plt.xticks([0, 1], ['No', 'Yes'])
plt.title("Ailede Ruhsal Hastalık Geçmişi")
plt.xlabel("Geçmiş")
plt.ylabel("Öğrenci Sayısı")
plt.grid(True)
plt.show()
percentage_yes = (df_cleaned["Family History of Mental Illness"].value_counts().loc[1] / df_cleaned["Family History of Mental Illness"].value_counts().sum()) * 100
print(f"%{percentage_yes:.2f} öğrencinin ailesinde ruhsal hastalık geçmişi var.")

# Depresyon durumunu gösteren grafik
plt.figure(figsize=(6,4))
sns.countplot(x="Depression", data=df_cleaned, palette="magma")
plt.xticks([0,1], ['No', 'Yes'])
plt.title("Depresyon Durumu Dağılımı")
plt.xlabel("Depresyon")
plt.ylabel("Öğrenci Sayısı")
plt.grid(True)
plt.show()
percentage_depressed = (df_cleaned["Depression"].value_counts().loc[1.0] / df_cleaned["Depression"].value_counts().sum()) * 100
print(f"Depresyonda olduğunu belirten öğrencilerin oranı: %{percentage_depressed:.2f}")

# Nihai istatistiksel özet: describe() çıktısını ve gelişmiş istatistikleri birleştiriyorum
summary_df = df_cleaned[num_cols].describe()
advanced_stats = pd.DataFrame({
    col: {
        "variance": df_cleaned[col].var(),
        "skewness": df_cleaned[col].skew(),
        "kurtosis": df_cleaned[col].kurt(),
        "range": df_cleaned[col].max() - df_cleaned[col].min()
    } for col in num_cols
})
final_summary = pd.concat([summary_df, advanced_stats])
print("\n=== Nihai İstatistiksel Özet ===")
print(final_summary.T)

#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\üçüncü bölüm : Beyan Srouji\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\



# Gerekli kütüphaneler

import matplotlib.pyplot as plt

from scipy.stats import kstest, probplot


# Numerik (Sayısal) sütunlar listesi

num_cols = ["Age", "CGPA", "Academic Pressure", "Study Satisfaction", "Work/Study Hours", "Financial Stress"]

# Test sonuçları saklamak için boş liste
# Verilerin normal dağılıma uyup uymadığını kontrol etmek için Kolmogorov-Smirnov testi yapıyoruz
# Kolmogorov-Smirnov testini neden seçtik?
# Pek çok istatistiksel analiz, verilerin normal dağılıma uymasını varsayar.
# Eğer veriler normal dağılmıyorsa, bu testlerin sonuçları yanıltıcı olabilir.
# Bu nedenle, verilerimizin normal dağılıma uyup uymadığını kontrol etmek için Kolmogorov-Smirnov testini kullanıyoruz.
# Bu test, verimizin dağılımını beklenen normal dağılımla karşılaştırarak istatistiksel anlamlılık sağlar.

ks_results = []


# Her sütun için normal dağılım kontrolü yapılır

#  K-S testi için Ortalama ve Standart sapma 
for col in num_cols:
    mean = df_cleaned[col].mean()   #Ortalama
    std = df_cleaned[col].std()     #Standart sapma

# Kolmogorov–Smirnov Testi – analiz edilecek değişkenler
# Eğer p-değeri > 0.05 ise, veri normal dağılabilir olarak kabul edilir.
# Eğer p-değeri < 0.05 ise, veri normal dağılmıyor demektir.

    stat, p = kstest(df_cleaned[col], 'norm', args=(mean, std))
    ks_results.append({
        "Değişken": col,
        "p-değeri": round(p, 5),
        "Sonuç": "Normal dağılabilir" if p > 0.05 else "Normal dağılmıyor"
    })




# Şimdi Q-Q Plot çiziyoruz

# Q-Q Plot (Quantile-Quantile Grafiği), verilerin normal dağılıma ne kadar uyduğunu görsel olarak kontrol etmek için kullanılır.
# Q-Q Plot neden seçildi?
# Çünkü verinin normal dağılıma uyup uymadığını görsel olarak hızlı bir şekilde görmek için mükemmel bir araçtır.
# Eğer noktalar çizilen doğruya yakınsa, bu verilerin normal dağıldığını gösterir.
# Aksi takdirde, veri normal dağılımdan sapmış demektir.

 # Q-Q Plot: Görsel olarak normal dağılıma uygunluk kontrolü

    plt.figure(figsize=(6, 4))
    probplot(df_cleaned[col], dist="norm", plot=plt)
    plt.title(f"{col} - Q-Q Plot")
    plt.grid(True)
    plt.show()

# Sonuçları tablo halinde ( DataFrame ) olarak göster 
ks_df = pd.DataFrame(ks_results)
print(ks_df)  

# Skewness (Çarpıklık)  ve  Kurtosis (Basıklık): 
# Skewness, verinin dağılımının simetrikliğini ölçer. 
# Kurtosis ise dağılımın zirvesinin yüksekliğini ölçer.
# Her iki değerin yorumlanması önemlidir, çünkü normal dağılımda çarpıklık sıfır ve basıklık 3 olmalıdır.

    
for col in num_cols:
    skew = df_cleaned[col].skew()      # Çarpıklık değeri
    kurt = df_cleaned[col].kurt()      # Basıklık değeri
    print(f"{col} → Çarpıklık: {skew:.2f} | Basıklık: {kurt:.2f}")
    
   
    
   
    
# Gruplara göre CGPA  depresyon durumuna göre ortalamalarını hesapla
group_means = df_cleaned.groupby("Depression")["CGPA"].mean()
print("Depresyon durumuna göre CGPA ortalamaları:\n")
print(group_means)

# Gruplara göre daha detaylı istatistiksel özet (count, mean, std, min, max...)
group_summary = df_cleaned.groupby("Depression")["CGPA"].describe()
print("\nİstatistiksel özet:\n")
print(group_summary)
 
# Boxplot, verilerin dağılımını, çeyrek değerlerini, medyanı ve aykırı değerleri görsel olarak gösterir.
# Bu grafik, depresyon durumunun CGPA üzerindeki etkisini açıkça gösterir.

#  Boxplot ile CGPA dağılımı depresyon durumuna göre görselleştirilir
plt.figure(figsize=(6,4))
sns.boxplot(x="Depression", y="CGPA", data=df_cleaned, palette="Set2")
plt.xticks([0, 1], ["Depresyonda Değil", "Depresyonda"])
plt.title("Depresyon Durumuna Göre CGPA Dağılımı")
plt.xlabel("Depresyon")
plt.ylabel("CGPA")
plt.grid(True)
plt.show()
   
   
    
# 1. Uyku süresine göre CGPA ortalamaları

# Farklı uyku sürelerinin CGPA üzerindeki etkisini görselleştirmek için Boxplot kullanıyoruz.
# Uyku süresi ve CGPA arasındaki ilişkiyi görsel olarak incelemek için bu grafiği oluşturuyoruz.

group1 = df_cleaned.groupby("Sleep Duration")["CGPA"].mean()
print("Uyku Süresine Göre CGPA Ortalamaları:\n")
print(group1)

# Uyku süresine göre CGPA Boxplot

plt.figure(figsize=(7,5))
sns.boxplot(x="Sleep Duration", y="CGPA", data=df_cleaned, palette="pastel")
plt.xticks(ticks=[0,1,2,3], labels=['<5h', '5-6h', '7-8h', '>8h'])
plt.title("Uyku Süresi ve CGPA İlişkisi")
plt.xlabel("Uyku Süresi")
plt.ylabel("CGPA")
plt.grid(True)
plt.show()


#2. Akademik baskı – Beslenme Alışkanlıklarına göre karşılaştırma (Dietary Habits)

group2 = df_cleaned.groupby("Dietary Habits")["Academic Pressure"].mean()
print("Beslenme Alışkanlıklarına Göre Akademik Baskı Ortalamaları:\n")
print(group2)

# Boxplot: Akademik baskı vs. beslenme

plt.figure(figsize=(7,5))
sns.boxplot(x="Dietary Habits", y="Academic Pressure", data=df_cleaned, palette="Set3")
plt.xticks(ticks=[0,1,2], labels=["Unhealthy", "Moderate", "Healthy"])
plt.title("Beslenme ve Akademik Baskı İlişkisi")
plt.xlabel("Beslenme Alışkanlığı")
plt.ylabel("Akademik Baskı")
plt.grid(True)
plt.show()

#3. Dereceye göre çalışma memnuniyeti

group3 = df_cleaned.groupby("Degree")["Study Satisfaction"].mean()
print("Dereceye Göre Çalışma Memnuniyeti Ortalamaları:\n")
print(group3)

# Boxplot: Derece vs memnuniyet

plt.figure(figsize=(7,5))
sns.boxplot(x="Degree", y="Study Satisfaction", data=df_cleaned, palette="cool")
plt.title("Dereceye Göre Çalışma Memnuniyeti")
plt.xlabel("Eğitim Derecesi")
plt.ylabel("Çalışma Memnuniyeti")
plt.grid(True)
plt.show()

# 4. İntihar düşüncesine göre finansal stres

# İntihar düşüncesi olan ve olmayan bireylerin finansal stres ortalamalarını hesaplıyoruz.
# Bu analiz, intihar düşüncelerinin finansal stres üzerindeki etkilerini araştırmak amacıyla yapılır.


group4 = df_cleaned.groupby("Have you ever had suicidal thoughts ?")["Financial Stress"].mean()
print("İntihar Düşüncesine Göre Finansal Stres Ortalamaları:\n")
print(group4)


# Boxplot: İntihar düşüncesi ve stres

# İntihar düşüncesi olan ve olmayan kişilerin finansal stres düzeylerini görselleştirmek için Boxplot kullanıyoruz.
# Bu grafik, intihar düşüncesinin finansal stres üzerindeki etkilerini görsel olarak gösterir.


plt.figure(figsize=(6,4))
sns.boxplot(x="Have you ever had suicidal thoughts ?", y="Financial Stress", data=df_cleaned, palette="Reds")
plt.xticks([0,1], ["Hayır", "Evet"])
plt.title("İntihar Düşüncesi ve Finansal Stres İlişkisi")
plt.xlabel("İntihar Düşüncesi")
plt.ylabel("Finansal Stres")
plt.grid(True)
plt.show()

        
        
#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\dörtüncü bölüm : MAHER ABDULRAQEB ALI GHALEB\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\


# Hedef sütunlar
target_columns = [
    "CGPA",
    "Academic Pressure",
    "Study Satisfaction",
    "Work/Study Hours",
    "Financial Stress"
]

# Sonuçları saklamak için liste
bootstrap_results = []

# Bootstrap fonksiyonu
def bootstrap_ci(data, num_samples=10000, confidence=0.95):
    n = len(data)
    means = [np.mean(np.random.choice(data, size=n, replace=True)) for _ in range(num_samples)]
    lower = np.percentile(means, (1 - confidence) / 2 * 100)
    upper = np.percentile(means, (1 + confidence) / 2 * 100)
    return round(lower, 2), round(upper, 2)

# Her sütun için hesaplama
for col in target_columns:
    data = df_cleaned[col].dropna().values
    mean = round(np.mean(data), 2)
    ci_lower, ci_upper = bootstrap_ci(data)

    bootstrap_results.append({
        "Değişken": col,
        "Ortalama": mean,
        "%95 Alt Sınır": ci_lower,
        "%95 Üst Sınır": ci_upper
    })


bootstrap_df = pd.DataFrame(bootstrap_results)
print(bootstrap_df.to_string(index=False))

#----------------------------------------------------

# Hipotezler

alpha = 0.05
nonparametric_results = []

# Hipotez 1: Depresyon ve CGPA
group0 = df_cleaned[df_cleaned["Depression"] == 0]["CGPA"]
group1 = df_cleaned[df_cleaned["Depression"] == 1]["CGPA"]
stat, p_val = mannwhitneyu(group0, group1)
nonparametric_results.append({
    "Hipotez": "Depresyon ve CGPA",
    "Test Türü": "Mann-Whitney U",
    "İstatistik Değeri": format(stat, ".2f"),
    "p-değeri": "{:.8f}".format(p_val),
    "Karar": "Boş hipotezi reddet" if p_val < alpha else "Boş hipotez reddedilemez"
})

# Depresyonda olan öğrenciler ile olmayanlar arasında CGPA ortalamaları açısından istatistiksel olarak
# anlamlı fark vardır. Bu, depresyonun akademik başarı üzerinde etkili olabileceğini göstermektedir.

#-------------------------------------------------------

# Hipotez 2: Uyku Süresi ve CGPA
groups = [df_cleaned[df_cleaned["Sleep Duration"] == i]["CGPA"] for i in sorted(df_cleaned["Sleep Duration"].unique())]
stat, p_val = kruskal(*groups)
nonparametric_results.append({
    "Hipotez": "Uyku Süresi ve CGPA",
    "Test Türü": "Kruskal-Wallis",
    "İstatistik Değeri": format(stat, ".2f"),
    "p-değeri": "{:.8f}".format(p_val),
    "Karar": "Boş hipotezi reddet" if p_val < alpha else "Boş hipotez reddedilemez"
})

# Öğrencilerin uyku süresine göre CGPA ortalamaları arasında anlamlı fark bulunmaktadır. 
# Bu sonuç, düzenli ve yeterli uyku süresinin akademik başarıyı olumlu etkileyebileceğini göstermektedir.

#-----------------------------------------------------

# Hipotez 3: Yaş ve CGPA (Spearman Korelasyon)
corr, p_val = spearmanr(df_cleaned["Age"], df_cleaned["CGPA"])
nonparametric_results.append({
    "Hipotez": "Yaş ve CGPA",
    "Test Türü": "Spearman Korelasyon",
    "İstatistik Değeri": format(corr, ".4f"),
    "p-değeri": "{:.8f}".format(p_val),
    "Karar": "Boş hipotezi reddet" if p_val < alpha else "Boş hipotez reddedilemez"
})

# Öğrencilerin yaşı ile CGPA ortalamaları arasında anlamlı bir ilişki bulunamamıştır. 
# Yaş değişkeninin akademik başarı üzerinde belirleyici bir etkisi gözlemlenmemiştir.



nonparametric_df = pd.DataFrame(nonparametric_results)
print(nonparametric_df.to_string(index=False))

# ----------------------------------------------------------

# Gözlem Oranları

# Oranlarını hesaplamak istediğimiz sütunların listesi
columns_to_analyze = [
    "Depression",
    "Have you ever had suicidal thoughts ?",
    "Financial Stress",
    "Dietary Habits",
    "Sleep Duration",
    "Degree"
]

for col in columns_to_analyze:
    print(f"\n=== {col} ===")
    
    ratios = df_cleaned[col].value_counts(normalize=True, dropna=False) * 100
    ratios = ratios.round(2)
    
# index'leri sayısal değere çevir ve sırala
    try:
        ratios.index = ratios.index.astype(float)
        ratios = ratios.sort_index()
    except:
        ratios = ratios.sort_index()  
    ratios = ratios.astype(str) + " %"
    print(ratios.to_string())


    