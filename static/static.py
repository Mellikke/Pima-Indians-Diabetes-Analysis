import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np 
from scipy import stats
from scipy.stats import norm
from scipy import stats
data = pd.read_csv("pima_diabetes_data.csv")

def ortalamaBul(x):
    veriAdedi=len(x)#Gelen verinin uzunluğunu alır
    if veriAdedi<=1:#Eğer tek bir veri varsa
        return veri #ortalama olarak onu döndürür
    else:           #Birden fazla veri varsa
        return sum(x)/veriAdedi #verilerin toplamını verilerin uzunluğuna bölerek ortalamayı hesaplar

def medyan(veri):
    veri = sorted(veri) # Veriyi sıralıyoruz 
    n = len(veri)# Verinin uzunluğunu alıyoruz
    
    if n == 0: # Eğer veri boşsa, medyan hesaplanamaz, None döndürülür
        return None  
    
    if n % 2 == 1:# Veri uzunluğu tek sayı ise
        return veri[n // 2]# Ortadaki eleman medyan olur
    else:# Veri uzunluğu çift sayı ise, ortadaki iki elemanın ortalaması medyan olur
        orta1 = veri[n // 2 - 1] # Ortadaki ilk eleman
        orta2 = veri[n // 2]     # Ortadaki ikinci eleman
        return (orta1 + orta2) / 2 # İki elemanın ortalamasını alıyoruz

def calculate_variance(value):
    mean = ortalamaBul(value) #ortalaması diğer fonksiyondan direkt hesaplıyoruz
    squared_diff = [(x - mean) ** 2 for x in value]  #ilgili değerleri varyans formülüne göre hesaplıyoruz
    return sum(squared_diff) / (len(value) - 1)       #ilgili değerleri varyans formülüne göre hesaplıyoruz ve çıkan sonucu döndürüyoruz

# Standart Sapma (Standard Deviation) Hesaplama
def calculate_std_dev(value):
    variance = calculate_variance(value) #varyansını alıyoruz
    return variance ** 0.5               #varyansın karekökü bize standart sapmayı verir


# Standart Hata (Standard Error) Hesaplama
def calculate_std_err(value):
    std_dev = calculate_std_dev(value)
    return std_dev / (len(value)**0.5)# Standart sapma, örneklem büyüklüğünün kareköküne bölünerek standart hata bulunur.




BMI=data["BMI"].dropna()
n=len(BMI)
mean=ortalamaBul(BMI)
var=calculate_variance(BMI)
se=calculate_std_err(BMI)
 #sum_BMI=sum(BMI)
#avarage_BMI=sum_BMI/len(BMI)
#median=BMI.median()
#variance = BMI.var()
#std_dev = BMI.std()
 
#std_err = stats.sem(BMI)

# %95 güven düzeyi için t değeri
t_crit = stats.t.ppf(1 - 0.05/2, df=n - 1)

# Ortalama için güven aralığı
mean_ci = (mean - t_crit * se, mean + t_crit * se)

# Varyans için %95 güven aralığı (ki-kare dağılımı)
chi2_lower = stats.chi2.ppf(0.025, df=n - 1)
chi2_upper = stats.chi2.ppf(0.975, df=n - 1)
var_ci = ((n - 1) * var / chi2_upper, (n - 1) * var / chi2_lower)

print("Ortalama için %95 Güven Araliği:", mean_ci)
print("Varyans için %95 Güven Araliği:", var_ci)

std = BMI.std(ddof=1)
E = 0.1                             # Maksimum hata payı
confidence_level = 0.90            # %90 güven düzeyi
z = norm.ppf(1 - (1 - confidence_level) / 2)  # z-değeri

# Minimum örneklem büyüklüğü hesaplama
n_required = (z * std / E) ** 2

print("Gerekli minimum örneklem sayisi:", round(n_required))


bmi_diyabetsiz = data[data['Outcome'] == 0]['BMI']
bmi_diyabetli = data[data['Outcome'] == 1]['BMI']

# İki bağımsız örneklem için t-testi
t_stat, p_value = stats.ttest_ind(bmi_diyabetsiz, bmi_diyabetli, equal_var=False)  # Welch's t-test

print("t-istatistiği:", t_stat)
print("p-değeri:", p_value)

alpha = 0.05
decision = "Reject H₀" if p_value < alpha else "Fail to reject H₀"

# plt.boxplot(BMI)
# plt.title('Boxplot of BMI')
# plt.ylabel('BMI')
# plt.show()


# plt.hist(BMI, bins=5, edgecolor='black')  # bins: sütun sayısı
# plt.title('Histogram of BMI')
# plt.xlabel('BMI Değeri')
# plt.ylabel('Frekans')
# plt.show()

# plt.figure(figsize=(8, 5))
# plt.hist(BMI, bins=10, edgecolor='black')
# plt.title("Histogram of BMI")
# plt.xlabel("BMI Değeri")
# plt.ylabel("Frekans")
# plt.grid(True)
# plt.tight_layout()
# plt.show()

print(ortalamaBul(BMI))
#print(sum_BMI)
#print(avarage_BMI)
#print(median)
print(medyan(BMI))
#print(calculate_median(BMI))
#print(variance)
print(calculate_variance(BMI))
#print(std_dev)
print(calculate_std_dev(BMI))
#print(std_err)
print(calculate_std_err(BMI))