import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas_datareader as pdr
import numpy as np

# set the start and end dates for the data
start_date = '1995-01-01'
end_date = '2024-01-01'

# download the data from FRED using pandas_datareader
gdpJP = web.DataReader('JPNRGDPEXP', 'fred', start_date, end_date)
gdpUK = web.DataReader('NGDPRSAXDCGBQ', 'fred', start_date, end_date)
log_gdpJP = np.log(gdpJP)
log_gdpUK = np.log(gdpUK)
# apply a Hodrick-Prescott filter to the data to extract the cyclical component
JPcycle, trend = sm.tsa.filters.hpfilter(log_gdpJP, lamb=1600)
UKcycle, trend = sm.tsa.filters.hpfilter(log_gdpUK, lamb=1600)

std_jp = JPcycle.std()
std_uk = UKcycle.std()

print(f"日本の循環変動成分の標準偏差: {std_jp:.4f}")
print(f"イギリスの循環変動成分の標準偏差: {std_uk:.4f}")

combined_cycle = pd.concat([JPcycle, UKcycle], axis=1).dropna()
combined_cycle.columns = ['JP', 'UK']

correlation = combined_cycle.corr().iloc[0,1]

print(f"日本とイギリスの循環成分の相関係数: {correlation:.4f}")

plt.figure(figsize=(12, 6))
plt.plot(combined_cycle.index, combined_cycle['JP'], label='Japan')
plt.plot(combined_cycle.index, combined_cycle['UK'], label='UK')
plt.axhline(0, linestyle='--', linewidth=0.8)
plt.xlabel("Year")
plt.ylabel("Cyclical Component (Log GDP)")
plt.title('Cyclical Components of Real GDP (JP vs UK)')
plt.grid(True)
plt.legend()
plt.show()
plt.tight_layout()
