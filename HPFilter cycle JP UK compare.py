import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas_datareader as pdr
import numpy as np

# set the start and end dates for the data
start_date = '1995-01-01'
end_date = '2025-01-01'

# download the data from FRED using pandas_datareader
gdpJP = web.DataReader('JPNRGDPEXP', 'fred', start_date, end_date)
gdpUK = web.DataReader('NGDPRSAXDCGBQ', 'fred', start_date, end_date)
log_gdpJP = np.log(gdpJP)
log_gdpUK = np.log(gdpUK)
# apply a Hodrick-Prescott filter to the data to extract the cyclical component
JPcycle1600, trend = sm.tsa.filters.hpfilter(log_gdpJP, lamb=1600)
UKcycle1600, trend = sm.tsa.filters.hpfilter(log_gdpUK, lamb=1600)

# Plot the trend component

plt.plot(JPcycle1600, label="JPCycle")
plt.plot(UKcycle1600, label="UKCycle")

plt.xlabel("Year")
plt.ylabel("Cyclical component (Log GDP)")
plt.title('Cyclical compornent')
plt.grid(True)
# Add a legend and show the plot
plt.legend()
plt.show()
