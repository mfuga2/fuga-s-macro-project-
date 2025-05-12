import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas_datareader as pdr
import numpy as np

# set the start and end dates for the data
start_date = '1955-01-01'
end_date = '2022-01-01'

# download the data from FRED using pandas_datareader
gdp = web.DataReader('NGDPRSAXDCGBQ', 'fred', start_date, end_date)
log_gdp = np.log(gdp)



# apply a Hodrick-Prescott filter to the data to extract the cyclical component
cycle10, trend = sm.tsa.filters.hpfilter(log_gdp, lamb=10)
cycle100, trend = sm.tsa.filters.hpfilter(log_gdp, lamb=100)
cycle1600, trend = sm.tsa.filters.hpfilter(log_gdp, lamb=1600)

# Plot the trend component
plt.plot(cycle10, label="Cycle (λ=10)")
plt.plot(cycle100, label="Cycle (λ=100)")
plt.plot(cycle1600, label="Cycle (λ=1600)")

plt.xlabel("Year")
plt.ylabel("Cyclical component (Log GDP)")

plt.grid(True)
# Add a legend and show the plot
plt.legend()
plt.show()
