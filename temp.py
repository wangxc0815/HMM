from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

file_name = r"C:\Users\Xiaochun.wang\Desktop\Data\Trend Following Data.xlsx"
rfr = 'H15T3M Index'

data_daily = pd.read_excel(file_name, "daily")
df_daily = data_daily[['Dates', rfr]].copy()
df_daily['Dates'] = pd.to_datetime(df_daily['Dates'])
df_daily.set_index("Dates", inplace=True)
df_daily_to_monthly = df_daily.resample('W').last()

df_daily_to_monthly.to_excel('H15T3M (weekly).xlsx')
