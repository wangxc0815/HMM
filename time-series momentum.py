from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


""" 
    Enter Inputs Here 
"""
file_name = r"C:\Users\Xiaochun.wang\Desktop\Data\Trend Following Data.xlsx"
index = 'SPX Index'
rfr = 'H15T3M Index'

start_date = datetime(1938, 12, 1)
end_date = datetime(2023, 6, 30)


""" 
    Import Dataset (Monthly Frequency) 
"""
# Import dataset
data_monthly = pd.read_excel(file_name, "monthly")
df_monthly = data_monthly[['Dates', index, rfr]].copy()
df_monthly.set_index("Dates", inplace=True)
# Change the last trading day of each month to the last calendar day of the month in order to make the dates consistent
df_monthly.index = df_monthly.index.to_period('M').to_timestamp('M')
df_monthly = df_monthly.loc[(df_monthly.index >= start_date) & (df_monthly.index <= end_date)]

""" Monthly Returns and Log Returns """
# Compute monthly returns of the underlying asset and 3M US Treasury
df_monthly['ret'] = df_monthly[index].pct_change(1)
df_monthly['log ret'] = np.log(1 + df_monthly['ret'])
df_monthly['risk free ret'] = (df_monthly[rfr] / 100 + 1) ** (1/12) - 1
# Fill up the unavailable 3M US Treasury returns with the historical average
df_monthly['risk free ret'] = df_monthly['risk free ret'].fillna(df_monthly['risk free ret'].mean())
df_monthly['risk free log ret'] = np.log(1 + df_monthly['risk free ret'])

df_monthly['cum log ret'] = df_monthly['log ret'].cumsum()

df_monthly['dollar growth'] = (1 + df_monthly['ret']).cumprod()
df_monthly['dd'] = 1 - df_monthly['dollar growth'] / df_monthly['dollar growth'].cummax()


# Compute the arithmetic average of trailing n-month return
df_monthly['2M avg ret'] = df_monthly['ret'].rolling(2).mean()
df_monthly['3M avg ret'] = df_monthly['ret'].rolling(3).mean()
df_monthly['6M avg ret'] = df_monthly['ret'].rolling(6).mean()
df_monthly['9M avg ret'] = df_monthly['ret'].rolling(9).mean()
df_monthly['12M avg ret'] = df_monthly['ret'].rolling(12).mean()

# Compute the ewma volatility of S&P 500
#df_monthly['vol ewma'] = df_monthly['ret'].shift(1).ewm(halflife=monthly_ewma_hl, ignore_na=True).std()

""" Monthly Excess Returns """
# Compute the excess return of S&P 500
df_monthly['excess ret'] = df_monthly['ret'] - df_monthly['risk free ret']

# Compute the arithmetic average of trailing n-month excess return
df_monthly['2M avg excess ret'] = df_monthly['excess ret'].rolling(2).mean()
df_monthly['3M avg excess ret'] = df_monthly['excess ret'].rolling(3).mean()
df_monthly['6M avg excess ret'] = df_monthly['excess ret'].rolling(6).mean()
df_monthly['9M avg excess ret'] = df_monthly['excess ret'].rolling(9).mean()
df_monthly['12M avg excess ret'] = df_monthly['excess ret'].rolling(12).mean()

"""
    Import Dataset (Daily Frequency)
"""
# Compute annualized 21-day volatilities
data_daily = pd.read_excel(file_name, "daily")
df_daily = data_daily[['Dates', index, rfr]].copy()
df_daily['Dates'] = pd.to_datetime(df_daily['Dates'])
df_daily.set_index("Dates", inplace=True)
df_daily = df_daily.loc[(df_daily.index >= start_date) & (df_daily.index <= end_date)]
df_daily['ret'] = df_daily[index].pct_change(1)   # daily returns
df_daily['risk free ret'] = (df_daily[rfr] / 100 + 1) ** (1 / 261) - 1
df_daily['risk free ret'] = df_daily['risk free ret'].fillna(df_daily['risk free ret'].mean())
df_daily['excess ret'] = df_daily['ret'] - df_daily['risk free ret']
#df_daily['var'] = df_daily['excess ret'].shift(1).ewm(halflife=60).var() * 261
df_daily['vol'] = df_daily['excess ret'].ewm(halflife=60).std() * np.sqrt(261)
df_daily_to_monthly = df_daily.resample('M').last()

# Merge the annualized variance to monthly data
df_monthly = pd.merge(df_monthly, df_daily_to_monthly[['vol']], left_index=True, right_index=True)


"""
    Construct Time Series Momentum Strategy 

"""

df_monthly.loc[df_monthly['ret'].shift(1) >= 0, '1M tsmom sign'] = 1
df_monthly.loc[df_monthly['ret'].shift(1) < 0, '1M tsmom sign'] = -1

df_monthly.loc[df_monthly['2M avg ret'].shift(1) >= 0, '2M tsmom sign'] = 1
df_monthly.loc[df_monthly['2M avg ret'].shift(1) < 0, '2M tsmom sign'] = -1

df_monthly.loc[df_monthly['3M avg ret'].shift(1) >= 0, '3M tsmom sign'] = 1
df_monthly.loc[df_monthly['3M avg ret'].shift(1) < 0, '3M tsmom sign'] = -1

df_monthly.loc[df_monthly['6M avg ret'].shift(1) >= 0, '6M tsmom sign'] = 1
df_monthly.loc[df_monthly['6M avg ret'].shift(1) < 0, '6M tsmom sign'] = -1

df_monthly.loc[df_monthly['9M avg ret'].shift(1) >= 0, '9M tsmom sign'] = 1
df_monthly.loc[df_monthly['9M avg ret'].shift(1) < 0, '9M tsmom sign'] = -1

df_monthly.loc[df_monthly['12M avg ret'].shift(1) >= 0, '12M tsmom sign'] = 1
df_monthly.loc[df_monthly['12M avg ret'].shift(1) < 0, '12M tsmom sign'] = -1

""" Monthly Returns of the Time-Series Momentum Strategy """
df_monthly['1M tsmom ret'] = df_monthly['1M tsmom sign'] * 0.4 / df_monthly['vol'].shift(1) * df_monthly['ret']
df_monthly['2M tsmom ret'] = df_monthly['2M tsmom sign'] * 0.4 / df_monthly['vol'].shift(1) * df_monthly['ret']
df_monthly['3M tsmom ret'] = df_monthly['3M tsmom sign'] * 0.4 / df_monthly['vol'].shift(1) * df_monthly['ret']
df_monthly['6M tsmom ret'] = df_monthly['6M tsmom sign'] * 0.4 / df_monthly['vol'].shift(1) * df_monthly['ret']
df_monthly['9M tsmom ret'] = df_monthly['9M tsmom sign'] * 0.4 / df_monthly['vol'].shift(1) * df_monthly['ret']
df_monthly['12M tsmom ret'] = df_monthly['12M tsmom sign'] * 0.4 / df_monthly['vol'].shift(1) * df_monthly['ret']

""" Monthly Log Returns of Time-Series Momentum Strategy """
df_monthly['1M tsmom log ret'] = np.log(1 + df_monthly['1M tsmom ret'])
df_monthly['2M tsmom log ret'] = np.log(1 + df_monthly['2M tsmom ret'])
df_monthly['3M tsmom log ret'] = np.log(1 + df_monthly['3M tsmom ret'])
df_monthly['6M tsmom log ret'] = np.log(1 + df_monthly['6M tsmom ret'])
df_monthly['9M tsmom log ret'] = np.log(1 + df_monthly['9M tsmom ret'])
df_monthly['12M tsmom log ret'] = np.log(1 + df_monthly['12M tsmom ret'])

df_monthly = df_monthly.drop(columns=[index, rfr])
df_monthly = df_monthly.dropna()


""" Cumulative Log Returns of Time-Series Momentum Strategy """
df_monthly['1M tsmom cum log ret'] = df_monthly['1M tsmom log ret'].cumsum()
df_monthly['2M tsmom cum log ret'] = df_monthly['2M tsmom log ret'].cumsum()
df_monthly['3M tsmom cum log ret'] = df_monthly['3M tsmom log ret'].cumsum()
df_monthly['6M tsmom cum log ret'] = df_monthly['6M tsmom log ret'].cumsum()
df_monthly['9M tsmom cum log ret'] = df_monthly['9M tsmom log ret'].cumsum()
df_monthly['12M tsmom cum log ret'] = df_monthly['12M tsmom log ret'].cumsum()

""" Drawdowns of Time-Series Momentum Strategy """
df_monthly['1M tsmom dollar growth'] = (1 + df_monthly['1M tsmom ret']).cumprod()
df_monthly['1M tsmom dd'] = 1 - df_monthly['1M tsmom dollar growth'] / df_monthly['1M tsmom dollar growth'].cummax()

df_monthly['2M tsmom dollar growth'] = (1 + df_monthly['2M tsmom ret']).cumprod()
df_monthly['2M tsmom dd'] = 1 - df_monthly['2M tsmom dollar growth'] / df_monthly['2M tsmom dollar growth'].cummax()

df_monthly['3M tsmom dollar growth'] = (1 + df_monthly['3M tsmom ret']).cumprod()
df_monthly['3M tsmom dd'] = 1 - df_monthly['3M tsmom dollar growth'] / df_monthly['3M tsmom dollar growth'].cummax()

df_monthly['6M tsmom dollar growth'] = (1 + df_monthly['6M tsmom ret']).cumprod()
df_monthly['6M tsmom dd'] = 1 - df_monthly['6M tsmom dollar growth'] / df_monthly['6M tsmom dollar growth'].cummax()

df_monthly['9M tsmom dollar growth'] = (1 + df_monthly['9M tsmom ret']).cumprod()
df_monthly['9M tsmom dd'] = 1 - df_monthly['9M tsmom dollar growth'] / df_monthly['9M tsmom dollar growth'].cummax()

df_monthly['12M tsmom dollar growth'] = (1 + df_monthly['12M tsmom ret']).cumprod()
df_monthly['12M tsmom dd'] = 1 - df_monthly['12M tsmom dollar growth'] / df_monthly['12M tsmom dollar growth'].cummax()

df_monthly['tsmom ret'].mean() / df_monthly['tsmom ret'].std() * np.sqrt(12)



tsmom_ret = df_monthly[['ret', '1M tsmom ret', '2M tsmom ret', '3M tsmom ret', '6M tsmom ret', '9M tsmom ret', '12M tsmom ret']].copy()

tsmom_stat = pd.concat([tsmom_ret.mean() * 12,
                        tsmom_ret.std() * np.sqrt(12),
                        tsmom_ret.mean() / tsmom_ret.std() * np.sqrt(12)], axis=1)

tsmom_stat.columns = ['Average Return (Annualized)', 'Volatility (Annualized)', 'Sharpe Ratio']

tsmom_stat['Cumulative Log Return'] \
    = [df_monthly['cum log ret'][-1], df_monthly['1M tsmom cum log ret'][-1], df_monthly['2M tsmom cum log ret'][-1],
       df_monthly['3M tsmom cum log ret'][-1], df_monthly['6M tsmom cum log ret'][-1], df_monthly['9M tsmom cum log ret'][-1],
       df_monthly['12M tsmom cum log ret'][-1]]

tsmom_stat['Maximum Drawdown'] \
    = [df_monthly['dd'].max(), df_monthly['1M tsmom dd'].max(), df_monthly['2M tsmom dd'].max(),
       df_monthly['3M tsmom dd'].max(), df_monthly['6M tsmom dd'].max(), df_monthly['9M tsmom dd'].max(),
       df_monthly['12M tsmom dd'].max()]

tsmom_stat.index = ['S&P500', '1M', '2M', '3M', '6M', '9M', '12M']
