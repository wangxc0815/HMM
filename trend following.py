from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.copy_on_write = True

""" 
    Enter Inputs Here 
"""
file_name = r"C:\Users\Xiaochun.wang\Desktop\Data\Trend Following Data.xlsx"
index = 'SPX Index'
rfr = 'H15T3M Index'

start_date = datetime(1938, 12, 1)
end_date = datetime(2023, 12, 31)


""" 
    Import Dataset (Monthly Frequency) 
"""
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

# Compute the arithmetic average of trailing n-month return
df_monthly['2M avg ret'] = df_monthly['ret'].rolling(2).mean()
df_monthly['3M avg ret'] = df_monthly['ret'].rolling(3).mean()
df_monthly['6M avg ret'] = df_monthly['ret'].rolling(6).mean()
df_monthly['9M avg ret'] = df_monthly['ret'].rolling(9).mean()
df_monthly['12M avg ret'] = df_monthly['ret'].rolling(12).mean()

""" Monthly Excess Returns """
# Compute the excess return of S&P 500
df_monthly['excess ret'] = df_monthly['ret'] - df_monthly['risk free ret']

# Compute the arithmetic average of trailing n-month excess return
df_monthly['2M avg excess ret'] = df_monthly['excess ret'].rolling(2).mean()
df_monthly['3M avg excess ret'] = df_monthly['excess ret'].rolling(3).mean()
df_monthly['6M avg excess ret'] = df_monthly['excess ret'].rolling(6).mean()
df_monthly['9M avg excess ret'] = df_monthly['excess ret'].rolling(9).mean()
df_monthly['12M avg excess ret'] = df_monthly['excess ret'].rolling(12).mean()

""" Monthly Treasury Rates """
df_monthly_macro = data_monthly[['Dates', 'H15T1Y Index', 'H15T10Y Index', 'H15T20Y Index', 'CPI YOY Index', 'CSI BARC Index']].copy()
df_monthly_macro.set_index("Dates", inplace=True)
df_monthly_macro.index = df_monthly_macro.index.to_period('M').to_timestamp('M')
df_monthly_macro['1y tr'] = df_monthly_macro['H15T1Y Index'] / 100
df_monthly_macro['10y tr'] = df_monthly_macro['H15T10Y Index'] / 100
df_monthly_macro['20y tr'] = df_monthly_macro['H15T20Y Index'] / 100
df_monthly_macro['1y 20y spread'] = df_monthly_macro['20y tr'] - df_monthly_macro['1y tr']
df_monthly_macro['cpi'] = df_monthly_macro['CPI YOY Index'] / 100
df_monthly_macro['credit spread'] = df_monthly_macro['CSI BARC Index'] / 100

"""
    Import Dataset (Daily Frequency)
"""
data_daily = pd.read_excel(file_name, "daily")
df_daily = data_daily[['Dates', index, rfr]].copy()
df_daily['Dates'] = pd.to_datetime(df_daily['Dates'])
df_daily.set_index("Dates", inplace=True)
df_daily = df_daily.loc[(df_daily.index >= start_date) & (df_daily.index <= end_date)]
df_daily['ret'] = df_daily[index].pct_change(1)   # daily returns
df_daily['risk free ret'] = (df_daily[rfr] / 100 + 1) ** (1 / 261) - 1
df_daily['risk free ret'] = df_daily['risk free ret'].fillna(df_daily['risk free ret'].mean())
df_daily['excess ret'] = df_daily['ret'] - df_daily['risk free ret']
# Compute the annualized EWMA volatility of the daily excess returns
df_daily['vol'] = df_daily['excess ret'].ewm(halflife=60, ignore_na=True).std() * np.sqrt(261)
df_daily_to_monthly = df_daily.resample('M').last()

# Merge the annualized volatility to monthly data
df_monthly = pd.merge(df_monthly, df_daily_to_monthly[['vol']], left_index=True, right_index=True)

# Merge the macro regime indicators to monthly data
df_monthly = pd.merge(df_monthly, df_monthly_macro[['1y tr', '1y 20y spread', 'cpi', 'credit spread']], left_index=True, right_index=True)


""" 
    Construct Trend Following Strategy Signals
        If the trailing N-month excess return (arithmetic average monthly excess return) is nonnegative,
        then we go to long one unit in the subsequent month, otherwise, we go to long US treasury.
        
"""
df_monthly.loc[df_monthly['excess ret'].shift(1) >= 0, '1M pos'] = 1
df_monthly.loc[df_monthly['excess ret'].shift(1) < 0, '1M pos'] = 0

df_monthly.loc[df_monthly['2M avg excess ret'].shift(1) >= 0, '2M pos'] = 1
df_monthly.loc[df_monthly['2M avg excess ret'].shift(1) < 0, '2M pos'] = 0

df_monthly.loc[df_monthly['3M avg excess ret'].shift(1) >= 0, '3M pos'] = 1
df_monthly.loc[df_monthly['3M avg excess ret'].shift(1) < 0, '3M pos'] = 0

df_monthly.loc[df_monthly['6M avg excess ret'].shift(1) >= 0, '6M pos'] = 1
df_monthly.loc[df_monthly['6M avg excess ret'].shift(1) < 0, '6M pos'] = 0

df_monthly.loc[df_monthly['9M avg excess ret'].shift(1) >= 0, '9M pos'] = 1
df_monthly.loc[df_monthly['9M avg excess ret'].shift(1) < 0, '9M pos'] = 0

df_monthly.loc[df_monthly['12M avg excess ret'].shift(1) >= 0, '12M pos'] = 1
df_monthly.loc[df_monthly['12M avg excess ret'].shift(1) < 0, '12M pos'] = 0

df_monthly = df_monthly.drop(columns=[index, rfr])
#df_monthly = df_monthly.dropna()
df_monthly = df_monthly.loc[df_monthly.index >= datetime(1940,1,1)]

""" Turnover of Trending Following Strategy """
first_date = df_monthly.index[0]
df_monthly.loc[df_monthly['1M pos'] != df_monthly['1M pos'].shift(1), '1M trade'] = 1
df_monthly.loc[df_monthly['1M pos'] == df_monthly['1M pos'].shift(1), '1M trade'] = 0
df_monthly.loc[first_date, '1M trade'] = 0 if df_monthly['1M pos'][first_date] == 0 else 1

df_monthly.loc[df_monthly['2M pos'] != df_monthly['2M pos'].shift(1), '2M trade'] = 1
df_monthly.loc[df_monthly['2M pos'] == df_monthly['2M pos'].shift(1), '2M trade'] = 0
df_monthly.loc[first_date, '2M trade'] = 0 if df_monthly['2M pos'][first_date] == 0 else 1

df_monthly.loc[df_monthly['3M pos'] != df_monthly['3M pos'].shift(1), '3M trade'] = 1
df_monthly.loc[df_monthly['3M pos'] == df_monthly['3M pos'].shift(1), '3M trade'] = 0
df_monthly.loc[first_date, '3M trade'] = 0 if df_monthly['3M pos'][first_date] == 0 else 1

df_monthly.loc[df_monthly['6M pos'] != df_monthly['6M pos'].shift(1), '6M trade'] = 1
df_monthly.loc[df_monthly['6M pos'] == df_monthly['6M pos'].shift(1), '6M trade'] = 0
df_monthly.loc[first_date, '6M trade'] = 0 if df_monthly['6M pos'][first_date] == 0 else 1

df_monthly.loc[df_monthly['9M pos'] != df_monthly['9M pos'].shift(1), '9M trade'] = 1
df_monthly.loc[df_monthly['9M pos'] == df_monthly['9M pos'].shift(1), '9M trade'] = 0
df_monthly.loc[first_date, '9M trade'] = 0 if df_monthly['9M pos'][first_date] == 0 else 1

df_monthly.loc[df_monthly['12M pos'] != df_monthly['12M pos'].shift(1), '12M trade'] = 1
df_monthly.loc[df_monthly['12M pos'] == df_monthly['12M pos'].shift(1), '12M trade'] = 0
df_monthly.loc[first_date, '12M trade'] = 0 if df_monthly['12M pos'][first_date] == 0 else 1

""" Monthly Returns of Trending Following Strategy """
df_monthly.loc[df_monthly['1M pos'] == 1, '1M adj ret'] = df_monthly['ret']
df_monthly.loc[df_monthly['1M pos'] == 0, '1M adj ret'] = df_monthly['risk free ret']

df_monthly.loc[df_monthly['2M pos'] == 1, '2M adj ret'] = df_monthly['ret']
df_monthly.loc[df_monthly['2M pos'] == 0, '2M adj ret'] = df_monthly['risk free ret']

df_monthly.loc[df_monthly['3M pos'] == 1, '3M adj ret'] = df_monthly['ret']
df_monthly.loc[df_monthly['3M pos'] == 0, '3M adj ret'] = df_monthly['risk free ret']

df_monthly.loc[df_monthly['6M pos'] == 1, '6M adj ret'] = df_monthly['ret']
df_monthly.loc[df_monthly['6M pos'] == 0, '6M adj ret'] = df_monthly['risk free ret']

df_monthly.loc[df_monthly['9M pos'] == 1, '9M adj ret'] = df_monthly['ret']
df_monthly.loc[df_monthly['9M pos'] == 0, '9M adj ret'] = df_monthly['risk free ret']

df_monthly.loc[df_monthly['12M pos'] == 1, '12M adj ret'] = df_monthly['ret']
df_monthly.loc[df_monthly['12M pos'] == 0, '12M adj ret'] = df_monthly['risk free ret']

""" Monthly Log Returns of Trending Following Strategy """
df_monthly.loc[df_monthly['1M pos'] == 1, '1M adj log ret'] = df_monthly['log ret']
df_monthly.loc[df_monthly['1M pos'] == 0, '1M adj log ret'] = df_monthly['risk free log ret']

df_monthly.loc[df_monthly['2M pos'] == 1, '2M adj log ret'] = df_monthly['log ret']
df_monthly.loc[df_monthly['2M pos'] == 0, '2M adj log ret'] = df_monthly['risk free log ret']

df_monthly.loc[df_monthly['3M pos'] == 1, '3M adj log ret'] = df_monthly['log ret']
df_monthly.loc[df_monthly['3M pos'] == 0, '3M adj log ret'] = df_monthly['risk free log ret']

df_monthly.loc[df_monthly['6M pos'] == 1, '6M adj log ret'] = df_monthly['log ret']
df_monthly.loc[df_monthly['6M pos'] == 0, '6M adj log ret'] = df_monthly['risk free log ret']

df_monthly.loc[df_monthly['9M pos'] == 1, '9M adj log ret'] = df_monthly['log ret']
df_monthly.loc[df_monthly['9M pos'] == 0, '9M adj log ret'] = df_monthly['risk free log ret']

df_monthly.loc[df_monthly['12M pos'] == 1, '12M adj log ret'] = df_monthly['log ret']
df_monthly.loc[df_monthly['12M pos'] == 0, '12M adj log ret'] = df_monthly['risk free log ret']

""" Cumulative Log Returns of Trending Following Strategy """
df_monthly['cum log ret'] = df_monthly['log ret'].cumsum()
df_monthly['1M cum log ret'] = df_monthly['1M adj log ret'].cumsum()
df_monthly['2M cum log ret'] = df_monthly['2M adj log ret'].cumsum()
df_monthly['3M cum log ret'] = df_monthly['3M adj log ret'].cumsum()
df_monthly['6M cum log ret'] = df_monthly['6M adj log ret'].cumsum()
df_monthly['9M cum log ret'] = df_monthly['9M adj log ret'].cumsum()
df_monthly['12M cum log ret'] = df_monthly['12M adj log ret'].cumsum()

""" Drawdowns of Trending Following Strategy """
df_monthly['dollar growth'] = (1 + df_monthly['ret']).cumprod()
df_monthly['dd'] = 1 - df_monthly['dollar growth'] / df_monthly['dollar growth'].cummax()

df_monthly['1M dollar growth'] = (1 + df_monthly['1M adj ret']).cumprod()
df_monthly['1M dd'] = 1 - df_monthly['1M dollar growth'] / df_monthly['1M dollar growth'].cummax()

df_monthly['2M dollar growth'] = (1 + df_monthly['2M adj ret']).cumprod()
df_monthly['2M dd'] = 1 - df_monthly['2M dollar growth'] / df_monthly['2M dollar growth'].cummax()

df_monthly['3M dollar growth'] = (1 + df_monthly['3M adj ret']).cumprod()
df_monthly['3M dd'] = 1 - df_monthly['3M dollar growth'] / df_monthly['3M dollar growth'].cummax()

df_monthly['6M dollar growth'] = (1 + df_monthly['6M adj ret']).cumprod()
df_monthly['6M dd'] = 1 - df_monthly['6M dollar growth'] / df_monthly['6M dollar growth'].cummax()

df_monthly['9M dollar growth'] = (1 + df_monthly['9M adj ret']).cumprod()
df_monthly['9M dd'] = 1 - df_monthly['9M dollar growth'] / df_monthly['9M dollar growth'].cummax()

df_monthly['12M dollar growth'] = (1 + df_monthly['12M adj ret']).cumprod()
df_monthly['12M dd'] = 1 - df_monthly['12M dollar growth'] / df_monthly['12M dollar growth'].cummax()

# df_monthly.index = pd.to_datetime(df_monthly.index).to_period('m')
# df_monthly_output = df_monthly[['6M pos', '9M pos', '12M pos']]
# df_monthly_output.to_excel('trend following output.xlsx')

""" 
    Generate a summary table of benchmark and trend following strategies performance including
        - annualized average return
        - annualized volatility
        - sharpe ratio, 
        - cumulative log return
        - maximum drawdown
        - average trade per year. 
        
"""
n_year = len(df_monthly) / 12

tf_ret = df_monthly[['ret', '1M adj ret', '2M adj ret', '3M adj ret', '6M adj ret', '9M adj ret', '12M adj ret']].copy()

tf_stat = pd.concat([tf_ret.mean() * 12,
                     tf_ret.std() * np.sqrt(12),
                     tf_ret.mean() / tf_ret.std() * np.sqrt(12)], axis=1)

tf_stat.columns = ['Average Return (Annualized)', 'Volatility (Annualized)', 'Sharpe Ratio']

tf_stat['Cumulative Log Return'] \
    = [df_monthly['cum log ret'][-1], df_monthly['1M cum log ret'][-1], df_monthly['2M cum log ret'][-1],
       df_monthly['3M cum log ret'][-1], df_monthly['6M cum log ret'][-1], df_monthly['9M cum log ret'][-1],
       df_monthly['12M cum log ret'][-1]]

tf_stat['Maximum Drawdown'] \
    = [df_monthly['dd'].max(), df_monthly['1M dd'].max(), df_monthly['2M dd'].max(),
       df_monthly['3M dd'].max(), df_monthly['6M dd'].max(), df_monthly['9M dd'].max(),
       df_monthly['12M dd'].max()]

tf_stat['#Trade/Year'] \
    = ["", df_monthly['1M trade'].sum() / n_year, df_monthly['2M trade'].sum() / n_year,
       df_monthly['3M trade'].sum() / n_year, df_monthly['6M trade'].sum() / n_year,
       df_monthly['9M trade'].sum() / n_year, df_monthly['12M trade'].sum() / n_year]

tf_stat.index = ['SP 500', '1M', '2M', '3M', '6M', '9M', '12M']

# Export the summary table to excel
tf_stat.to_excel('tf_stat.xlsx')

""" 
    Plot cummulative log returns and drawdowns of benchmark and trend following strategies
        
"""
#colors = ['royalblue', 'darkorange', 'gold', 'lightcoral', 'forestgreen']
# labels = ['SP500', '1M', '2M', '3M', '6M', '9M', '12M']
# tf_cum_log_ret = df_monthly[['cum log ret', '1M cum log ret', '2M cum log ret', '3M cum log ret',
#                              '6M cum log ret', '9M cum log ret', '12M cum log ret']].copy()
# tf_dd = df_monthly[['dd', '1M dd', '2M dd', '3M dd', '6M dd', '9M dd', '12M dd']].copy()
labels = ['SP500', '6M', '9M', '12M']
tf_cum_log_ret = df_monthly[['cum log ret', '6M cum log ret', '9M cum log ret', '12M cum log ret']].copy()
tf_dd = df_monthly[['dd', '6M dd', '9M dd', '12M dd']].copy()

# Plot the cumulative log returns of S&P 500 and trend following strategies
plt.rcParams["figure.figsize"] = [12, 4]
plt.title('S&P 500 vs. Trend Following Strategy')
plt.plot(tf_cum_log_ret, label=labels)
plt.xlabel('Dates')
plt.ylabel('Cumulative Log Returns')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.grid()
plt.show()

# Plot the drawdowns of S&P 500 and trend following strategies
plt.rcParams["figure.figsize"] = [12, 4]
plt.title('S&P 500 vs. Trend Following Strategy')
plt.plot(tf_dd, label=labels)
plt.xlabel('Dates')
plt.ylabel('Drawdowns')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.grid()
plt.show()

# Export the data to excel
df_monthly.to_excel('Trend Following - SPX.xlsx')


""" 
    Incorporate macro regimes into the trend following strategies 
    - High/low yield regimes
    - High/low yield spread regimes
    - Uptrend/downtrend yield regimes
    - High/low inflation regimes
    
"""

# Function used to compute the statistics of each regime
def compute_regime_statistics(ind, regime, data):
    regime_ret = data.loc[data[ind] == regime, ['ret', '1M adj ret', '2M adj ret', '3M adj ret', '6M adj ret', '9M adj ret', '12M adj ret']]
    regime_dd = data.loc[data[ind] == regime, ['dd', '1M dd', '2M dd', '3M dd', '6M dd', '9M dd', '12M dd']]
    regime_trade = data.loc[data[ind] == regime, ['1M trade', '2M trade', '3M trade', '6M trade', '9M trade', '12M trade']]

    n_year_regime = len(regime_ret) / 12

    # average, volatility, and sharpe ratio
    regime_stat = pd.concat([regime_ret.mean() * 12,
                             regime_ret.std() * np.sqrt(12),
                             regime_ret.mean() / regime_ret.std() * np.sqrt(12)], axis=1)

    regime_stat.columns = ['Average Return (Annualized)', 'Volatility (Annualized)', 'Sharpe Ratio']
    regime_stat.index = ['index', '1M', '2M', '3M', '6M', '9M', '12M']

    regime_stat['Maximum Drawdown'] = regime_dd.max().to_list()

    regime_stat['#Trade/Year'] = ""
    regime_stat.loc[1:, '#Trade/Year'] = np.asarray(regime_trade.sum()) / n_year_regime

    return regime_stat

#-----------------------------------------------------------------------------------------------------------------------
""" High/Low Interest Rate Regimes """
# df_monthly['1y tr'].describe()
#
# plt.hist(df_monthly['1y tr'], bins=100)
# plt.axvline(x=df_monthly['1y tr'].mean(), ls='--', color='lightcoral')

# Define the regimes
ir_avg = df_monthly['1y tr'].mean()
df_monthly.loc[df_monthly['1y tr'] >= ir_avg, 'ir ind'] = 1  # high interest rate
df_monthly.loc[df_monthly['1y tr'] < ir_avg, 'ir ind'] = 0   # low interest rate

# Get the statistics of each regime
high_ir_stat = compute_regime_statistics(ind='ir ind', regime=1, data=df_monthly)
low_ir_stat = compute_regime_statistics(ind='ir ind', regime=0, data=df_monthly)

# Export the tables to excel
high_ir_stat.to_excel('high ir.xlsx')
low_ir_stat.to_excel('low ir.xlsx')

# Plot the cumulative log returns of S&P 500 and trend following strategies with interest rate regimes
plt.rcParams["figure.figsize"] = [12, 4]
plt.title('S&P 500 vs. Trend Following Strategy')
plt.plot(tf_cum_log_ret, label=labels)
for month in df_monthly.loc[df_monthly['ir ind'] == 1].index:
    plt.axvspan(month, month + timedelta(days=31), color="lightgrey", alpha=0.5)
plt.xlabel('Dates')
plt.ylabel('Cumulative Log Returns')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.grid()
plt.show()

#-----------------------------------------------------------------------------------------------------------------------
""" High/Low 1Y-20Y Yield Spread Regimes """
# df_monthly['1y 20y spread'].describe()
#
# plt.hist(df_monthly['1y 20y spread'], bins=100)
# plt.axvline(x=df_monthly['1y 20y spread'].mean(), ls='--', color='lightcoral')

# define the regimes
irsp_avg = df_monthly['1y 20y spread'].mean()
df_monthly.loc[df_monthly['1y 20y spread'] >= irsp_avg, 'irsp ind'] = 1  # high 1Y - 20Y yield spread
df_monthly.loc[df_monthly['1y 20y spread'] < irsp_avg, 'irsp ind'] = 0   # low 1Y - 20Y yield spread

# Get the statistics of each regime
high_irsp_stat = compute_regime_statistics(ind='irsp ind', regime=1, data=df_monthly)
low_irsp_stat = compute_regime_statistics(ind='irsp ind', regime=0, data=df_monthly)

# Export the tables to excel
high_irsp_stat.to_excel('high irsp.xlsx')
low_irsp_stat.to_excel('low irsp.xlsx')

# Plot the cumulative log returns of S&P 500 and trend following strategies with yield spread regimes
plt.rcParams["figure.figsize"] = [12, 4]
plt.title('S&P 500 vs. Trend Following Strategy')
plt.plot(tf_cum_log_ret, label=labels)
for month in df_monthly.loc[df_monthly['irsp ind'] == 1].index:
    plt.axvspan(month, month + timedelta(days=31), color="lightgrey", alpha=0.5)
plt.xlabel('Dates')
plt.ylabel('Cumulative Log Returns')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.grid()
plt.show()

#-----------------------------------------------------------------------------------------------------------------------
""" Before & After September 1981 """
df_monthly.loc[df_monthly.index < datetime(1981, 9, 1), 'irtr ind'] = 1
df_monthly.loc[df_monthly.index >= datetime(1981, 9, 1), 'irtr ind'] = 0

# Get the statistics of each regime
up_irtr_stat = compute_regime_statistics(ind='irtr ind', regime=1, data=df_monthly)
down_irtr_stat = compute_regime_statistics(ind='irtr ind', regime=0, data=df_monthly)

# Export the tables to excel
up_irtr_stat.to_excel('up irtr.xlsx')
down_irtr_stat.to_excel('down irtr.xlsx')

#-----------------------------------------------------------------------------------------------------------------------
""" High/Low Inflation Regimes """
df_monthly['cpi'].describe()

plt.hist(df_monthly['cpi'], bins=100)
plt.axvline(x=df_monthly['cpi'].mean(), ls='--', color='lightcoral')

# define the regimes
cpi_avg = df_monthly['cpi'].mean()
df_monthly.loc[df_monthly['cpi'] >= cpi_avg, 'cpi ind'] = 1  # high inflation
df_monthly.loc[df_monthly['cpi'] < cpi_avg, 'cpi ind'] = 0   # low inflation

# Get the statistics of each regime
high_cpi_stat = compute_regime_statistics(ind='cpi ind', regime=1, data=df_monthly)
low_cpi_stat = compute_regime_statistics(ind='cpi ind', regime=0, data=df_monthly)

# Export the tables to excel
high_cpi_stat.to_excel('high cpi.xlsx')
low_cpi_stat.to_excel('low cpi.xlsx')

# Plot the cumulative log returns of S&P 500 and trend following strategies with inflation regimes
plt.rcParams["figure.figsize"] = [12, 4]
plt.title('S&P 500 vs. Trend Following Strategy')
plt.plot(tf_cum_log_ret, label=labels)
for month in df_monthly.loc[df_monthly['cpi ind'] == 1].index:
    plt.axvspan(month, month + timedelta(days=31), color="lightgrey", alpha=0.5)
plt.xlabel('Dates')
plt.ylabel('Cumulative Log Returns')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.grid()
plt.show()

#-----------------------------------------------------------------------------------------------------------------------
""" High/Low Credit Spread Regimes """
df_monthly['credit spread'].describe()

plt.hist(df_monthly['credit spread'], bins=100)
plt.axvline(x=df_monthly['credit spread'].mean(), ls='--', color='lightcoral')

# define the regimes
cs_avg = df_monthly['credit spread'].mean()
df_monthly.loc[df_monthly['credit spread'] >= cs_avg, 'cs ind'] = 1  # high credit spread
df_monthly.loc[df_monthly['credit spread'] < cs_avg, 'cs ind'] = 0   # low credit spread

# Get the statistics of each regime
high_cs_stat = compute_regime_statistics(ind='cs ind', regime=1, data=df_monthly)
low_cs_stat = compute_regime_statistics(ind='cs ind', regime=0, data=df_monthly)

# Export the tables to excel
high_cs_stat.to_excel('high cs.xlsx')
low_cs_stat.to_excel('low cs.xlsx')

# Plot the cumulative log returns of S&P 500 and trend following strategies with inflation regimes
plt.rcParams["figure.figsize"] = [12, 4]
plt.title('S&P 500 vs. Trend Following Strategy')
plt.plot(tf_cum_log_ret, label=labels)
for month in df_monthly.loc[df_monthly['cs ind'] == 1].index:
    plt.axvspan(month, month + timedelta(days=31), color="lightgrey", alpha=0.5)
plt.xlabel('Dates')
plt.ylabel('Cumulative Log Returns')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.grid()
plt.show()


#-----------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------
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

""" 
    Generate a summary table of benchmark and time-series momentum strategies performance including
        - annualized average return
        - annualized volatility
        - sharpe ratio, 
        - cumulative log return
        - maximum drawdown

"""
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

