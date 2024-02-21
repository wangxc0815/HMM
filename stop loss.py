import sys
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


""" 
    Enter Inputs Here 
"""
file_name = r"C:\Users\Xiaochun.wang\Desktop\Data\Trend Following Data.xlsx"
index = 'SPX Index'
rfr = 'H15T3M Index'

start_date = datetime.datetime(1940, 1, 1)
end_date = datetime.datetime(2023, 12, 31)

"""
    Import Dataset (Weekly Frequency) 
"""
data_weekly = pd.read_excel(file_name, "weekly")
df_weekly = data_weekly[['Dates', index, rfr]].copy()
df_weekly['Dates'] = pd.to_datetime(df_weekly['Dates'])
df_weekly = df_weekly.loc[(df_weekly['Dates'] >= datetime.datetime(1928, 1, 1)) & (df_weekly['Dates'] <= datetime.datetime(2023, 12, 31))]

# Compute weekly returns, log returns, dollar growth, and 1-year maximum drawdowns
df_weekly['ret'] = df_weekly[index].pct_change(1)   # weekly returns
df_weekly['log ret'] = np.log(1 + df_weekly['ret'])
df_weekly['dollar growth'] = np.cumprod(1 + df_weekly['ret'])
#df_weekly = df_weekly.dropna().reset_index(drop=True)
df_weekly = df_weekly.reset_index(drop=True)
df_weekly['1-year dd'] = df_weekly[index]/df_weekly[index].shift(1).rolling(52).max()-1   # 1-year maximum drawdown
df_weekly.loc[df_weekly['1-year dd'] > 0, '1-year dd'] = 0

df_weekly['risk free ret'] = (df_weekly[rfr] / 100 + 1) ** (1/52) - 1
# Fill up the unavailable 3M US Treasury returns with the historical average
df_weekly['risk free ret'] = df_weekly['risk free ret'].fillna(df_weekly['risk free ret'].mean())
df_weekly['risk free log ret'] = np.log(1 + df_weekly['risk free ret'])


# """
#     Import Dataset (Daily Frequency)
# """
# # Compute annualized 21-day volatilities
# data_daily = pd.read_excel(file_name, "daily")
# df_daily = data_daily[['Dates', 'SPX']].copy()
# df_daily['Dates'] = pd.to_datetime(df_daily['Dates'])
# df_daily['ret'] = df_daily['SPX'].pct_change(1)   # daily returns
# df_daily['1-month vol'] = df_daily['ret'].shift(1).rolling(21).var() * 252   # annualized 21-day volatility
#
# # Merge the volatilities
# df = pd.merge(df_weekly, df_daily[['Dates', '1-month vol']], on='Dates', how='left')
#
# # Fill in the missing volatilities
# dates_missing_vol = df.loc[df['1-month vol'].isnull(), 'Dates']
# # Replace NA with the volatility from previous date
# for i in dates_missing_vol:
#     prev_date = i + datetime.timedelta(days=-1)
#     df.loc[df['Dates'] == i, '1-month vol'] = df_daily.loc[df_daily['Dates'] == prev_date, '1-month vol'].squeeze()
#
# dates_missing_vol1 = df.loc[df['1-month vol'].isnull(), 'Dates']
# # Replace NA with the volatility from two days ago
# for i in dates_missing_vol1:
#     prev_date = i + datetime.timedelta(days=-2)
#     df.loc[df['Dates'] == i, '1-month vol'] = df_daily.loc[df_daily['Dates'] == prev_date, '1-month vol'].squeeze()
#
# dates_missing_vol2 = df.loc[df['1-month vol'].isnull(), 'Dates']
# # Replace NA in the week of Sep 10, 2010
# df.loc[df['Dates'] == datetime.datetime(2001,9,14),'1-month vol'] \
#     = df_daily.loc[df_daily['Dates'] == datetime.datetime(2001,9,10), '1-month vol'].squeeze()
#
# # Change the last trading day of each week to Sunday in order to make the dates consistent
# df.set_index("Dates", inplace=True)
# df.index = df.index.to_period('W').to_timestamp('W')

""" Construct the Stop Loss Strategy """
def apply_stop_loss_strategy(start_date, end_date, df, loss_level, gain_level, x):
    # take the subset of the original dataset
    data = df.loc[(df['Dates'] >= start_date) & (df['Dates'] <= end_date)].copy()
    data.index = list(range(0, len(data.axes[0])))
    data['cum log ret'] = data['log ret'].cumsum()

    # initialize
    reset = 0
    total_reset = 0
    data['reset point'] = 'NaN'  # cumulative log returns when exit/reset
    data.loc[0, 'reset point'] = data.loc[1, 'reset point'] = data['cum log ret'][0] #= data.loc[2, 'reset point']

    peak = bottom = data['cum log ret'][0]

    data['pos'] = 'NaN'  # hold: 1  not hold: 0
    data.loc[0, 'pos'] = data.loc[1, 'pos'] = 1 #= data.loc[2, 'pos']

    # (for testing)
    test = np.zeros((len(data), 3))
    rebalance = pd.DataFrame(columns=['start', 'end', 'duration', 'trigger'])
    start = end = data['Dates'][0].to_period('W').to_timestamp('W')
    duration = 1
    trigger = ''

    for i in range(2, len(data)):
        duration += 1

        # if we currently hold the asset
        if data['pos'][i - 1] == 1:
            """ Exit Signal """
            # if it drops from the peak by loss level%, then exit
            # if data['2-lms dd'][i-3] <= loss_level:
            if data['cum log ret'][i - 1] - peak <= loss_level \
                    and data['cum log ret'][i - 2] - peak <= loss_level:
                # and data['cum log ret'][i-3] - peak <= loss_level:
                # and data['cum log ret'][i-4] - peak <= loss_level:
                data.loc[i, 'pos'] = 0
                data.loc[i, 'reset point'] = data['cum log ret'][i]
            # if loss level is not triggered, then stay in
            else:
                data.loc[i, 'pos'] = 1
                data.loc[i, 'reset point'] = data['reset point'][i - 1]

        # if we currenty dont hold the asset
        else:
            """ Re-enter Signal """
            # if it bounces back from the bottom by gain level% or it reaches (the previous exit point + x%), then re-enter
            if (data['cum log ret'][i - 1] - bottom >= gain_level \
                and data['cum log ret'][i - 2] - bottom >= gain_level) \
                    or (test[i - 1, 0] == 0 and test[i - 2, 0] == 0 \
                        and data['cum log ret'][i - 1] >= data['reset point'][i - 1] + x \
                        and data['cum log ret'][i - 2] >= data['reset point'][i - 2] + x):
                # and data['cum log ret'][i-3] - bottom >= gain_level:
                # and data['cum log ret'][i-4] - bottom >= gain_level:
                data.loc[i, 'pos'] = 1
                data.loc[i, 'reset point'] = data['cum log ret'][i]
            # if gain level is not triggered, then stay out
            else:
                data.loc[i, 'pos'] = 0
                data.loc[i, 'reset point'] = data['reset point'][i - 1]

        # duration += 1
        if data['pos'][i] == 0 and data['pos'][i - 1] == 1:   # buy --> sell
            end = data['Dates'][i - 1].to_period('W').to_timestamp('W')
            loss = data['cum log ret'][i - 1] - peak
            trigger = 'Trigger Loss Level'
            new_entry = pd.DataFrame([[start, end, duration, trigger]], columns=['start', 'end', 'duration', 'trigger'])
            rebalance = pd.concat([rebalance, new_entry], sort=False)

            reset = 1
            duration = 1
            start = data['Dates'][i].to_period('W').to_timestamp('W')
        elif data['pos'][i] == 1 and data['pos'][i - 1] == 0:   # sell --> buy
            end = data['Dates'][i - 1].to_period('W').to_timestamp('W')
            if data['cum log ret'][i - 1] - bottom >= gain_level:
                gain = data['cum log ret'][i - 1] - bottom
                trigger = 'Trigger Gain Level'
            else:
                gain = data['cum log ret'][i - 1] - bottom
                trigger = 'Reach the Previous Exit Point'
            new_entry = pd.DataFrame([[start, end, duration, trigger]], columns=['start', 'end', 'duration', 'trigger'])
            rebalance = pd.concat([rebalance, new_entry], sort=False)

            reset = 1
            duration = 1
            start = data['Dates'][i].to_period('W').to_timestamp('W')
        else:   # no change
            reset = 0

        # reset = 1 if data['pos'][i] != data['pos'][i - 1] else 0

        # (for testing - trace the peak and bottom after each reset)
        test[i, :] = [reset, peak, bottom]

        if reset == 1:
            peak = bottom = data['cum log ret'][i]
            duration = 0
        else:
            if data['cum log ret'][i] > peak:
                peak = data['cum log ret'][i]
            if data['cum log ret'][i] < bottom:
                bottom = data['cum log ret'][i]

        i = i + 1

    data.loc[data['pos'] == 0, 'adj ret'] = data['risk free ret']
    data.loc[data['pos'] == 1, 'adj ret'] = data['ret']
    data.loc[data['pos'] == 0, 'adj log ret'] = data['risk free log ret']
    data.loc[data['pos'] == 1, 'adj log ret'] = data['log ret']
    data['adj cum log ret'] = data['adj log ret'].cumsum()

    #avg_ret = np.mean(data['ret']) * 52
    avg_adj_ret = np.mean(data['adj ret']) * 52

    #sharpe_ratio = np.mean(data['ret']) / np.std(data['ret']) * (52 ** (1 / 2))
    adj_sharpe_ratio = np.mean(data['adj ret']) / np.std(data['adj ret']) * (52 ** (1 / 2))

    total_reset = np.sum(test[:, 0]) + 1

    return data, test, avg_adj_ret, adj_sharpe_ratio, total_reset, rebalance


""" For a sensitivity test of a range of the exit and re-enter thresholds """
loss_level = -np.arange(0.05, 0.155, 0.01)   # Exit triggers
gain_level = np.arange(0.05, 0.155, 0.01)   # Re-enter triggers
x = 0.01   # Re-enter if the return reaches the previous exit point + x%

avg_adj_ret_table = np.zeros((len(loss_level), len(gain_level)))
adj_sharpe_ratio_table = np.zeros((len(loss_level), len(gain_level)))
total_reset_table = np.zeros((len(loss_level), len(gain_level)))

for i in range(0, len(loss_level)):
    for j in range(0, len(gain_level)):
        data, test, avg_adj_ret, adj_sharpe_ratio, total_reset, rebalance \
            = apply_stop_loss_strategy(start_date, end_date, df_weekly, loss_level[i], gain_level[j], x)
        avg_adj_ret_table[i, j] = avg_adj_ret
        adj_sharpe_ratio_table[i, j] = adj_sharpe_ratio
        total_reset_table[i, j] = total_reset

# Export the summary tables to excel
adj_sharpe_ratio_table = pd.DataFrame(adj_sharpe_ratio_table)
adj_sharpe_ratio_table.to_excel('sharpe ratio.xlsx')

avg_adj_ret_table = pd.DataFrame(avg_adj_ret_table)
avg_adj_ret_table.to_excel('avg ret.xlsx')

total_reset_table = pd.DataFrame(total_reset_table)
total_reset_table.to_excel('total reset.xlsx')


""" For a single test case of the exit and re-enter threshold """
loss_level = -0.07   # Exit triggers
gain_level = 0.07   # Re-enter triggers
x = 0.04   # Re-enter if the return reaches the previous exit point + x%

data, test, avg_adj_ret, adj_sharpe_ratio, total_reset, rebalance \
    = apply_stop_loss_strategy(start_date, end_date, df_weekly, loss_level, gain_level, x)
avg_ret = np.mean(data['ret']) * 52
vol = np.std(data['ret']) * (52 ** (1 / 2))
sharpe_ratio = avg_ret / vol

# Plot the cumulative log returns of S&P 500 and stop loss strategy
plt.figure(figsize=(20, 4))
plt.title('Stop Loss')
plt.plot(data['Dates'], data['cum log ret'], label='S&P 500')
plt.plot(data['Dates'], data['adj cum log ret'], label='Stop Loss')
plt.xlabel('Dates')
plt.ylabel('Cumulative Log Returns')
plt.legend()
plt.show()

# Export the single test case to excel
test = pd.DataFrame(test)
test.index = data['Dates']
test.columns = ['reset', 'peak', 'bottom']
test.to_excel('test.xlsx')

data.to_excel('data.xlsx')

data.set_index("Dates", inplace=True)
data.index = pd.to_datetime(data.index).to_period('m')
tf_data = pd.read_excel('trend following output.xlsx')
tf_data.set_index("Dates", inplace=True)
tf_data.index = pd.to_datetime(tf_data.index).to_period('m')

sl_tf_data = pd.merge(data, tf_data, left_index=True, right_index=True)


# data.to_excel('data.xlsx')
sl_tf_data.to_excel('stop loss & trend following.xlsx')