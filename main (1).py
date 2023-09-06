import sys
from datetime import datetime
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from warnings import simplefilter

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

np.set_printoptions(threshold=sys.maxsize)

pd.options.mode.chained_assignment = None
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

""" Hidden Markov Model """
def array_alloc(nweeks, nstates):
    pto = np.zeros([nweeks + 1, nstates])
    pq = np.zeros([nstates, nstates])
    p = np.zeros([nweeks, nstates * nstates])
    px = np.zeros([nweeks, nstates])
    density = np.zeros([nweeks, nstates])
    f = np.zeros([nweeks, nstates])
    norm = np.zeros(nweeks)
    loglike = np.zeros(nweeks)

    mu = np.zeros(nstates)
    sigma = np.zeros(nstates)

    return ([pto, pq, p, px, density, f, norm, loglike, mu, sigma])

def array_realloc(nweeks, nstates):
    pto = np.zeros([nweeks + 1, nstates])
    p = np.zeros([nweeks, nstates * nstates])
    px = np.zeros([nweeks, nstates])
    density = np.zeros([nweeks, nstates])
    f = np.zeros([nweeks, nstates])
    norm = np.zeros(nweeks)
    loglike = np.zeros(nweeks)

    return ([pto, p, px, density, f, norm, loglike])

# def winsorize(series, lower=0.5, upper=99.5):
#     """ windsorize """
#     pc_low  = np.percentile(series, lower)
#     pc_high = np.percentile(series, upper)
#     wseries = series.copy()
#     wseries[wseries<pc_low] = pc_low
#     wseries[wseries>pc_high] = pc_high
#     return wseries

# def smooth_series(wseries, seed=7, hl=1):
#     """ smoothing """
#     a = np.exp(-np.log(2)/hl)
#     b = 1 - a

#     update = np.mean(wseries[:seed])
#     smooth = list()
#     for element in wseries:
#         update = update*a + element*b
#         smooth.append(update)
#     return np.array(smooth)

def initialize_hmm(mu, sigma, pq, nstates, series):
    """ place holder parms for testing """
    std = np.std(series)

    for k in range(nstates):
        sigma[k] = std
        pq[k, k] = 3.0
        mu[k] = np.percentile(series, 100 * (k + 1) / (nstates + 1))

def regime_fit(nstates, smooth, pto, tr, p, px, density, f, norm, loglike, mu, sigma):
# def regime_fit(nstates, smooth, pto, pq, p, px, density, f, norm, loglike, mu, sigma):
    nweeks = len(smooth)

    tpi = np.sqrt(2.0 * np.pi)
    nrm = 1.0 / (tpi * sigma)

    # zpq = np.exp(pq)
    # pz = (zpq.T / zpq.sum(axis=1))
    pz = tr.T
    ic = 0
    for element in pz:
        p[:, ic:(ic + nstates)] = element
        ic = ic + nstates

    pto[0, :] = 1 / nstates

    for t in range(nweeks):
        ic = 0
        for k in range(nstates):
            px[t, k] = np.dot(pto[t, :], p[t, ic:(ic + nstates)])
            ic = ic + nstates

        y = (smooth[t] - mu) / sigma
        eps = y * y / 2.0
        density[t, :] = np.exp(-eps) * nrm
        f[t, :] = density[t, :] * px[t, :]
        norm[t] = np.sum(f[t, :])
        loglike[t] = np.log(norm[t])
        pto[t + 1, :] = f[t, :] / norm[t]


    return np.sum(loglike)

def initialize_search(series):
    sigma_min = np.std(series) * 0.2
    sigma_max = np.std(series) * 3.0
    mu_min = min(series)
    mu_max = max(series)
    pq_min = -10
    pq_max = 10

    return (mu_min, mu_max, sigma_min, sigma_max, pq_min, pq_max)

def kim_filter(pto, tr, px, nweeks, nstates):
# def kim_filter(pto, px, nweeks, nstates):
    ptos = np.zeros([nweeks, nstates])
    ps = np.zeros([nweeks, nstates * nstates])
    # zpq = np.exp(pq)
    # pz = (zpq.T / zpq.sum(axis=1))
    pz = tr.T

    ptos[nweeks - 1, :] = pto[nweeks, :]
    for t in range(nweeks - 1, 0, -1):
        for k in range(nstates):
            ps[t, k * nstates:((k + 1) * nstates)] = pto[t, k] * ptos[t, :] * pz[:, k] / px[t, :]
        for k in range(nstates):
            ptos[t - 1, k] = sum(ps[t, k * nstates:(k + 1) * nstates])

    return ptos

def search(sigma_min, sigma_max, dsigma, mu_min, mu_max, dmu, pq_min, pq_max, dpq, series):
    for m in range(3):
        for state in range(nstates):
            # ic = 0
            mlike = regime_fit(nstates, series, pto, pq, p, px, density, f, norm, loglike, mu, sigma)
            best = sigma[state]
            for sig in np.arange(sigma_min[state], sigma_max[state], dsigma[state]):
                sigma[state] = sig
                like = regime_fit(nstates, series, pto, pq, p, px, density, f, norm, loglike, mu, sigma)
                if like > mlike:
                    mlike = like
                    best = sig
                # ic = ic + 1
            sigma[state] = best
            maxlike = regime_fit(nstates, series, pto, pq, p, px, density, f, norm, loglike, mu, sigma)
            # print("sigma:", state, ic)

        for state in range(nstates):
            for element in range(0, nstates):
                # ic = 0
                best = pq[state][element]
                mlike = regime_fit(nstates, series, pto, pq, p, px, density, f, norm, loglike, mu, sigma)
                for update in np.arange(pq_min[state, element], pq_max[state, element], dpq[state][element]):
                    pq[state][element] = update
                    like = regime_fit(nstates, series, pto, pq, p, px, density, f, norm, loglike, mu, sigma)
                    if like > mlike:
                        mlike = like
                        best = update
                    # ic = ic + 1
                # print("pq:", state, element, ic)
                pq[state][element] = best
                maxlike = regime_fit(nstates, series, pto, pq, p, px, density, f, norm, loglike, mu, sigma)

    for state in range(nstates):
        mlike = regime_fit(nstates, series, pto, pq, p, px, density, f, norm, loglike, mu, sigma)
        best = mu[state]
        # ic = 0
        for mux in np.arange(mu_min[state], mu_max[state], dmu[state]):
            mu[state] = mux
            like = regime_fit(nstates, series, pto, pq, p, px, density, f, norm, loglike, mu, sigma)
            if like > mlike:
                mlike = like
                best = mux
            # ic = ic + 1
        # print("mu:", state, ic)
        mu[state] = best
        maxlike = regime_fit(nstates, series, pto, pq, p, px, density, f, norm, loglike, mu, sigma)

def results(ptos, likeli):
    fig, ax = plt.subplots(2, 1, figsize=(20, 8))
    ax[0].stackplot(range(ptos.shape[0]), ptos.T, colors=colors)
    ax[0].plot(np.array(rates) * 10 + .5, color="red", linewidth=0.5)
    ax[0].set_ylim([0, 1])
    ax[0].set_xlim([0, len(rates)])
    ax[0].set_ylabel("Regime Probability", color="blue")
    ax[0].set_xlabel("Date")

    ax[1].plot(likeli, color="blue")
    ax[1].grid(color="lightblue", alpha=0.5)
    ax[1].set_xlabel("iterations", color="blue")
    ax[1].set_ylabel("likelihood", color="blue")
    ax[1].set_xlim([0, None])

    plt.show()


""" 
    Enter Inputs Here 
"""
start_date = datetime(1940, 1, 1)
end_date = datetime(2023, 6, 30)


""" 
    Import Dataset (Daily Frequency) 
"""
# Import dataset
data_daily = pd.read_excel(r"C:\Users\Xiaochun.wang\Desktop\Data\Stop Loss Data.xlsx", "daily")

df_daily = data_daily[['Dates', 'SPX']].copy()
df_daily.set_index("Dates", inplace=True)

""" Daily Returns """
# Compute daily returns of S&P 500
df_daily['ret'] = df_daily['SPX'].pct_change(1)

# Winsorize daily returns
df_daily['ret 1'] = df_daily['ret']
threshold_lower = np.nanpercentile(df_daily['ret'], 1)
df_daily.loc[df_daily['ret'] <= threshold_lower, 'ret 1'] = threshold_lower
threshold_upper = np.nanpercentile(df_daily['ret'], 99)
df_daily.loc[df_daily['ret'] >= threshold_upper, 'ret 1'] = threshold_upper

# Compute the exponential weighted moving average of daily returns
df_daily['ret ewma'] = df_daily['ret 1'].ewm(span=42, ignore_na=True).mean()

""" Daily Log Returns """
# Compute daily log returns of S&P 500
df_daily['log ret'] = np.log(1 + df_daily['ret'])

# Winsorize daily log returns
df_daily['log ret 1'] = df_daily['log ret']
threshold_lower = np.nanpercentile(df_daily['log ret'], 0.1)
df_daily.loc[df_daily['log ret'] <= threshold_lower, 'log ret 1'] = threshold_lower
threshold_upper = np.nanpercentile(df_daily['log ret'], 99.9)
df_daily.loc[df_daily['log ret'] >= threshold_upper, 'log ret 1'] = threshold_upper

# Compute the exponential weighted moving average of daily log returns
df_daily['log ret ewma'] = df_daily['log ret 1'].ewm(span=63, ignore_na=True).mean()

""" 1M Volatility """
# Compute annualized 21-day volatilities of S&P 500
df_daily['1-month vol'] = df_daily['log ret 1'].rolling(21).var() * (252 ** (1/2))

# Compute the exponential weighted moving average of 1-month volatility
df_daily['1-month vol ewma'] = df_daily['1-month vol'].ewm(span=63, ignore_na=True).mean()

# Select the time period
df_daily = df_daily.loc[(df_daily.index >= start_date) & (df_daily.index <= end_date)]


"""
    Import Dataset (Weekly Frequency) 
"""
# Import dataset
data_weekly = pd.read_excel(r"C:\Users\Xiaochun.wang\Desktop\Data\Stop Loss Data.xlsx", "weekly")

df_weekly = data_weekly[['Dates', 'SPX', '3M TBILL']].copy()
df_weekly.set_index("Dates", inplace=True)

""" Weekly Returns """
# Compute weekly returns of S&P 500 and 3M tbill
df_weekly['ret'] = df_weekly['SPX'].pct_change(1)
df_weekly['3m tbill ret'] = df_weekly['3M TBILL']
df_weekly['3m tbill log ret'] = np.log(1 + df_weekly['3m tbill ret'])

# Winsorize weekly returns
df_weekly['ret 1'] = df_weekly['ret']
threshold_lower = np.nanpercentile(df_weekly['ret'], 5)
df_weekly.loc[df_weekly['ret'] <= threshold_lower, 'ret 1'] = threshold_lower
threshold_upper = np.nanpercentile(df_weekly['ret'], 95)
df_weekly.loc[df_weekly['ret'] >= threshold_upper, 'ret 1'] = threshold_upper

# Compute the exponential weighted moving average of daily returns
df_weekly['ret ewma'] = df_weekly['ret 1'].ewm(span=24, ignore_na=True).mean()

""" Weekly Log Returns """
# Compute weekly log returns of S&P 500
df_weekly['log ret'] = np.log(1 + df_weekly['ret'])

# Winsorize weekly log returns
df_weekly['log ret 1'] = df_weekly['log ret']
threshold_lower = np.nanpercentile(df_weekly['log ret'], 0.1)
df_weekly.loc[df_weekly['log ret'] <= threshold_lower, 'log ret 1'] = threshold_lower
threshold_upper = np.nanpercentile(df_weekly['log ret'], 99.9)
df_weekly.loc[df_weekly['log ret'] >= threshold_upper, 'log ret 1'] = threshold_upper

# Compute the exponential weighted moving average of daily log returns
df_weekly['log ret ewma'] = df_weekly['log ret 1'].ewm(span=24, ignore_na=True).mean()

# Select the time period
df_weekly = df_weekly.loc[(df_weekly.index >= start_date) & (df_weekly.index <= end_date)]

# Compute the cumulative log returns during the period
df_weekly['cum log ret'] = df_weekly['log ret'].cumsum()

""" 1M Volatility """
# Subset the 21-day volatilities from daily dataset
df_daily_vol = df_daily[['1-month vol', '1-month vol ewma']].copy()

# Change the last trading day of each week to Sunday in order to make the dates consistent
df_weekly.index = df_weekly.index.to_period('W').to_timestamp('W')

# Convert the daily series of the 21-day volatilities into weekly series
ohlc_dict = {'1-month vol': 'last', '1-month vol ewma': 'last'}
df_weekly_vol = df_daily_vol.resample('W').agg(ohlc_dict)

# Merge the weekly series of the 21-day volatilities into weekly dataset
df_weekly = df_weekly.join(df_weekly_vol)
df_weekly = df_weekly.dropna()

""" 
    Define Recession Regime 
"""
recession_regime = {}

recession_regime['start'] = ['1960-04-01', '1969-12-01', '1973-11-01', '1980-01-01', '1981-07-01', '1990-07-01', '2001-03-01', '2007-12-01', '2020-02-01']
recession_regime['end'] = ['1961-02-28', '1970-11-30', '1975-03-31', '1980-07-31', '1982-11-30', '1991-03-31', '2001-11-30', '2009-06-30', '2020-04-30']

recession_regime = pd.DataFrame(recession_regime)

# Plot the cumulative log returns and 21-day volatilities
# plt.rcParams["figure.figsize"] = [20, 4]
#
# fig, ax1 = plt.subplots()
# ax1.plot(df_weekly['1-month vol'], lw=2, color="cornflowerblue")
# ax1.set_ylabel(r"21-Day Volatilities", fontsize=12, color="cornflowerblue")
# for label in ax1.get_yticklabels():
#     label.set_color("cornflowerblue")
# ax1.grid(False)
#
# ax2 = ax1.twinx()
# ax2.plot(df_weekly['cum log ret'], color="darkorange")
# ax2.set_ylabel(r"Cumulative Log Returns", fontsize=12, color="darkorange")
# for label in ax2.get_yticklabels():
#     label.set_color("darkorange")
# ax2.grid(False)

""" 
    Build Hidden Markov Model 
"""
# df_weekly_1 = df_weekly.loc[(df_weekly.index >= datetime(1940, 1, 1)) & (df_weekly.index <= datetime(1960, 1, 1))]
# df_weekly_1 = df_weekly.loc[(df_weekly.index >= datetime(1960, 1, 1)) & (df_weekly.index <= datetime(1980, 1, 1))]
# df_weekly_1 = df_weekly.loc[(df_weekly.index >= datetime(1980, 1, 1)) & (df_weekly.index <= datetime(2000, 1, 1))]
# df_weekly_1 = df_weekly.loc[(df_weekly.index >= datetime(2000, 1, 1)) & (df_weekly.index <= datetime(2023, 6, 30))]

# smooth_rates = df_weekly_1['log ret ewma'].values
# dates = df_weekly_1.index

smooth_rates = df_weekly['log ret ewma'].values
dates = df_weekly.index

nstates = 3
nperiods = len(smooth_rates)
colors = ["indianred", "cornflowerblue", "darkorange", "gold"]

[pto, pq, p, px, density, f, norm, loglike, mu, sigma] = array_alloc(nperiods, nstates)
rates = smooth_rates.copy()

initialize_hmm(mu, sigma, pq, nstates, rates)

mu_min, mu_max, sigma_min, sigma_max, pq_min, pq_max = initialize_search(rates)
mu_min = mu_min * np.ones(nstates)
mu_max = mu_max * np.ones(nstates)
sigma_min = sigma_min * np.ones(nstates)
sigma_max = sigma_max * np.ones(nstates)
pq_min = pq_min * np.ones([nstates, nstates])
pq_max = pq_max * np.ones([nstates, nstates])

sigma_min_o = sigma_min.copy()
sigma_max_o = sigma_max.copy()
mu_min_o = mu_min.copy()
mu_max_o = mu_max.copy()
pq_min_o = pq_min.copy()
pq_max_o = pq_max.copy()

dmu = (mu_max - mu_min) / 100.0
dsigma = (sigma_max - sigma_min) / 100.0
dpq = (pq_max - pq_min) / 100.0

window = 10
step = 2

k = 0
likeli = [0]

# Initialize the parameters
# mu = np.array([-0.008, 0.001, 0.007])
# sigma = np.array([0.0035, 0.0015, 0.003])
# p_00 = np.exp(-np.log(2)/8)   # low - low
# p_11 = np.exp(-np.log(2)/24)   # mid - mid
# p_22 = np.exp(-np.log(2)/24)   # high - high
# tr = np.array([[p_00, 1 - p_00, 0],
#                [(1 - p_11) * (1/4), p_11, (1 - p_11) * (3/4)],
#                [(1 - p_22) * (1/8), (1 - p_22) * (7/8), p_22]])

# mu = np.array([-0.008, 0.001, 0.007])
# sigma = np.array([0.0035, 0.0015, 0.003])
# p_00 = np.exp(-np.log(2)/52)   # low - low
# p_11 = np.exp(-np.log(2)/(52 * 3))   # mid - mid
# p_22 = np.exp(-np.log(2)/(52 * 2))   # high - high
# pq = np.array([[p_00, (1 - p_00) * (7/8), (1 - p_00) * (1/8)],
#                [(1 - p_11) * (1/4), p_11, (1 - p_11) * (3/4)],
#                [(1 - p_22) * (1/8), (1 - p_22) * (7/8), p_22]])

mu = np.array([-0.008, 0.001, 0.008])
sigma = np.array([0.004, 0.0005, 0.004])
p_00 = np.exp(-np.log(2)/8)   # low - low
p_11 = np.exp(-np.log(2)/24)   # mid - mid
p_22 = np.exp(-np.log(2)/24)   # high - high
tr = np.array([[p_00, 1 - p_00, 0],
               [(1 - p_11) * (1/4), p_11, (1 - p_11) * (3/4)],
               [(1 - p_22) * (1/8), (1 - p_22) * (7/8), p_22]])

# for k in range(10):
while k < 20:
    # search(sigma_min, sigma_max, dsigma, mu_min, mu_max, dmu, pq_min, pq_max, dpq, rates)
    like = regime_fit(nstates, rates, pto, tr, p, px, density, f, norm, loglike, mu, sigma)
    likeli.append(like)
    print("k = " + str(k), ", like = " + str(like))

    dsigma = dsigma / step
    dmu = dmu / step
    dpq = dpq / step

    print("sigma = " + str(sigma))
    dsigma_min = sigma - window * dsigma
    dsigma_max = sigma + (window + 0.01) * dsigma
    sigma_min[dsigma_min > sigma_min_o] = dsigma_min[dsigma_min > sigma_min_o]
    sigma_max[dsigma_max < sigma_max_o] = dsigma_max[dsigma_max < sigma_max_o]

    print("mu = " + str(mu))
    dmu_min = mu - window * dmu
    dmu_max = mu + (window + 0.01) * dmu
    mu_min[dmu_min > mu_min_o] = dmu_min[dmu_min > mu_min_o]
    mu_max[dmu_max < mu_max_o] = dmu_max[dmu_max < mu_max_o]

    print("pq = " + str(pq))
    dpq_min = pq - window * dpq
    dpq_max = pq + (window + 0.01) * dpq
    pq_min[dpq_min > pq_min_o] = dpq_min[dpq_min > pq_min_o]
    pq_max[dpq_max < pq_max_o] = dpq_max[dpq_max < pq_max_o]

    k = k + 1
    if likeli[k] - likeli[k - 1] <= 0.05:
        break

    # ptos = kim_filter(pto, px, nperiods, nstates)
    # results(ptos, likeli)

ptos = kim_filter(pto, tr, px, nperiods, nstates)
# results(ptos, likeli)

# Fit the return time series with the optimized parameters
# rates_all = df_weekly['log ret ewma'].values
test = df_weekly['log ret ewma'].loc[(df_weekly.index >= datetime(1940, 1, 1)) & (df_weekly.index <= datetime(2023, 7, 1))]
rates_all = test.values
nweeks = len(rates_all)
[pto, p, px, density, f, norm, loglike] = array_realloc(nweeks, nstates)

like_all = regime_fit(nstates, rates_all, pto, tr, p, px, density, f, norm, loglike, mu, sigma)

""" Export the results to excel """
# HMM probabilities without kim filter
pto = pd.DataFrame(pto)
pto = pto.iloc[1:,:]
pto.index = df_weekly['log ret ewma'].index
# pto.columns = ['prob 0', 'prob 1']
pto.columns = ['prob 0', 'prob 1', 'prob 2']
pto['prob 0 ewma'] = pto['prob 0'].ewm(span=4, ignore_na=True).mean()
pto['prob 1 ewma'] = pto['prob 1'].ewm(span=4, ignore_na=True).mean()
pto['prob 2 ewma'] = pto['prob 2'].ewm(span=4, ignore_na=True).mean()
pto.to_excel('pto.xlsx')

# HMM probabilities with kim filter
ptos = pd.DataFrame(ptos)
ptos.index = df_weekly['log ret ewma'].index
ptos.columns = ['prob 0', 'prob 1', 'prob 2']
ptos.to_excel('ptos.xlsx')

""" Construct the buy/sell strategy """
# df_final = df_weekly[['ret', 'log ret', '3m tbill ret', '3m tbill log ret']].join(pto)   # without kim filter
df_final = df_weekly[['ret', 'log ret', '3m tbill ret', '3m tbill log ret']].join(ptos)   # with kim filter
df_final.loc[df_final['prob 0'].shift(1) >= 0.5, 'pos'] = 0
df_final.loc[df_final['prob 0'].shift(1) < 0.5, 'pos'] = 1
df_final['pos'][0] = 0
df_final.loc[df_final['pos'] == df_final['pos'].shift(1), 'rebalance'] = 0
df_final.loc[df_final['pos'] != df_final['pos'].shift(1), 'rebalance'] = 1
df_final.loc[df_final['pos'] == 1, 'adj ret'] = df_final['ret']
df_final.loc[df_final['pos'] == 0, 'adj ret'] = df_final['3m tbill ret']
df_final.loc[df_final['pos'] == 1, 'adj log ret'] = df_final['log ret']
df_final.loc[df_final['pos'] == 0, 'adj log ret'] = df_final['3m tbill log ret']
df_final['dollar growth'] = np.cumprod(1 + df_final['adj ret'])
df_final['high value'] = df_final['dollar growth'].cummax()
df_final['dd'] = (df_final['high value'] - df_final['dollar growth']) / df_final['high value']
df_final['cum log ret'] = df_final['log ret'].cumsum()
df_final['cum adj log ret'] = df_final['adj log ret'].cumsum()

# Generate the summary table
stat1 = [df_final['adj ret'].mean() * 52,                                       # annualized returns
         (df_final['adj ret'].std() ** 2) * 52,                                 # annualized volatilities
         df_final['adj ret'].mean() / df_final['adj ret'].std() * np.sqrt(52),  # sharpe ratio
         df_final['dd'].max(),                                                  # maximum drawdown
         df_final['cum adj log ret'][-1],                                       # cumulative log return
         df_final['rebalance'].sum()]                                           # num of rebalances


# Plot the distributions of S&P 500 returns and strategy returns
plt.rcParams["figure.figsize"] = [8, 4]
plt.title('S&P 500 Weekly Returns and HMM Adjusted Weekly Returns')
plt.hist(df_final['ret'].values, bins=100, histtype='bar', density=True, ec='lightsalmon', alpha=0.75, label='S&P 500')
plt.hist(df_final['adj ret'].values, bins=100, histtype='bar', density=True, ec='lightsteelblue', alpha=0.75, label='HMM')
plt.xlim(-0.1, 0.1)
plt.legend()

# Get the quantiles of S&P 500 returns and strategy returns
stat = pd.DataFrame([[np.quantile(df_final['ret'],0), np.quantile(df_final['ret'],0.25), np.quantile(df_final['ret'],0.5),
                      np.quantile(df_final['ret'],0.75), np.quantile(df_final['ret'],1)],
                     [np.quantile(df_final['adj ret'],0), np.quantile(df_final['adj ret'],0.25), np.quantile(df_final['adj ret'],0.5),
                      np.quantile(df_final['adj ret'],0.75), np.quantile(df_final['adj ret'],1)]])
stat.index = ['S&P 500', 'HMM']
stat.columns = ['0th', '25th', '50th', '75th', '100th']


# Compute the transition matrix
# e_pq = np.exp(pq)
# # tr = [e_pq[0,:] / sum(e_pq[0,:]),
# #       e_pq[1,:] / sum(e_pq[1,:])]
# tr = [e_pq[0,:] / sum(e_pq[0,:]),
#       e_pq[1,:] / sum(e_pq[1,:]),
#       e_pq[2,:] / sum(e_pq[2,:])]
# tr = np.array(tr)

# Compute the weight of each distribution
# wt = np.dot(tr.T, [1/2, 1/2])
wt = np.dot(tr.T, [1/3, 1/3, 1/3])

for i in range(100):
    wt = np.dot(tr.T, wt)


# Plot the histogram with the weighted distributions
x = df_weekly['log ret ewma'].values
y = x.reshape(-1,1)

x_axis = x
x_axis.sort()

plt.rcParams["figure.figsize"] = [12, 4]
plt.title('S&P 500 Weekly Log Returns')
# plt.hist(y, bins=100, histtype='bar', density=True, ec='lightsteelblue', alpha=0.5)
# plt.plot(x_axis, wt[0] * stats.norm.pdf(x_axis, mu[0], sigma[0]), c='indianred')
# plt.plot(x_axis, wt[1] * stats.norm.pdf(x_axis, mu[1], sigma[1]), c='royalblue')
# plt.plot(x_axis, wt[2] * stats.norm.pdf(x_axis, mu[2], sigma[2]), c='forestgreen')
plt.plot(x_axis, stats.norm.pdf(x_axis, mu[0], sigma[0]), c='indianred')
plt.plot(x_axis, 0.15 * stats.norm.pdf(x_axis, mu[1], sigma[1]), c='royalblue')
plt.plot(x_axis, stats.norm.pdf(x_axis, mu[2], sigma[2]), c='forestgreen')
plt.axvline(x=mu[0], linestyle='dashed', c='indianred')
plt.axvline(x=mu[1], linestyle='dashed', c='royalblue')
plt.axvline(x=mu[2], linestyle='dashed', c='forestgreen')
plt.xlim(-0.05, 0.025)
# plt.ylim(0, 90)
plt.grid()

"""
    Variance Risk Premium
"""
# Import dataset
data_vix = pd.read_excel(r"C:\Users\Xiaochun.wang\Desktop\Data\Stop Loss Data.xlsx", "vix")

df_vix = data_vix[['Dates', 'SPX Index', 'VIX Index', 'VIX1D Index', 'VIX9D Index', 'VIX3M Index', 'VIX6M Index']].copy()
df_vix.set_index("Dates", inplace=True)
df_vix['vix'] = df_vix['VIX Index'] / 100
df_vix['1D vix'] = df_vix['VIX1D Index'] / 100
df_vix['9D vix'] = df_vix['VIX9D Index'] / 100
df_vix['3M vix'] = df_vix['VIX3M Index'] / 100
df_vix['6M vix'] = df_vix['VIX6M Index'] / 100

df_vix['vix ewma'] = df_vix['vix'].ewm(span=10, ignore_na=True).mean()
df_vix['log vix'] = np.log(df_vix['vix'])

""" Daily Returns """
# Compute daily returns, daily log returns, and cumulative log returns of S&P 500
df_vix['ret'] = df_vix['SPX Index'].pct_change(1)
df_vix['log ret'] = np.log(1 + df_vix['ret'])
df_vix['cum log ret'] = df_vix['log ret'].cumsum()

""" 1M Volatility """
# Compute annualized 1-month exponentially-weighted sigma of S&P 500
df_vix['1M sigma'] = df_vix['ret'].ewm(span=21, ignore_na=True).std() * np.sqrt(252) * 100
df_vix['1M sigma ewma'] = df_vix['1M sigma'].ewm(span=10, ignore_na=True).mean()

""" Variance Risk Premium """
# Compute the spread of implied and realized volatility
df_vix['vrp'] = (df_vix['VIX Index'] - df_vix['1M sigma']) / 100
df_vix['vrp ewma'] = df_vix['vrp'].ewm(span=10, ignore_na=True).mean()

# Select the time period
#df_daily = df_daily.loc[(df_daily.index >= start_date) & (df_daily.index <= end_date)]

# Plot the implied and realized volatility
plt.figure(figsize=(20,8))
plt.subplot(2, 1, 1)
plt.title('Implied and Realized Volatility')
plt.plot(df_vix['VIX Index'], label='Implied Vol')
plt.plot(df_vix['1M sigma'], label='Realized Vol')
for i in range(5, len(recession_regime)):
    plt.axvspan(recession_regime['start'][i], recession_regime['end'][i], color='lightgrey', alpha=0.8)
plt.xlabel('Dates')
plt.ylabel('Volatility')
plt.legend()

plt.rcParams["figure.figsize"] = [17, 4]
fig, ax1 = plt.subplots()
ax1.plot(df_vix['vrp ewma'], lw=2, color="steelblue")
ax1.fill_between(df_vix.index, df_vix['vrp ewma'], color='skyblue', alpha=0.75)
ax1.set_ylabel(r'Variance Risk Premium', fontsize=12, color='steelblue')
for label in ax1.get_yticklabels():
    label.set_color('steelblue')
#ax1.grid(False)

ax2 = ax1.twinx()
ax2.plot(df_vix['cum log ret'], color='chocolate')
ax2.set_ylabel(r'Cumulative Log Returns', fontsize=12, color='chocolate')
for label in ax2.get_yticklabels():
    label.set_color('chocolate')
ax2.grid(False)

plt.tight_layout()
plt.show()

""" 
    VIX Term Premia
"""
df_vix['1M-3M vix premia'] = df_vix['vix'] - df_vix['3M vix']

# Plot VIX
plt.title('VIX')
plt.plot(df_vix['1D vix'], label='1D VIX')
plt.plot(df_vix['9D vix'], label='9D VIX')
plt.plot(df_vix['vix'], label='VIX')
plt.plot(df_vix['3M vix'], label='3M VIX')
plt.plot(df_vix['6M vix'], label='6M VIX')
for i in range(5, len(recession_regime)):
    plt.axvspan(recession_regime['start'][i], recession_regime['end'][i], color='lightgrey', alpha=0.8)
plt.xlabel('Dates')
plt.ylabel('VIX')
plt.legend()
plt.show()

# Plot VIX term Premia
plt.rcParams["figure.figsize"] = [17, 4]
fig, ax1 = plt.subplots()
ax1.plot(df_vix['1M-3M vix premia'].dropna(), lw=2, color="steelblue")
ax1.set_ylabel(r'1M-3M VIX Term Premia', fontsize=12, color='steelblue')
for label in ax1.get_yticklabels():
    label.set_color('steelblue')
#ax1.grid(False)

ax2 = ax1.twinx()
ax2.plot(df_vix['cum log ret'][df_vix['1M-3M vix premia'].dropna().index], color='chocolate')
ax2.set_ylabel(r'Cumulative Log Returns', fontsize=12, color='chocolate')
for label in ax2.get_yticklabels():
    label.set_color('chocolate')
ax2.grid(False)

plt.tight_layout()
plt.show()

""" 
    HMM - Training and Testing
"""
#train = df_vix['vrp ewma'].dropna()
#train = df_vix['vrp ewma'][df_vix.index <= datetime(2020, 1, 1)].dropna()
#train = df_vix['log vix'].dropna()
train = df_vix['log vix'][df_vix.index <= datetime(2020, 1, 1)].dropna()
smooth_rates = train.values
dates = train.index

# smooth_rates = df_vix['1M-3M vix premia'].dropna().values
# dates = df_vix['1M-3M vix premia'].dropna().index

nstates = 2
nperiods = len(smooth_rates)
colors = ["indianred", "cornflowerblue", "darkorange", "gold"]

[pto, pq, p, px, density, f, norm, loglike, mu, sigma] = array_alloc(nperiods, nstates)
rates = smooth_rates.copy()

initialize_hmm(mu, sigma, pq, nstates, rates)

mu_min, mu_max, sigma_min, sigma_max, pq_min, pq_max = initialize_search(rates)
mu_min = mu_min * np.ones(nstates)
mu_max = mu_max * np.ones(nstates)
sigma_min = sigma_min * np.ones(nstates)
sigma_max = sigma_max * np.ones(nstates)
pq_min = pq_min * np.ones([nstates, nstates])
pq_max = pq_max * np.ones([nstates, nstates])

sigma_min_o = sigma_min.copy()
sigma_max_o = sigma_max.copy()
mu_min_o = mu_min.copy()
mu_max_o = mu_max.copy()
pq_min_o = pq_min.copy()
pq_max_o = pq_max.copy()

dmu = (mu_max - mu_min) / 100.0
dsigma = (sigma_max - sigma_min) / 100.0
dpq = (pq_max - pq_min) / 100.0

window = 10
step = 2

k = 0
likeli = [0]

while k < 20:
    search(sigma_min, sigma_max, dsigma, mu_min, mu_max, dmu, pq_min, pq_max, dpq, rates)
    like = regime_fit(nstates, rates, pto, tr, p, px, density, f, norm, loglike, mu, sigma)
    likeli.append(like)
    print("k = " + str(k), ", like = " + str(like))

    dsigma = dsigma / step
    dmu = dmu / step
    dpq = dpq / step

    print("sigma = " + str(sigma))
    dsigma_min = sigma - window * dsigma
    dsigma_max = sigma + (window + 0.01) * dsigma
    sigma_min[dsigma_min > sigma_min_o] = dsigma_min[dsigma_min > sigma_min_o]
    sigma_max[dsigma_max < sigma_max_o] = dsigma_max[dsigma_max < sigma_max_o]

    print("mu = " + str(mu))
    dmu_min = mu - window * dmu
    dmu_max = mu + (window + 0.01) * dmu
    mu_min[dmu_min > mu_min_o] = dmu_min[dmu_min > mu_min_o]
    mu_max[dmu_max < mu_max_o] = dmu_max[dmu_max < mu_max_o]

    print("pq = " + str(pq))
    dpq_min = pq - window * dpq
    dpq_max = pq + (window + 0.01) * dpq
    pq_min[dpq_min > pq_min_o] = dpq_min[dpq_min > pq_min_o]
    pq_max[dpq_max < pq_max_o] = dpq_max[dpq_max < pq_max_o]

    k = k + 1
    if likeli[k] - likeli[k - 1] <= 0.05:
        break

    # ptos = kim_filter(pto, px, nperiods, nstates)
    # results(ptos, likeli)

ptos = kim_filter(pto, tr, px, nperiods, nstates)

ptos = pd.DataFrame(ptos)
ptos.index = dates
ptos.columns = ['prob 0', 'prob 1']
#pto.columns = ['prob 0', 'prob 1', 'prob 2']

#df_final = df_vix[['ret', 'log ret', 'vrp']].join(ptos)
df_final = df_vix[['ret', 'log ret', 'log vix']].join(ptos)
df_final = df_final.dropna()

plt.figure(figsize=(20,4))
plt.title('HMM - VIX')
plt.stackplot(df_final.index, df_final['prob 0'], df_final['prob 1'], colors=colors, edgecolor='none', alpha=0.5)
plt.grid(False)
plt.gca().twinx().plot(df_final['vix'])
plt.grid(False)
plt.show()

plt.figure(figsize=(20,4))
plt.title('HMM - Variance Risk Premium')
plt.stackplot(df_final.index, df_final['prob 0'], df_final['prob 1'], colors=colors, edgecolor='none', alpha=0.5)
plt.grid(False)
plt.gca().twinx().plot(df_final['vrp'])
plt.grid(False)
plt.show()

ptos = pd.DataFrame(ptos)
ptos.index = dates
ptos.columns = ['prob 0', 'prob 1']
#pto.columns = ['prob 0', 'prob 1', 'prob 2']

df_final = df_vix[['ret', 'log ret', '1M-3M vix premia']].join(ptos)
df_final = df_final.dropna()

plt.figure(figsize=(20,4))
plt.title('HMM - VIX Term Premia')
plt.stackplot(df_final.index, df_final['prob 0'], df_final['prob 1'], colors=colors, edgecolor='none', alpha=0.5)
plt.grid(False)
plt.gca().twinx().plot(df_final['1M-3M vix premia'])
plt.grid(False)
plt.show()

