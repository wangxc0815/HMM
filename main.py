import sys
import timeit
from datetime import datetime
from datetime import timedelta

from dateutil.relativedelta import relativedelta
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.mixture import GaussianMixture
from warnings import simplefilter

np.set_printoptions(threshold=sys.maxsize)
#pd.set_option('display.float_format', lambda x: '%.3f' % x)

pd.options.mode.chained_assignment = None
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


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

def winsorize(series, lower=0.5, upper=99.5):
    """ windsorize """
    pc_low  = np.percentile(series, lower)
    pc_high = np.percentile(series, upper)
    wseries = series.copy()
    wseries[wseries<pc_low] = pc_low
    wseries[wseries>pc_high] = pc_high
    return wseries

def smooth_series(wseries, seed=7, hl=1):
    """ smoothing """
    a = np.exp(-np.log(2)/hl)
    b = 1 - a

    update = np.mean(wseries[:seed])
    smooth = list()
    for element in wseries:
        update = update*a + element*b
        smooth.append(update)
    return np.array(smooth)

def initialize_hmm(mu, sigma, pq, nstates, series):
    """ place holder parms for testing """
    std = np.std(series)

    for k in range(nstates):
        sigma[k] = std
        pq[k, k] = 3.0
        mu[k] = np.percentile(series, 100 * (k + 1) / (nstates + 1))

def regime_fit(nstates, smooth, pto, pq, p, px, density, f, norm, loglike, mu, sigma):
    nweeks = len(smooth)

    tpi = np.sqrt(2.0 * np.pi)
    nrm = 1.0 / (tpi * sigma)

    zpq = np.exp(pq)
    pz = (zpq.T / zpq.sum(axis=1))
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
    pq_max =  10

    return (mu_min, mu_max, sigma_min, sigma_max, pq_min, pq_max)

def kim_filter(pto, px, nweeks, nstates):
    ptos = np.zeros([nweeks, nstates])
    ps = np.zeros([nweeks, nstates*nstates])
    zpq = np.exp(pq)
    pz = (zpq.T / zpq.sum(axis=1))

    ptos[nweeks-1,:] = pto[nweeks,:]
    for t in range(nweeks-1, 0, -1):
        for k in range(nstates):
            ps[t,k*nstates:((k+1)*nstates)] = pto[t,k] * ptos[t,:] * pz[:,k] / px[t,:]
        for k in range(nstates):
            ptos[t-1,k] = sum(ps[t,k*nstates:(k+1)*nstates])

    return ptos

def search(sigma_min, sigma_max, dsigma, mu_min, mu_max, dmu, pq_min, pq_max, dpq, series):

    for m in range(3):
        # #sigma
        # for state in range(nstates):
        #     #ic = 0
        #     mlike = regime_fit(nstates, series, pto, pq, p, px, density, f, norm, loglike, mu, sigma)
        #     best = sigma[state]
        #     for sig in np.arange(sigma_min[state], sigma_max[state], dsigma[state]):
        #         sigma[state] = sig
        #         like = regime_fit(nstates, series, pto, pq, p, px, density, f, norm, loglike, mu, sigma)
        #         if like>mlike:
        #             mlike = like
        #             best = sig
        #         #ic = ic + 1
        #     sigma[state] = best
        #     maxlike = regime_fit(nstates, series, pto, pq, p, px, density, f, norm, loglike, mu, sigma)
        #     #print("sigma:", state, ic)


        for state in range(nstates):
            for element in range(0, nstates):
                #ic = 0
                best = pq[state][element]
                mlike = regime_fit(nstates, series, pto, pq, p, px, density, f, norm, loglike, mu, sigma)
                for update in np.arange(pq_min[state, element], pq_max[state, element], dpq[state][element]):
                    pq[state][element] = update
                    like = regime_fit(nstates, series, pto, pq, p, px, density, f, norm, loglike, mu, sigma)
                    if like>mlike:
                        mlike = like
                        best = update
                    #ic = ic + 1
                #print("pq:", state, element, ic)
                pq[state][element] = best
                maxlike = regime_fit(nstates, series, pto, pq, p, px, density, f, norm, loglike, mu, sigma)

    # #mu
    # for state in range(nstates):
    #     mlike = regime_fit(nstates, series, pto, pq, p, px, density, f, norm, loglike, mu, sigma)
    #     best = mu[state]
    #     #ic = 0
    #     for mux in np.arange(mu_min[state], mu_max[state], dmu[state]):
    #         mu[state] = mux
    #         like = regime_fit(nstates, series, pto, pq, p, px, density, f, norm, loglike, mu, sigma)
    #         if like>mlike:
    #             mlike = like
    #             best = mux
    #         #ic = ic + 1
    #     #print("mu:", state, ic)
    #     mu[state] = best
    #     maxlike = regime_fit(nstates, series, pto, pq, p, px, density, f, norm, loglike, mu, sigma)

# mu, sigma, pq

def results(ptos, likeli):
    fig, ax = plt.subplots(2, 1, figsize=(12, 12))
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



""" Inputs """
start_date = datetime(1975, 1, 1)
end_date = datetime(2023, 6, 30)

"""    
   Daily Frequency
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
df_daily['log ret ewma'] = df_daily['log ret 1'].ewm(span=42, ignore_na=True).mean()

""" 1M Volatility """
# Compute annualized 21-day volatilities of S&P 500
df_daily['1-month vol'] = df_daily['log ret 1'].rolling(21).var() * (252 ** (1/2))

# Compute the exponential weighted moving average of 1-month volatility
df_daily['1-month vol ewma'] = df_daily['1-month vol'].ewm(span=42, ignore_na=True).mean()

# Select the time period
df_daily = df_daily.loc[(df_daily.index >= start_date) & (df_daily.index <= end_date)]

# Compute the exponential weighted moving average of 1-month volatility
df_daily['1-month vol ewma'] = df_daily['1-month vol'].ewm(span=42, ignore_na=True).mean()

# Select the time period
df_daily = df_daily.loc[(df_daily.index >= start_date) & (df_daily.index <= end_date)]


"""
    Weekly Frequency
"""
# Import dataset
data_weekly = pd.read_excel(r"C:\Users\Xiaochun.wang\Desktop\Data\Stop Loss Data.xlsx", "weekly")

df_weekly = data_weekly[['Dates', 'SPX']].copy()
df_weekly.set_index("Dates", inplace=True)

""" Weekly Returns """
# Compute weekly returns of S&P 500
df_weekly['ret'] = df_weekly['SPX'].pct_change(1)

# Winsorize weekly returns
df_weekly['ret 1'] = df_weekly['ret']
threshold_lower = np.nanpercentile(df_weekly['ret'], 5)
df_weekly.loc[df_weekly['ret'] <= threshold_lower, 'ret 1'] = threshold_lower
threshold_upper = np.nanpercentile(df_weekly['ret'], 95)
df_weekly.loc[df_weekly['ret'] >= threshold_upper, 'ret 1'] = threshold_upper

# Compute the exponential weighted moving average of daily returns
df_weekly['ret ewma'] = df_weekly['ret 1'].ewm(span=8, ignore_na=True).mean()

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
df_weekly['log ret ewma'] = df_weekly['log ret 1'].ewm(span=8, ignore_na=True).mean()

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


""" Hidden Markov Model """
nperiods = 52 * 20
nstates = 3
colors = ["indianred", "cornflowerblue", "darkorange", "gold"]

prob_low = []
pos = [0]

#with open('output.txt', 'w') as f:
#sys.stdout = f

start = timeit.timeit()

#for i in df_weekly.index[:1]:
for i in df_weekly.index[:(-nperiods + 1)]:
    print("Starting date is " + str(i))
    i_end = i + timedelta(weeks=nperiods)
    temp = df_weekly.loc[(df_weekly.index >= i) & (df_weekly.index < i_end)]

    smooth_rates = temp['log ret ewma'].values
    #nperiods = len(smooth_rates)

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

    dmu = (mu_max - mu_min) / 50.0
    dsigma = (sigma_max - sigma_min) / 50.0
    dpq = (pq_max - pq_min) / 50.0

    window = 10
    step = 2

    k = 0
    likeli = [0]

    while k < 20:
    #for k in range(6):
        search(sigma_min, sigma_max, dsigma, mu_min, mu_max, dmu, pq_min, pq_max, dpq, rates)
        like = regime_fit(nstates, rates, pto, pq, p, px, density, f, norm, loglike, mu, sigma)
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
        if likeli[k] - likeli[k - 1] <= 0.1:
            break
        #ptos = kim_filter(pto, px, nperiods, nstates)

    ptos = kim_filter(pto, px, nperiods, nstates)
    ptos = pd.DataFrame(ptos)
    ptos.index = temp.index
    ptos.columns = ['p0', 'p1', 'p2']
    sumprod = np.dot(temp['log ret ewma'], ptos)
    ind = sumprod.argmin()
    i_prob_low = ptos.tail(1).values[0, ind]
    prob_low.append(i_prob_low)
    i_pos = 0 if i_prob_low >= 0.6 else 1
    pos.append(i_pos)

end = timeit.timeit()
process_time = start - end
print('The total processing time is ' + str(process_time))

#sys.stdout = sys.__stdout__

prob_low = pd.DataFrame(prob_low)
pos = pos[:-1]
pos = pd.DataFrame(pos)
df_final = pd.concat([prob_low, pos], axis=1)
df_final.index = df_weekly.index[(nperiods-1):]
df_final.columns = ['prob low', 'pos']

df_final = df_final.join(df_weekly[['ret', 'log ret']])
df_final.loc[df_final['pos'] == 1, 'adj ret'] = df_final['ret']
df_final.loc[df_final['pos'] == 0, 'adj ret'] = 0
df_final.loc[df_final['pos'] == 1, 'adj log ret'] = df_final['log ret']
df_final.loc[df_final['pos'] == 0, 'adj log ret'] = 0
df_final = df_final.dropna()

df_final.to_excel("HMM (1975 - 2013, winodw = 20yr).xlsx")

# plt.figure(figsize=(20, 4))
# plt.title('HMM - S&P 500 (2-Month EWMA Weekly Log Returns)')
# plt.stackplot(dates, ptos.T, colors=colors, edgecolor='none', alpha=0.5)
# plt.grid(False)
# plt.gca().twinx().plot(df_weekly['log ret'])
# plt.grid(False)
# plt.show()

# Export the data to excel
# ptos = pd.DataFrame(ptos)
# ptos.index = df_weekly.index
# ptos.columns = ['prob 0', 'prob 1', 'prob 2']
# ptos.to_excel('HMM - Returns (1940 - 1960).xlsx')