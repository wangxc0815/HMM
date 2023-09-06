import numpy as np
import pandas as pd
from datetime import datetime
from datetime import timedelta
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

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


# def read_data(path, file, sheet, series):
#     # raw = pd.read_excel(path+file, sheet_name=sheet)
#     raw = pd.read_excel(file, sheet_name=sheet)
#     raw.set_index("Dates", inplace=True)
#     spx = raw[series]

#     dates = [element.date() for element in spx.index]
#     spx = spx.reindex(dates)

#     return spx


# def rates_from_levels(values):
#     series = values[1:]/values[:-1] - 1
#     return series


def winsorize(series, lower=0.5, upper=99.5):
    """ windsorize """
    pc_low = np.percentile(series, lower)
    pc_high = np.percentile(series, upper)
    wseries = series.copy()
    wseries[wseries < pc_low] = pc_low
    wseries[wseries > pc_high] = pc_high
    return wseries


def smooth_series(wseries, seed=7, hl=1):
    """ smoothing """
    a = np.exp(-np.log(2) / hl)
    b = 1 - a

    update = np.mean(wseries[:seed])
    smooth = list()
    for element in wseries:
        update = update * a + element * b
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
    pq_max = 10

    return (mu_min, mu_max, sigma_min, sigma_max, pq_min, pq_max)


def kim_filter(pto, px, nweeks, nstates):
    ptos = np.zeros([nweeks, nstates])
    ps = np.zeros([nweeks, nstates * nstates])
    zpq = np.exp(pq)
    pz = (zpq.T / zpq.sum(axis=1))

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
            mlike = regime_fit(nstates, series, pto, pq, p, px, density, f, norm, loglike, mu, sigma)
            best = sigma[state]
            for sig in np.arange(sigma_min, sigma_max, dsigma):
                sigma[state] = sig
                like = regime_fit(nstates, series, pto, pq, p, px, density, f, norm, loglike, mu, sigma)
                if like > mlike:
                    mlike = like
                    best = sig
            sigma[state] = best
            maxlike = regime_fit(nstates, series, pto, pq, p, px, density, f, norm, loglike, mu, sigma)

        for state in range(nstates):
            if state == 0:
                for element in range(nstates - 1):
                    best = pq[state][element]
                    mlike = regime_fit(nstates, series, pto, pq, p, px, density, f, norm, loglike, mu, sigma)
                    for update in np.arange(pq_min, pq_max, dpq):
                        pq[state][element] = update
                        like = regime_fit(nstates, series, pto, pq, p, px, density, f, norm, loglike, mu, sigma)
                        if like > mlike:
                            mlike = like
                            best = update
                    pq[state][element] = best
                    maxlike = regime_fit(nstates, series, pto, pq, p, px, density, f, norm, loglike, mu, sigma)

            else:
                for element in range(1, nstates):
                    best = pq[state][element]
                    mlike = regime_fit(nstates, series, pto, pq, p, px, density, f, norm, loglike, mu, sigma)
                    for update in np.arange(pq_min, pq_max, dpq):
                        pq[state][element] = update
                        like = regime_fit(nstates, series, pto, pq, p, px, density, f, norm, loglike, mu, sigma)
                        if like > mlike:
                            mlike = like
                            best = update
                    pq[state][element] = best
                    maxlike = regime_fit(nstates, series, pto, pq, p, px, density, f, norm, loglike, mu, sigma)

    for state in range(nstates):
        mlike = regime_fit(nstates, series, pto, pq, p, px, density, f, norm, loglike, mu, sigma)
        best = mu[state]
        for mux in np.arange(mu_min, mu_max, dmu):
            mu[state] = mux
            like = regime_fit(nstates, series, pto, pq, p, px, density, f, norm, loglike, mu, sigma)
            if like > mlike:
                mlike = like
                best = mux
        mu[state] = best
        maxlike = regime_fit(nstates, series, pto, pq, p, px, density, f, norm, loglike, mu, sigma)