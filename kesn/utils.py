from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
from scipy import stats as st


def calc_sigam(x):
    km = euclidean_distances(x, x)
    d_max = np.max(km) / np.sqrt(2 * x.shape[0])
    return d_max


def rho_w(w: np.ndarray):
    [p, _] = np.linalg.eig(w)
    r = np.max(np.abs(p))
    return r


def bic_value(pars, rss, n_sample):
    return pars * np.log(n_sample) - np.log(rss / n_sample)


def get_interval(x):
    xm = np.mean(x, axis=1)
    ste = st.sem(x, axis=1)
    interval = st.norm.interval(alpha=0.95, loc=xm, scale=ste)
    return xm, interval


