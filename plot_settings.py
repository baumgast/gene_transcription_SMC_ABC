import numpy as np
import scipy as sp
import matplotlib
from matplotlib import gridspec
import pylab as pl

matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 14
matplotlib.rcParams['ytick.labelsize'] = 14
matplotlib.rcParams['legend.fontsize'] = 11

matplotlib.rcParams['font.family'] = ['sans-serif']
matplotlib.rcParams['font.sans-serif'] = ['Arial']
matplotlib.rcParams['text.usetex'] = False
matplotlib.rcParams['axes.grid'] = False

matplotlib.rcParams['svg.fonttype'] = 'none'

fig_width = 11
fig_height_single = 5

colormap_conditions = matplotlib.cm.viridis
colors_conditions = colormap_conditions(sp.linspace(0, 1, 8))
colormap_benchmark = matplotlib.cm.Set1
colors_benchmark = colormap_benchmark(sp.linspace(0, 1, 10))
color_bg = 'darkgrey'
alpha_bg = 0.025

line_wdth = 3

# ACF window positions for steady state data
rr = np.arange(0, 125 + 1, 5)
win = np.zeros((len(rr), 2))
win[:, 0] = rr
win[:, 1] = rr + 125


# helper functions
def calc_histogram(data):
    bins = np.logspace(0, 5, 50)
    n = data.shape[0] * data.shape[1]
    hist = np.histogram(data.reshape(n, ), bins=bins)
    Hist = np.array(hist[0], dtype=float)
    return bins[0:len(bins) - 1], Hist / n


def AutocorrelateAllData(DataTC, deltaT=1):
    res = np.empty((DataTC.shape))
    count = 0
    while count < DataTC.shape[1]:
        if np.sum(DataTC[:, count] == 0) == DataTC.shape[0]:
            rr = np.zeros(res.shape[0]) + 10
            res[:, count] = rr
        else:
            res[:, count] = AutoCorrelate(DataTC[:, count], deltaT)[1]
        count = count + 1
    return res


def AutoCorrelate(signal, deltaT):
    signal = signal - np.mean(signal)
    res = np.correlate(signal, signal, mode='full')
    res = res / np.max(res)
    out = np.empty((2, len(signal)))
    out[0] = np.arange(0, len(signal)) * deltaT

    out[1] = res[np.argmax(res):]
    return out


def acf_sliding_window(data, windows=win):
    TS = []
    ACF = []
    for ww in windows:
        dd = data[int(ww[0]):int(ww[1]), :]
        acf = AutocorrelateAllData(dd, 1)
        TS.append(dd)
        ACF.append(acf)
    return TS, ACF


def average_acf(ACF):
    res = np.zeros((len(ACF), ACF[0].shape[0]))
    for ii, acf in enumerate(ACF):
        res[ii] = acf.mean(axis=1)
    return res


def bootstrap_acf_hl(acf, bins):
    acf_hl_hist = np.zeros((len(acf), len(bins) - 1))
    for ii in np.arange(0, len(acf)):
        hl = ACF_HL_multi(acf[ii])
        hist = np.histogram(hl, bins, normed=1)[0]
        acf_hl_hist[ii] = hist
    return acf_hl_hist


def bootstrap_acf_lag(acf, bins):
    acf_lag_hist = np.zeros((len(acf), len(bins) - 1))
    for ii in np.arange(0, len(acf)):
        lag = acf[ii][1, :]
        hist = np.histogram(lag, bins, normed=1)[0]
        acf_lag_hist[ii] = hist
    return acf_lag_hist


def ACF_HL(acf):
    ind_below = np.where(acf <= 0.5)[0]
    x_low = ind_below[0]
    x_high = ind_below[0] - 1
    acf_low = acf[x_low]
    acf_high = acf[x_high]

    slope = acf_low - acf_high
    inter = acf_low - slope * x_low
    hl = (0.5 - inter) / slope
    return hl


def ACF_HL_multi(ACF):
    res = np.zeros(ACF.shape[1])
    index = np.arange(0, ACF.shape[1])
    for ii in index:
        res[ii] = ACF_HL(ACF[:, ii])
    return res


def nice_boxplots(bp):
    for box in bp['boxes']:
        box.set(color='dimgrey', lw=1)
        box.set(facecolor=color_bg)

    for whisker in bp['whiskers']:
        whisker.set(color='dimgrey', linewidth=1, linestyle='solid')

    for cap in bp['caps']:
        cap.set(color='dimgrey', linewidth=1)

    for median in bp['medians']:
        median.set(color='dimgrey', linewidth=1)

    for flier in bp['fliers']:
        flier.set(marker='.', color='none', alpha=0.5)
