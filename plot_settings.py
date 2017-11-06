import numpy as np
import scipy as sp
import matplotlib
from matplotlib import gridspec
import pylab as pl
import SMC_ABC_recoded as smc

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
figheight_single = 5

colormap_conditions = matplotlib.cm.viridis
colors_conditions = colormap_conditions(sp.linspace(0, 1, 8))
colormap_benchmark = matplotlib.cm.Set1
colors_benchmark = colormap_benchmark(sp.linspace(0, 1, 10))
color_bg = 'darkgrey'
alpha_bg = 0.025

line_wdth = 3

rr = np.arange(0,125+1,5)
win = np.zeros((len(rr),2))
win[:,0] = rr
win[:,1] = rr+125


# helper functions
def acf_sliding_window(data,windows = win):
    TS = []
    ACF = []
    for ww in windows:
        dd = data[int(ww[0]):int(ww[1]),:]
        acf = AutocorrelateAllData(dd,1)
        TS.append(dd)
        ACF.append(acf)
    return TS,ACF

def calc_histogram(data):
    bins = np.logspace(0, 5, 50)
    n = data.shape[0] * data.shape[1]
    hist = np.histogram(data.reshape(n, ), bins = bins)
    Hist = np.array(hist[0], dtype = float)
    return bins[0:len(bins) - 1], Hist / n
