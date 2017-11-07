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
