from os import listdir
import glob

import numpy as np
import scipy as sp
import scipy.signal as sps
import pandas as pd

import singleCellFunctions as scf

# reporter structure
speed = 2520.
promoterLength = 88000
stemStart = 1300
nStems = 24
lenStem = 60

# E2 concentrations in pM
conc = [3., 5., 10., 20., 1000.]
# corresponding intensities of single transcripts
TranscriptInt = [41.25, 30.0, 26.25, 26.25, 26.25]
# steepness of the assumed sigmoid functions that create a single transcript
alpha = 3

# calculate the delays
tauOn = (stemStart + nStems * lenStem) / speed
tauOff = promoterLength / speed


# functions
# load the data
def LoadData(path):
    '''
    Load data using pandas data frames
    :param path: filename
    :return: a list of pandas data frames
    '''
    Data = []
    for data in path:
        dat = pd.read_table(data,
                            sep = '\s*',
                            skiprows = 1,
                            index_col = 1,
                            engine = 'python')
        Data.append(dat)
    return Data


def ExtractData(Data, column):
    '''
    Write the loaded data from pandas data frames into one single 2D array
    :param Data: list of data frames
    :param column: column of the data frames that is extracted
    :return: 2d arrarys: data and time
    '''
    res = np.zeros((Data[0].shape[0], len(Data)))
    time = np.zeros((Data[0].shape[0], len(Data)))
    count = 0
    while count < len(Data):
        res[:, count] = Data[count].ix[:, column]
        time[:, count] = Data[count].index
        count = count + 1
    return res, time


def ImportData(path, column):
    '''
    read in data from multiple text files and write one desired column into one output array.
    Both for actual data and mock sites
    :param path: path to the file(s)
    :param column: column of the original data to extract
    :return: two 2D arrays, actual data and mock data
    '''

    # creat lists of files in the path directory
    dataFile = glob.glob(path + '*Int.dat')
    mockFile = glob.glob(path + '*mock.dat')

    # load the actual data
    Data = LoadData(dataFile)
    Mock = LoadData(mockFile)

    # extract the desired column
    DataTC, TimeData = ExtractData(Data, column)
    MockTC, TimeMock = ExtractData(Mock, column)

    return DataTC, TimeData, MockTC, TimeMock


def SmoothData(data, kernelSize):
    index = np.arange(0, data.shape[1])
    dataSmoothed = np.zeros((data.shape[0], data.shape[1]))
    for n in index:
        dataSmoothed[:, n] = sps.medfilt(data[:, n], kernelSize)
    return dataSmoothed


def AutocorrelateAllData(DataTC, deltaT):
    '''
    Calculate the autocorrelations for all extracted signals
    :param DataTC: 2D numpy array with the time courses in columns
    :param deltaT: imaging time interval
    :return: 2d numpy array with the calculated autocorrelations in columns
    '''
    res = np.zeros((DataTC.shape))
    count = 0
    while count < DataTC.shape[1]:
        res[:, count] = scf.AutoCorrelate(DataTC[:, count], deltaT)[1]
        count = count + 1
    return res




def KStest(data, mock):
    pval = sp.stats.ks_2samp(data, mock)[1]
    return pval


def KStestMulti(Data, Mock):
    nn = np.arange(0, Data.shape[1])
    pvals = []
    for n in nn:
        pvals.append(KStest(Data[:, n], Mock[:, n]))
    return np.array(pvals)


def singleTranscript(t_i, a, alpha, tauOn, tauOff, tt):
    s = a / (1 + sp.exp(-alpha * (tt - t_i - tauOn))) - a / (1 + sp.exp(-alpha * (tt - t_i - tauOff)))
    return s


