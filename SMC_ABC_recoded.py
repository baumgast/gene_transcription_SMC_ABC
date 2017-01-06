import numpy as np
import scipy as sp
from scipy import stats
import ipyparallel
import glob
import time

# create a load balanced view to run stochastic simulations on multiple cores using ipyparallel.
rc = ipyparallel.Client()
rc.block = True
view = rc.direct_view()


'''
All model topologies as list. Each list entry is a list itself and the three integers indicate the number of ON states,
the number of OFF states and the source of extrinsic noise. In total 5 topologies (2,3,4, and two ten state models) and
eight sources of extrinsic noise are included yielding 40 different models
'''
models = [[1, 1, 0], [1, 1, 1], [1, 1, 2], [1, 1, 3], [1, 1, 4], [1, 1, 5], [1, 1, 6], [1, 1, 7],
          [1, 2, 0], [1, 2, 1], [1, 2, 2], [1, 2, 3], [1, 2, 4], [1, 2, 5], [1, 2, 6], [1, 2, 7],
          [2, 2, 0], [2, 2, 1], [2, 2, 2], [2, 2, 3], [2, 2, 4], [2, 2, 5], [2, 2, 6], [2, 2, 7],
          [1, 9, 0], [1, 9, 1], [1, 9, 2], [1, 9, 3], [1, 9, 4], [1, 9, 5], [1, 9, 6], [1, 9, 7],
          [2, 8, 0], [2, 8, 1], [2, 8, 2], [2, 8, 3], [2, 8, 4], [2, 8, 5], [2, 8, 6], [2, 8, 7]]


'''
Allowed transitions in model space during SMC ABC model fitting. The fourty different entries list the allowed transitions
for each included model. The numbers are the indicies for the model list.
'''
model_transitions = [[1, 2, 3, 4, 8],
                     [0, 2, 3, 4, 5, 6, 7, 9],
                     [0, 1, 3, 4, 5, 10],
                     [0, 1, 2, 4, 5, 11],
                     [0, 1, 2, 3, 7, 12],
                     [1, 2, 6, 7, 13],
                     [1, 3, 5, 7, 14],
                     [1, 4, 5, 6, 15],
                     [9, 10, 11, 12, 0, 16, 24],
                     [8, 10, 11, 12, 13, 14, 15, 1, 17, 25],
                     [8, 9, 11, 12, 13, 2, 18, 26],
                     [8, 9, 10, 12, 14, 3, 19, 27],
                     [8, 9, 10, 11, 15, 4, 20, 28],
                     [8, 10, 14, 15, 5, 21, 29],
                     [9, 11, 13, 15, 6, 22, 30],
                     [9, 12, 13, 14, 7, 23, 31],
                     [17, 18, 19, 20, 8, 32],
                     [16, 17, 18, 19, 20, 21, 22, 23, 9, 33],
                     [16, 17, 19, 20, 21, 10, 34],
                     [16, 17, 18, 20, 22, 11, 35],
                     [16, 17, 18, 19, 23, 12, 36],
                     [17, 18, 22, 23, 13, 37],
                     [17, 19, 21, 23, 14, 38],
                     [17, 20, 21, 22, 15, 39],
                     [25, 26, 27, 28, 8],
                     [24, 25, 26, 27, 28, 29, 30, 9],
                     [24, 25, 27, 28, 29, 10],
                     [24, 25, 26, 28, 30, 11],
                     [24, 25, 26, 27, 31, 12],
                     [25, 26, 30, 31, 13],
                     [25, 27, 29, 31, 14],
                     [25, 28, 29, 30, 15],
                     [33, 34, 35, 36, 16],
                     [32, 34, 35, 36, 37, 38, 39, 17],
                     [32, 33, 35, 36, 37],
                     [32, 33, 34, 36, 39, 19],
                     [32, 33, 34, 35, 39, 20],
                     [33, 34, 37, 39, 21],
                     [33, 35, 37, 39, 22],
                     [31, 36, 37, 38, 23]]

'''
Width of the log normal perturbation kernels for the main model parameters: burst size, on time and off time
'''
sigmas = [0.5, 0.2, 0.6]
# fit with T as local variable
# sigmas_global = [0.5,0.2,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6]
# fit with burst as local variable
sigmas_global = [0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.4]
# fit inhibitors with DMSO control, T local
# sigmas_global = [0.2,0.2,0.4,0.4]
# fit inhibitors with DMSO control, burst local
# sigmas_global = [0.2,0.2,0.2,0.4]

'''
Experimental parameters:
deltaT: imaging time interval
alpha: scaling factor between number of RNAs and light intensity
max_lag: index up to which the global autocorrelation function between simulation and data are compared.
'''
deltaT = 3
alpha = 32.8
alpha = 33.15
alpha = 23.3
alpha_global = [33.15,33.15,33.15,33.15,33.15,33.15,23.3,23.3]
max_lag = 17

# upload the used data instead of the multi column CSV files
column = 9

'''
Window positions for the ACF calculation (sliding window)
'''
rr = np.arange(0,125+1,5)
win = np.zeros((len(rr),2))
win[:,0] = rr
win[:,1] = rr+125

#path_boot = '/Users/stephan/Dropbox/python/Simulations-Thesis/ipython-NBs/Current-IPyNBs/ABC/Fitting_Cre/1st_rep/bootstrap_features/'


def start_particles(n_particles, mean_tau=10, mean_T=70, models=models):
    '''
    Sample a set of particles (model and parameters) from prior distributions
    :param n_particles: Integer, number particles to create
    :param mean_tau: Float, mean value of the prior on time distribution, an exponential distribution
    :param mean_T: Float, mean value of the prior off time distribution, an exponential distribution
    :param models: list, the set of models
    :return: 2D Array, first axis: particles, 2nd axis: parameters
    '''

    res = np.zeros((n_particles, 20))
    res[:, 0] = 1 / n_particles

    model_ind = np.random.random_integers(0, len(models) - 1, n_particles)
    res[:, 1] = model_ind

    tau = sp.random.exponential(mean_tau, n_particles)
    km = sp.random.lognormal(sp.log(5), 0.7, n_particles)
    burst = tau * km
    T = sp.random.exponential(mean_T, n_particles)

    res[:, 8] = burst
    res[:, 9] = tau
    res[:, 10] = T

    for ii, mod in enumerate(model_ind):
        res[ii, 2:5] = models[mod]

        p = res[ii, 4]
        if p == 2 or p == 5:
            res[ii, 5] = sp.random.uniform(1, 8)
        elif p == 3 or p == 6:
            res[ii, 6] = sp.random.uniform(1, 8)
        elif p == 4 or p == 7:
            res[ii, 7] = sp.random.uniform(1, 8)

        n = res[ii, 2]
        N = res[ii, 3]
        if n == 2:
            res[ii, 11] = sp.random.uniform(0.8, 1.)
        if N == 2:
            res[ii, 12] = sp.random.uniform(0.8, 1.)
        elif N == 8:
            res[ii, 12:19] = sp.random.uniform(0.8, 1, N - 1)
        elif N == 9:
            res[ii, 12:20] = sp.random.uniform(0.8, 1, N - 1)

    return res


def add_noise_log_normal(simulation, alpha, scale, shape):
    '''
    Add log normally distributed noise to simulated RNA counts
    :param simulation: Array containing simulated time courses of RNA counts
    :param alpha: Float; Scaling factor between RNA number and fluorescence light intensity
    :param scale: Float; Scale parameter of a log normal position (position of the peak)
    :param shape: Float; Shape of of a log normal distribution ('width')
    :return: Array of the same size as simulation with the added noise
    '''
    res = simulation * alpha + sp.random.lognormal(sp.log(scale), shape, size=simulation.shape)
    return res


def fit_noise_model(mock):
    '''
    Use the background light intensity to estimate the parameters of the log normal noise model
    :param mock: 2D Array
    :return: list, scale and shape parameter of the fitted log normal distribution
    '''
    shape, loc, scale = sp.stats.lognorm.fit(mock.reshape(mock.shape[0] * mock.shape[1], ), floc=0)
    return scale, shape


def Gillespie_promoter(particle, particle_trafo, tf, sync):
    '''
    Implentation of the gillespie algorithm to simulate the stochastic promoter switching and the transcription intitation
    events.
    :param particle: 1D Array, particle in burst, tau, T_off parameterisation
    :param particle_trafo: 1D array, transformed particle to reaction rates
    :param tf: float, termination time of the simulation
    :param sync: list, first entry: bool, if true simulation starts in a defined initial state, else in a random state
    :return: list, time points of transcriptional intitation
    '''
    param = create_parameter_vector(particle_trafo)
    birth = param[0]
    promoter_rates = param[1:]
    if sync[0] == True:
        initial_state = sync[1]
    else:
        initial_state = sp.random.random_integers(1, len(promoter_rates))
    #print 'initial state',initial_state

    active_states = np.arange(1, particle[2] + 1)
    #print 'actice states', active_states

    time_rna = []
    time_rna.append(0)

    tt = 0
    state = initial_state
    rna = 0

    a1 = promoter_rates[initial_state - 1]
    a2 = birth

    while tt < tf + 80:
        rr = sp.random.uniform(0, 1, 2)
        if state in active_states:
            a0 = a1 + a2
            tt = tt + 1. / a0 * sp.log(1. / rr[0])

            if a2 >= rr[1] * a0:
                rna = rna + 1
                time_rna.append(tt)
            else:
                state = state + 1
        else:
            a0 = a1
            tt = tt + 1. / a0 * sp.log(1. / rr[0])
            state = state + 1

        if state == len(promoter_rates) + 1:
            state = 1

        a1 = promoter_rates[state - 1]

    return time_rna


def singleTranscript(t_i, tauOn, tauOff, tt, a=1., steep=4.):
    '''
    Add the deterministic fluorescence signal of a single transcript to the one time point of transcriptional intitation
    :param t_i: float, initiation time
    :param tauOn: float, time delay for the polymerase to reach the stem loop section of the transcript, ie when the
    fluorescence signal becomes visible
    :param tauOff: float, time delay for the polymerase to finish the transcript, ie when the fluorescence signal
    disappears
    :param tt:1D array, time vector with the imaging time interval
    :param a: float, intensity of a single transcript
    :param steep: float, steepness of the transition from no signal to signal
    :return:
    '''
    s = a / (1 + sp.exp(-steep * (tt - t_i - tauOn))) - a / (1 + sp.exp(-steep * (tt - t_i - tauOff)))
    return s


def MultipleTranscripts(T_i, tauOn, tauOff, tt):
    '''
    Add the deterministic signal of single transcripts to a list of initiation events
    :param T_i: list, initiation events created by the gillespie simulations
    :param tauOn: float, time delay for the polymerase to reach the stem loop section of the transcript, ie when the
    fluorescence signal becomes visible
    :param tauOff: loat, time delay for the polymerase to finish the transcript, ie when the fluorescence signal
    disappears
    :param tt: 1D array, time vector with the imaging time interval
    :return: 1D array, the summed RNA count over time
    '''
    if len(T_i) == 1:
        return np.zeros(len(tt))
    else:
        S = np.zeros((len(T_i)-1, len(tt)))
        index = np.arange(1, len(T_i))
        for n in index:
            S[n-1, :] = singleTranscript(T_i[n], tauOn, tauOff, tt)
        return S.sum(axis=0)


def p_i_from_u_i(u_i):
    if isinstance(u_i, float):
        res = np.zeros(2)
        denominator = 1 + u_i
        p_1 = 1. / denominator
        p_2 = 1 - p_1
        res[0] = p_1
        res[1] = p_2
    else:
        res = np.zeros(len(u_i) + 1)
        denominator = 1 + sp.sum(np.cumprod(u_i[0:len(u_i)]))
        p_1 = 1. / denominator
        p_i = np.cumprod(u_i) / denominator

        res[0] = p_1
        res[1:] = p_i

    return res


def coord_trafo(particles):
    '''
    Transform the parameters of a particle into kinetic reaction rates for the gillespie simulations
    :param particles: 1D array, particle
    :return: 1D array, transformed particle
    '''
    res = np.zeros((particles.shape[0], 12))
    res[:, 0] = particles[:, 8] / particles[:, 9]

    for ii, part in enumerate(particles):
        if part[2] == 2:
            mu = part[11]
            pi = p_i_from_u_i(mu)
            res[ii, 1:3] = 1. / pi / part[9]
        else:
            res[ii, 1] = 1. / part[9]
        if part[3] > 1:
            n_u = np.sum(part[12:] > 0)
            u_i = part[12:12 + n_u]
            pi = p_i_from_u_i(u_i)
            res[ii, 3:3 + n_u + 1] = 1. / pi / part[10]
        else:
            res[ii, 3] = 1. / part[10]
    return res


def create_parameter_vector(param):
    '''
    Extract kinetic parameters from a single particle for stochastic simulations
    :param param: 1D array, single particle
    :return: 1D array, kinetic parameters
    '''
    ind = np.where(param > 0)[0]
    res = param[ind]

    return res


def simulate_single_particle(particle, n_sim, tf, view=view, deltaT=deltaT, sync = [False]):
    '''
    Simulate a data set of RNA counts over time for a single particle.
    :param particle: 1D array, particle containint the model and parameters
    :param n_sim: int, number of time courses to simulate
    :param tf: float, termination time for the gillespie algorithm
    :param view: load balanced view to parallelise stochastic simulations
    :param deltaT: float, imaging time interval
    :param sync: list, first entry: bool, if true simulation starts in a defined initial state, else in a random state
    :return: two 2D arrays, 1st: simulated RNA counts, 2nd: sampled parameter values as caused by extrinisc noise
    '''
    n_sim = int(n_sim)
    part = np.zeros((n_sim, len(particle))) + particle
    TF = np.zeros(n_sim) + tf
    if particle[5] > 0:
        km = abs(sp.random.normal(particle[8] / particle[9], particle[5], n_sim))
        part[:, 8] = km * particle[9]
    elif particle[6] > 0:
        part[:, 9] = abs(sp.random.normal(particle[9], particle[6], n_sim))
    elif particle[7] > 0:
        part[:, 10] = abs(sp.random.normal(particle[10], particle[7], n_sim))

    part_trafo = coord_trafo(part)

    Sync = np.zeros((n_sim,2)) + sync[0:2]
    time_rna = view.map(Gillespie_promoter, part, part_trafo, TF, Sync)

    if particle[4] == 1. or particle[4] == 5.:
        speed = sp.random.uniform(2, 5, n_sim) * 1000
        if sync[0] == True:
            tau_on = sync[2]/speed
            tau_off = sync[3]/speed
        else:
            tau_on = 2740. / speed
            tau_off = 88000. / speed
    else:
        speed = 3.5 * 1000
        tau_on = np.zeros(n_sim) + 2740 / speed
        tau_off = np.zeros(n_sim) + 88000 / speed

    if sync[0] == True:
        tt = np.arange(0, tf, deltaT)
        TT = np.zeros((n_sim, len(tt))) + tt
        ind_tt = np.where(tt >= 0)[0]
    else:
        tt = np.arange(0, tf + 80, deltaT)
        TT = np.zeros((n_sim, len(tt))) + tt
        ind_tt = np.where(tt >= 80)[0]

    res = view.map(MultipleTranscripts, time_rna, tau_on, tau_off, TT)
    res = np.vstack(res)
    res = res.transpose()

    perturbed = np.zeros((n_sim, 8))
    perturbed[:, 0:7] = part[:, 4:11]
    perturbed[:, 7] = speed

    return res[ind_tt, :], perturbed


def simulate_multiple_particles(particles, n_sim, tf, sync = [False]):
    '''
    Simulate a full particle population
    :param particles: 2D array, 1st axis particles, 2nd axis: parameters
    :param n_sim: int, number of time courses to simulate
    :param tf: float, termination time for the gillespie algorithm
    :param sync: list, first entry: bool, if true simulation starts in a defined initial state, else in a random state
    :return: two 3D arrays, 1st: simulations: 1st axis: time, 2nd axis cells, 3rd axis: particles;
    2nd array, sampled parameter values from extrinsic noise: 1st axis: parameters, 2nd: cells, 3rd: particles
    '''
    rows = particles.shape[0]
    N_sim = np.zeros(rows) + n_sim
    TF = np.zeros(rows) + tf

    res = map(simulate_single_particle, particles, N_sim, TF)

    for ii, rr in enumerate(res):
        if ii == 0:
            sim = rr[0]
            perturb = rr[1]
        else:
            sim = np.dstack((sim, rr[0]))
            perturb = np.dstack((perturb, rr[1]))

    return sim, perturb

def simulate_single_particle_dual(particle, n_sim, tf, view = view, deltaT = deltaT, sync = [False], velo = 2.5):
    '''
    Simulate a dual allele data set for a single particle, i.e. simulate two time courses per cell.
    :param particle: 1D array, particle containing the model and parameters
    :param n_sim: int, number of time courses to simulate
    :param tf: float, termination time for the gillespie algorithm
    :param view: load balanced view to parallelise stochastic simulations
    :param deltaT: float, imaging time interval
    :param sync: list, first entry: bool, if true simulation starts in a defined initial state, else in a random state
    :param velo: float, elongation rate of the RNA polymerase in kb/min
    :return: three 2d arrays: 1st: first allele, 2nd: second allele, 3rd: sampled parameters from extrinsic noise
    '''
    n_sim = int(n_sim)
    part = np.zeros((n_sim, len(particle))) + particle
    TF = np.zeros(n_sim) + tf
    if particle[5] > 0:
        km = abs(sp.random.normal(particle[8] / particle[9], particle[5], n_sim))
        part[:, 8] = km * particle[9]
    elif particle[6] > 0:
        part[:, 9] = abs(sp.random.normal(particle[9], particle[6], n_sim))
    elif particle[7] > 0:
        part[:, 10] = abs(sp.random.normal(particle[10], particle[7], n_sim))

    part_trafo = coord_trafo(part)

    Sync = np.zeros((n_sim, 2)) + sync[0:2]
    time_rna_1 = view.map(Gillespie_promoter, part, part_trafo, TF, Sync)
    time_rna_2 = view.map(Gillespie_promoter, part, part_trafo, TF, Sync)

    if particle[4] == 1. or particle[4] == 5.:
        speed = sp.random.uniform(1, 5, n_sim) * 1000
        if velo != 2.5:
            speed = velo * 1000
        # print speed
        if sync[0] == True:
            tau_on = sync[2] / speed
            tau_off = sync[3] / speed
        else:
            tau_on = np.zeros(n_sim) + 2740. / speed
            tau_off = np.zeros(n_sim) + 88000. / speed
    else:
        speed = velo * 1000
        tau_on = np.zeros(n_sim) + 2740 / speed
        tau_off = np.zeros(n_sim) + 88000 / speed

    if sync[0] == True:
        tt = np.arange(0, tf, deltaT)
        TT = np.zeros((n_sim, len(tt))) + tt
        ind_tt = np.where(tt >= 0)[0]
    else:
        tt = np.arange(0, tf + 80, deltaT)
        TT = np.zeros((n_sim, len(tt))) + tt
        ind_tt = np.where(tt >= 80)[0]

    res_1 = view.map(MultipleTranscripts, time_rna_1, tau_on, tau_off, TT)
    res_1 = np.vstack(res_1)
    res_1 = res_1.transpose()

    res_2 = view.map(MultipleTranscripts, time_rna_2, tau_on, tau_off, TT)
    res_2 = np.vstack(res_2)
    res_2 = res_2.transpose()

    perturbed = np.zeros((n_sim, 8))
    perturbed[:, 0:7] = part[:, 4:11]
    perturbed[:, 7] = speed

    return res_1[ind_tt, :], res_2[ind_tt, :], perturbed

def distance_kolmogorov(data, sim):
    '''
    Calculate the kolmogrov statistic for two data sets
    :param data: 2D array, experimental data
    :param sim: 2D array, simulations
    :return: float, kolmogorov statistic
    '''
    res = sp.stats.ks_2samp(data.reshape(data.shape[0] * data.shape[1], ), sim.reshape(sim.shape[0] * sim.shape[1], ))[
        0]
    return res


def AutoCorrelate(signal, deltaT):
    '''
	Calculate the autocorrelation function
	:param signal: 1d array, input signal
	:param deltaT: , float, experimental time resolution
	:return: 2D array, 1st column: lag, 2nd column: acf(lag)
	'''
    signal = signal - np.mean(signal)
    res = np.correlate(signal, signal, mode='full')
    res = res / np.max(res)
    out = np.empty((2, len(signal)))
    out[0] = np.arange(0, len(signal)) * deltaT

    out[1] = res[np.argmax(res):]
    return out


def AutocorrelateAllData(DataTC, deltaT=1):
    '''
    Calculate the autocorrelations for all extracted signals
    :param DataTC: 2D numpy array with the time courses in columns
    :param deltaT: imaging time interval
    :return: 2d numpy array with the calculated autocorrelations in columns
    '''
    res = np.empty((DataTC.shape))
    count = 0
    while count < DataTC.shape[1]:
        if np.sum(DataTC[:,count] == 0) ==  DataTC.shape[0]:
            rr = np.zeros(res.shape[0]) + 10
            res[:,count] = rr
        else:
            res[:, count] = AutoCorrelate(DataTC[:, count], deltaT)[1]
        count = count + 1
    return res


def acf_sliding_window(data,windows = win):
    TS = []
    ACF = []
    for ww in windows:
        dd = data[int(ww[0]):int(ww[1]),:]
        acf = AutocorrelateAllData(dd,1)
        TS.append(dd)
        ACF.append(acf)
    return TS,ACF


def ACF_HL(acf):
    ind_below = np.where(acf <= 0.5)[0]
    x_low = ind_below[0]
    x_high = ind_below[0] - 1
    acf_low = acf[x_low]
    acf_high = acf[x_high]

    slope = acf_low - acf_high
    inter = acf_low - slope*x_low
    hl = (0.5 - inter)/slope
    return hl

def ACF_HL_multi(ACF):
    res = np.zeros(ACF.shape[1])
    index = np.arange(0, ACF.shape[1])
    for ii in index:
        res[ii] = ACF_HL(ACF[:, ii])
    return res


def ACF_HL_distance(acf_hl_data, acf_hl_sim):
    res = sp.stats.ks_2samp(acf_hl_data, acf_hl_sim)[0]
    return res


def grbf(x1, x2, sigma):
    '''Calculates the Gaussian radial base function kernel'''
    n, nfeatures = x1.shape
    m, mfeatures = x2.shape

    k1 = np.sum((x1 * x1), 1)
    q = np.tile(k1, (m, 1)).transpose()
    del k1

    k2 = np.sum((x2 * x2), 1)
    r = np.tile(k2, (n, 1))
    del k2

    h = q + r
    del q, r

    # The norm
    h = h - 2 * np.dot(x1, x2.transpose())
    h = np.array(h, dtype=float)

    return np.exp(-1 * h / (2 * pow(sigma, 2)))


def kernelwidth(x1, x2):
    '''Function to estimate the sigma parameter

       The RBF kernel width sigma is computed according to a rule of thumb:

       Pick sigma such that the exponent of exp(- ||x-y|| / (2*sigma2)),
       in other words ||x-y|| / (2*sigma2),  equals 1 for the median distance x
       and y of all distances between points from both data sets X and Y.
    '''
    n, nfeatures = x1.shape
    m, mfeatures = x2.shape

    k1 = np.sum((x1 * x1), 1)
    q = np.tile(k1, (m, 1)).transpose()
    del k1

    k2 = np.sum((x2 * x2), 1)
    r = np.tile(k2, (n, 1))
    del k2

    h = q + r
    del q, r

    # The norm
    h = h - 2 * np.dot(x1, x2.transpose())
    h = np.array(h, dtype=float)

    mdist = np.median([i for i in h.flat if i])

    sigma = sp.sqrt(mdist / 2.0)
    if not sigma:
        sigma = 1

    return sigma


def mmd(x1, x2, sigma=None, verbose=False):
    '''Calculates the unbiased mmd from two arrays x1 and x2

    sigma: the parameter for grbf. If None sigma is estimated

    Returns (sigma, mmd)

    Calculation of MMD as implemented by V. Van Ash, code published together with his PhD thesis.

    '''
    if x1.size != x2.size:
        raise ValueError('Arrays should have an equal amount of instances')

    # Number of instances
    m, nfeatures = x1.shape

    # Calculate sigma
    if sigma is None:
        sigma = kernelwidth(x1, x2)
    if verbose:
        print 'Got kernelwidth'

    # Calculate the kernels
    Kxx = grbf(x1, x1, sigma)
    if verbose:
        print 'Got Kxx'
    Kyy = grbf(x2, x2, sigma)
    if verbose:
        print 'Got Kyy'
    s = Kxx + Kyy
    del Kxx, Kyy

    Kxy = grbf(x1, x2, sigma)
    if verbose:
        print 'Got Kxy'
    s = s - Kxy - Kxy
    del Kxy
    if verbose:
        print 'Got sum'

    # For unbiased estimator: subtract diagonal
    s = s - np.diag(s.diagonal())

    value = np.sum(s) / (m * (m - 1))

    return sigma, value



def average_acf(ACF):
    res = np.zeros((len(ACF),ACF[0].shape[0]))
    for ii,acf in enumerate(ACF):
        res[ii] = acf.mean(axis = 1)
    return res


def acf_hl_lag(data,windows=win):
    ts,acf = acf_sliding_window(data,windows)
    acf_m = average_acf(acf)

    hl = np.zeros((len(acf),data.shape[1]))
    lag = np.zeros((len(acf),data.shape[1]))
    for ii,aa in enumerate(acf):
        hl[ii] = ACF_HL_multi(aa)
        lag[ii] = aa[1,:]
    return acf_m.mean(axis = 0),hl,lag

def distance_all(data,simulations,windows = win,max_lag = max_lag):
    '''
    Calculate the distance metric for the simulations created by a population of particles.
    :param data: 2D array, experimental data
    :param simulations: 3D simulation, 1st axis: time, 2nd axis: cells, 3rd axis: particles
    :param windows: 2D array, window positions for ACF calculations
    :param max_lag: float, index up to which ACFs are compared
    :return: 1d array, 5 individual distances and their sum
    '''
    res = np.zeros((simulations.shape[2],6))
    rr = np.zeros(6)
    acf_data,hl_data,lag_data = acf_hl_lag(data,windows)

    ind = np.arange(0,simulations.shape[2])
    for ii in ind:
        sim = simulations[:,:,ii]
        acf_sim,hl_sim,lag_sim = acf_hl_lag(sim,windows)
        rr[0] = distance_kolmogorov(data,sim)
        rr[1] = np.sum((acf_data[1:max_lag] - acf_sim[1:max_lag])**2)
        rr[2] = distance_kolmogorov(hl_data,hl_sim)
        rr[3] = distance_kolmogorov(lag_data,lag_sim)
        rr[4] = mmd(data,sim)[1]
        rr[5] = rr[0:5].sum()
        res[ii] = rr
    return res


def smc_abc_start(data, mock, n_particles, alpha = alpha, deltaT=deltaT):
    '''
    Direct creation of a random start population by sampling from priors and direct evaliation of the particles
    :param data: 2d array, experimental data
    :param mock: 2d array, background signal, same size as data
    :param n_particles: int, number of particles for the population
    :param alpha: float, scaling factor between RNA number and fluorescence intensity
    :param deltaT: float, imaging time interval in minutes
    :return: list of arrays, 1st: 2d array, sampled particles, 2nd: 3d array, simulations,
    3rd: 2d array, distances, 4th: 3d array, sampled parameter values from extrinsc noise
    '''
    run = time.time()
    n_sim = data.shape[1]
    tf = data.shape[0] * deltaT
    particles = start_particles(n_particles)

    simulations, perturbations = simulate_multiple_particles(particles, n_sim, tf)

    scale, shape = fit_noise_model(mock)
    sim_noise = add_noise_log_normal(simulations, alpha, scale, shape)

    distance = distance_all(data, sim_noise)

    ind_sort = np.argsort(distance[:, 5])
    distance = distance[ind_sort, :]
    sim_noise = sim_noise[:, :, ind_sort]
    perturbations = perturbations[:, :, ind_sort]
    particles = particles[ind_sort, :]
    particles[:, 0] = 1. / n_particles

    end = time.time()
    print 'Min start distance:', np.round(distance[:, 5].min(), 3)
    print 'Mean start distance:', np.round(distance[:, 5].mean(), 3)
    print 'Max start distance:', np.round(distance[:, 5].max(), 3)
    print 'STD of start distance:', np.round(distance[:, 5].std(), 3)
    print 'Duration:', np.round((end - run) / 60, 2), 'min'

    return particles, sim_noise, distance, perturbations
        

def proposal_model(model, threshold, model_transitions=model_transitions):
    '''
    Part of the perturbation kernel for the SMC ABC algorithm, proposes a new model based on the current model
    :param model: int, index of the current model
    :param threshold: float, between 0 and 1, probability of a move
    :param model_transitions: list of allowed model transitions
    :return: int, new model index
    '''
    rr = sp.random.uniform(0, 1, 1)
    old_model = model
    if rr > threshold:
        transitions = model_transitions[int(old_model)]
        index = int(np.random.random_integers(0, len(transitions) - 1, 1)[0])
        return transitions[index]
    else:
        return old_model


def proposal_u_i(u_i, lam=2):
    '''
    Part of the perturbation kernel. Proposes new u_i values from a beta distributin. u_i values describe the
    switching rates between individual promoter stats.
    :param u_i: 1d array, vector of the current u_i values
    :param lam: float, rescaling factor
    :return: 1d array, vector of new u_i values
    '''
    alpha = 1. + lam * u_i
    beta = 1. + lam * (1. - u_i)
    return sp.random.beta(alpha, beta)


def proposal_b_tau_T(param, sigmas=sigmas):
    '''
    Part of the perturbation kernel, proposes new values for burst size, on time and off time from a log normal distribution
    :param param: 1d array, parameter vector: burst, t_on, t_off
    :param sigmas: list of floats, shape parameters of the individual log normal distributions
    :return:
    '''
    res = sp.random.lognormal(sp.log(param), sigmas)
    return res


def propose_sigma(sigma, sig=1):
    '''
    part of the perturbation kernel, propose a new value for strength of the exterinsic noise.
    :param sigma:
    :param sig:
    :return:
    '''
    res = abs(sp.random.normal(sigma, sig, 1))
    return res


def find_sigma(particle):
    '''
    Select the correct value of the strength of the extrinsic noise based on the current model.
    :param particle:
    :return:
    '''
    p = particle[4]
    if p == 2 or p == 5:
        ind = 5
        sigma = particle[ind]
    elif p == 3 or p == 6:
        ind = 6
        sigma = particle[ind]
    elif p == 4 or p == 7:
        ind = 6
        sigma = particle[ind]
    return ind, sigma


def create_new_particles(accepted_particles, n_new, threshold, models=models):
    '''
    Sample new particles based on accepted particle positions and proposal distributions.
    :param accepted_particles: 2d array, accepted current particles
    :param n_new: int, number of new particles to create
    :param threshold: float, between 0 and 1, probability for a move in model space
    :param models: list of allowed models
    :return: 2 array, created particles
    '''
    #print 'create, threshold', threshold
    weights = accepted_particles[:, 0] / np.sum(accepted_particles[:, 0])
    particle_index = np.arange(0, accepted_particles.shape[0])
    res = np.zeros((n_new, accepted_particles.shape[1]))

    index = np.arange(0, n_new)
    for ii in index:
        ind = np.random.choice(particle_index, 1, 1, weights)
        old_particle = accepted_particles[ind].reshape(accepted_particles.shape[1], )
        burst_tau_T = proposal_b_tau_T(old_particle[8:11])
        new_particle = np.zeros(len(old_particle))
        new_particle[8:11] = burst_tau_T

        new_model = proposal_model(old_particle[1],threshold)
        new_particle[1] = new_model
        new_particle[2:5] = models[int(new_model)]
        n = models[int(new_model)][0]
        N = models[int(new_model)][1]
        p = models[int(new_model)][2]

        if n == old_particle[2]:
            if n > 1:
                mu = old_particle[11]
                new_particle[11] = proposal_u_i(mu)
        else:
            if n > old_particle[2]:
                new_particle[11] = sp.random.uniform(0.8, 1)

        if N == old_particle[3]:
            if N > 1:
                ui = old_particle[12:12 + N - 1]
                new_particle[12:12 + N - 1] = proposal_u_i(ui)
        else:
            if N > old_particle[3]:
                new_particle[12:12 + old_particle[3] - 1] = old_particle[12:12 + old_particle[3] - 1]
                new_particle[12 + old_particle[3] - 1:12 + N - 1] = sp.random.uniform(0.8, 1, N - old_particle[3])

            else:
                new_particle[12:12 + N - 1] = old_particle[12:12 + N - 1]

        if p == old_particle[4]:
            if p > 1:
                ind, sigma = find_sigma(old_particle)
                new_particle[ind] = propose_sigma(sigma)
        else:
            if p > 1:
                ind, sigma = find_sigma(new_particle)
                new_particle[ind] = sp.random.uniform(1, 8, 1)

        res[ii] = new_particle

    return res


def log_normal(x, mu, sigma):
    res = 1. / (x * sigma * sp.sqrt(2 * sp.pi)) * sp.exp(-(sp.log(x) - sp.log(mu)) ** 2 / 2 / sigma ** 2)
    return res


def prior_b_tau_T(param):
    '''
    Calculate the prior probability of the positions of the current particles for the main parameters: burst, t_on, t_off.
    :param param: 1d array
    :return: float
    '''
    km = param[0] / param[1]
    tau = param[1]
    T = param[2]

    prior_km = log_normal(km, mu=5, sigma=0.7)
    prior_tau = 1. / 10 * sp.exp(-1. / 10 * tau)
    prior_T = 1. / 70 * sp.exp(-1. / 70 * T)

    return prior_km * prior_tau * prior_T


def proposal_density(param_old, param_new, sigmas=sigmas):
    '''
    Calculate the probability for the positions of current particles in search space based on the previous positions.
    Only for the main parameters: burst, t_on and t_off
    :param param_old: 1d array, old parameters
    :param param_new: 1d array, new parameters
    :param sigmas: list of shape parameters of the proposal log normal distributions
    :return:
    '''
    burst_old = param_old[0]
    tau_old = param_old[1]
    T_old = param_old[2]
    burst_new = param_new[0]
    tau_new = param_new[1]
    T_new = param_new[2]

    den_burst = log_normal(burst_new, burst_old, sigmas[0])
    den_tau = log_normal(tau_new, tau_old, sigmas[1])
    den_T = log_normal(T_new, T_old, sigmas[2])

    return den_burst * den_tau * den_T


def norm_weights(old_particles, new_part):
    '''
    Calculate the normalisation factor for the new particles based on the previous particles
    :param old_particles: 2d array, old particls
    :param new_part: 2d array, new particles
    :return: float
    '''
    old_weights = old_particles[:, 0] / np.sum(old_particles[:, 0])
    res = np.zeros(len(old_weights))
    for ii, old_part in enumerate(old_particles):
        res[ii] = proposal_density(old_part[8:11], new_part[8:11])
    return np.sum(res * old_weights)


def weigh_particles(old_particles, new_particles):
    '''
    Calculate weights of new particles based on their position and the position of the predecessors in search space.
    :param old_particles: 2d array, particles from the previous generation
    :param new_particles: 2d array, particles from the current population to be weighted
    :return: 1d array, calculated weights
    '''
    for ii, new_part in enumerate(new_particles):
        prior = prior_b_tau_T(new_part[8:11])
        norm = norm_weights(old_particles, new_part)
        new_particles[ii, 0] = prior / norm
    return new_particles


def create_new_particles_below_max_dist(data, mock, selected_particles, max_dist, n_new, alpha, threshold, deltaT=deltaT):
    #print 'max dist, threshold', threshold
    n_sim = data.shape[1]
    tf = data.shape[0] * deltaT
    max_tries = 20
    tries = 0
    n_accepted = 0
    n_created = 0
    factor = 1
    scale, shape = fit_noise_model(mock)

    while tries < max_tries:

        if (n_new - n_accepted) < 10:
            factor = 2
        new = factor * (n_new - n_accepted)
        factor = 1

        if tries == 0 or n_accepted == 0:
            new_particles = create_new_particles(selected_particles, new, threshold)
        else:
            new_particles = create_new_particles(np.vstack((selected_particles, accepted_particles)), new, threshold)
        n_created = n_created + new_particles.shape[0]

        simulations, perturbations = simulate_multiple_particles(new_particles, n_sim, tf)

        sim_noise = add_noise_log_normal(simulations, alpha, scale, shape)
        del simulations

        distance = distance_all(data, sim_noise)
        ind_sort = np.argsort(distance[:, 5])
        distance = distance[ind_sort, :]
        new_particles = new_particles[ind_sort, :]
        sim_noise = sim_noise[:, :, ind_sort]
        perturbations = perturbations[:, :, ind_sort]

        accepted_particle_index = np.where(distance[:, 5] < max_dist)[0]

        if len(accepted_particle_index) == 0:
            print 'Try:', tries, 'No particles accepted'

        elif len(accepted_particle_index) > 0 and n_accepted == 0:
            print 'Try:', tries, 'First particles accepted:', len(accepted_particle_index)
            accepted_particles = weigh_particles(selected_particles, new_particles[accepted_particle_index, :])
            n_accepted = n_accepted + len(accepted_particle_index)
        elif len(accepted_particle_index) > 0 and n_accepted > 0:
            new_particles[accepted_particle_index, :] = weigh_particles(accepted_particles,
                                                                        new_particles[accepted_particle_index, :])
            accepted_particles = np.vstack((accepted_particles, new_particles[accepted_particle_index, :]))
            n_accepted = n_accepted + len(accepted_particle_index)
            print 'Try:', tries, 'More particles accepted:', n_accepted

        if tries == 0:
            created_particles = new_particles
            created_perturbations = perturbations
            created_simulations = sim_noise
            created_distances = distance
        else:
            created_particles = np.vstack((created_particles, new_particles))
            created_perturbations = np.dstack((created_perturbations, perturbations))
            created_simulations = np.dstack((created_simulations, sim_noise))
            created_distances = np.vstack((created_distances, distance))

        ind_sort = np.argsort(created_distances[:, 5])[0:n_new]
        created_particles = created_particles[ind_sort, :]
        created_perturbations = created_perturbations[:, :, ind_sort]
        created_simulations = created_simulations[:, :, ind_sort]
        created_distances = created_distances[ind_sort, :]

        if n_accepted >= n_new:
            print 'Enough particles accepted:', n_accepted
            print '# Created particles:', n_created
            return [created_particles[0:n_new],
                    created_simulations[:, :, 0:n_new],
                    created_distances[0:n_new, :],
                    created_perturbations[:, :, 0:n_new],
                    n_accepted,
                    n_created]

        if tries == 10 and n_accepted < float(n_new) / 2:
            max_tries = 15
            print 'Slow creation of new particles, reduced number of tries to:', max_tries

        tries = tries + 1

    print 'Not enough particles accepted:', n_accepted
    print '# Created particles:', n_created
    return [created_particles[0:n_new],
            created_simulations[:, :, 0:n_new],
            created_distances[0:n_new, :],
            created_perturbations[:, :, 0:n_new],
            n_accepted,
            n_created]


def smc_abc(data, mock, start_population, iterations, stop, threshold = 0.6, alpha = alpha, save=[False]):
    #print 'smc, threshold', threshold
    old_particles = start_population[0]
    old_sim = start_population[1]
    old_distance = start_population[2]
    dist_max_old = old_distance[:, 5].max()
    old_perturbations = start_population[3]

    print '# particles better than stop:', np.sum(old_distance[:, 5] < stop)
    print 'Min start distance:', np.round(old_distance[:, 5].min(), 3)
    print 'Mean start distance:', np.round(old_distance[:, 5].mean(), 3)
    print 'Max start distance:', np.round(old_distance[:, 5].max(), 3)
    print 'STD of start distance:', np.round(old_distance[:, 5].std(), 3)
    print 'Mean of burst, tau and T:', np.round(old_particles[:, 8:11].mean(axis=0), 3)
    print 'STD of burst, tau and T:', np.round(old_particles[:, 8:11].std(axis=0), 3)

    Max_distances = []
    N_accepted = []
    N_created = []

    percentile = 0.2
    n_particles = old_particles.shape[0]
    n_old = int(percentile * n_particles)
    n_new = old_particles.shape[0] - n_old
    index_max_distance = int(percentile * n_particles)

    Run = time.time()
    ii = 0
    while ii < iterations:
        run = time.time()
        print 'Iteration:', ii + 1
        max_distance = old_distance[index_max_distance, 5]
        print 'Threshold distance', np.round(max_distance, 3)
        Max_distances.append(max_distance)

        selected_particles = old_particles[0:index_max_distance, :]
        new_particles = create_new_particles_below_max_dist(data, mock, selected_particles, max_distance, n_new, alpha, threshold)

        N_accepted.append(new_particles[4])
        N_created.append(new_particles[5])

        cand_particles = np.vstack((old_particles, new_particles[0]))
        cand_simulations = np.dstack((old_sim, new_particles[1]))
        cand_distances = np.vstack((old_distance, new_particles[2]))
        cand_perturbations = np.dstack((old_perturbations, new_particles[3]))

        ind_sort = np.argsort(cand_distances[:, 5])[0:n_particles]
        cand_particles = cand_particles[ind_sort, :]
        cand_simulations = cand_simulations[:, :, ind_sort]
        cand_distances = cand_distances[ind_sort, :]
        cand_perturbations = cand_perturbations[:, :, ind_sort]

        if ii == 0:
            all_particles = np.dstack((old_particles, cand_particles[0:n_particles, :]))
            all_distances = np.dstack((old_distance, cand_distances[0:n_particles, :]))
        else:
            all_particles = np.dstack((all_particles, cand_particles[0:n_particles, :]))
            all_distances = np.dstack((all_distances, cand_distances[0:n_particles, :]))

        end = time.time()
        print '# particles better than stop:', np.sum(cand_distances[0:n_particles, 5] < stop)
        print 'Min distance of current population:', np.round(cand_distances[0:n_particles, 5].min(), 3)
        print 'Mean distance of current population:', np.round(cand_distances[0:n_particles, 5].mean(), 3)
        print 'Max distance of current population:', np.round(cand_distances[0:n_particles, 5].max(), 3)
        print 'Mean of burst, tau and T:', np.round(cand_particles[0:n_particles, 8:11].mean(axis=0), 3)
        print 'STD of burst, tau and T:', np.round(cand_particles[0:n_particles, 8:11].std(axis=0), 3)
        print 'Duration of', ii + 1, 'th iteration', np.round((end - run) / 60, 2), 'min'

        if save[0] == True:
            np.save(save[1] + save[2] + str(ii) + '.npy', [all_particles,
                                                           cand_simulations[:, :, 0:n_particles],
                                                           all_distances,
                                                           cand_perturbations,
                                                           N_accepted,
                                                           N_created,
                                                           Max_distances])

        stop_distance = cand_distances[0:n_particles, 5].max()
        if stop_distance <= stop:
            print 'STOP: minimal distance reached'
            print 'Duration of iterations:', np.round((time.time() - Run) / 60, 2), 'min'
            return [all_particles,
                    cand_simulations[:, :, 0:n_particles],
                    all_distances,
                    cand_perturbations[:, :, 0:n_particles],
                    N_accepted,
                    N_created,
                    Max_distances]

        dist_max_new = cand_distances[0:n_particles, 5].max()
        if ii > 1:
            if dist_max_new > 0.98 * dist_max_old:
                print 'STOP: no improvement gained, stopped after', ii, 'iterations'
                print 'Duration of iterations:', np.round((time.time() - Run) / 60, 2), 'min'
                return [all_particles,
                        cand_simulations[:, :, 0:n_particles],
                        all_distances,
                        cand_perturbations[:, :, 0:n_particles],
                        N_accepted,
                        N_created,
                        Max_distances]
        dist_max_old = dist_max_new

        old_distance = cand_distances[0:n_particles, :]
        old_particles = cand_particles[0:n_particles, :]
        old_sim = cand_simulations[:, :, 0:n_particles]
        old_perturbations = cand_perturbations[:, :, 0:n_particles]
        ii = ii + 1

    print 'Last iteration finished, max distance:', cand_distances[0:n_particles, 5].max()
    print 'Duration of iterations:', np.round((time.time() - Run) / 60, 2), 'min'
    return [all_particles,
            cand_simulations[:, :, 0:n_particles],
            all_distances,
            cand_perturbations[:, :, 0:n_particles],
            N_accepted,
            N_created,
            Max_distances]


    ## TODO: Global fitting: all data sets at once


def smc_start_multi(n_particles,n_sim, deltaT = deltaT):
    '''
    Sample random start particles and simulate them
    :param n_particles: int, number of particles to create
    :param n_sim: int, number cells to simulate
    :param deltaT: float, imaging time interval
    :return:
    '''
    run = time.time()
    tf = 250. * deltaT
    particles = start_particles(n_particles)

    simulations, perturbations = simulate_multiple_particles(particles, n_sim, tf)
    end = time.time()
    print 'Duration:',np.round((end - run)/60,2)
    return [particles,simulations,perturbations]

def particle_distance_lut(data,mock,paths,alpha = alpha):
    '''
    Find the best particles from a large start population compared to an experimental data set. Used to generate a suitable
    start population.
    :param data: 2D array, experimental data
    :param mock: 2D array, background signal, same size as data
    :param paths: list, 1st entry: path to the candidate particles, 2nd entry: path to a folder to save individual particles
    :param alpha: float, scaling between RNA and fluorescence intensity
    :return: two 2D arrays, 1st: distances of all candidate particles for each features; 2nd array: all particle parameters
    '''
    file_names = glob.glob(paths[0]+'*.npy')
    shape_data = data.shape
    scale,shape = fit_noise_model(mock)

    run = time.time()
    count = 0
    index = np.arange(0,len(file_names))
    for ii in index:
        print 'File:', ii+1,'/',len(file_names)
        res = np.load(file_names[ii])
        par = res[0]
        sim = res[1]
        sim = sim[0:shape_data[0],0:shape_data[1],:]
        sim_noise = add_noise_log_normal(sim,alpha,scale,shape)
        per = res[2]
        per = per[0:shape_data[1],:,:]

        if ii == 0:
            parameter = res[0]
            dist = distance_all(data,sim_noise)
            distance = dist
        else:
            parameter = np.vstack((parameter,par))
            dist = distance_all(data,sim_noise)
            distance = np.vstack((distance,dist))

        ind = np.arange(0,par.shape[0])
        for jj in ind:
            np.save(paths[1]+'particle_'+str(count)+'.npy',[par[jj],sim_noise[:,:,jj],per[:,:,jj],dist[jj]])
            count = count + 1
    end = time.time()
    print 'Duration:',np.round((end - run)/60,2),'min'
    return distance,parameter

def corrected_filelist(path):
    '''
    Correct the numbering of the candidate particles
    :param path: string, path to folder containing all particles
    :return: list, containing the paths to all particles in corrected order
    '''
    file_list = glob.glob(path+'*.npy')
    ind = np.arange(0,len(file_list))
    files = []
    for ii in ind:
        files.append(path+'particle_'+str(ii)+'.npy')
    return files

def best_start_particles(files,distances,n_best):
    '''
    Find the best particles of the prepared ones for a suitable SMC ABC start population
    :param files: list of strings, paths to all particles in corrected order
    :param distances: 2D array, distances of all particles, same order as in files
    :param n_best: int, number particles to be in the start population
    :return: list of arrays, 1st: 2D array, particles; 2nd: 3D array all simulations, 3rd: 3D array, sampled parameter
    values from extrinsic noise
    '''
    run = time.time()
    ind_sort = np.argsort(distances[:,5])[0:n_best]
    dist = distances[ind_sort,:]
    for ii,ind in enumerate(ind_sort):
        part = np.load(files[ind])
        if ii == 0:
            particles = part[0]
            simulations = part[1]
            perturbations = part[2]
        else:
            particles = np.vstack((particles,part[0]))
            simulations = np.dstack((simulations,part[1]))
            perturbations = np.dstack((perturbations,part[2]))
    end = time.time()
    print 'Duration:',np.round((end - run)/60,2),'min'
    return [particles,simulations,dist,perturbations]

#========================================================
# Global fitting of one model to all available data sets

def sample_beta(scale,shift,n_samples, lam = 5):
    center = 0.5
    alpha = 1 + lam*center
    beta = 1 + lam*(1 - center)
    rr = shift + scale*sp.random.beta(alpha,beta,n_samples)
    return rr

def global_smc_abc_start(data,mock,n_particles,selected_particles,scale,shift,
                         fix = 1):
    run = time.time()
    particles = np.zeros((n_particles,20,len(data)))
    particles[:,0,:] = 1./n_particles
    particles[:,1,:] = 5
    particles[:,2,:] = 1
    particles[:,3,:] = 1
    particles[:,4,:] = 5

    sel_part = np.vstack(selected_particles)

    sig_km_mean = sel_part[:,5].mean()
    sig_km_std = sel_part[:,5].std()
    sig_km = abs(sp.random.normal(sig_km_mean,sig_km_std,n_particles).reshape(n_particles,1))
    sig_km = np.repeat(sig_km,len(data),axis = 1)
    particles[:,5,:] = particles[:,5,:] + sig_km

    if fix == 1:
        print 'Only off time as local variable'
        km_mean = sel_part[:,8].mean()
        km_std = sel_part[:,8].mean()
        km = abs(sp.random.normal(km_mean,km_std,n_particles).reshape(n_particles,1))
        km = np.repeat(km,len(data),axis = 1)
        particles[:,8,:] = particles[:,8,:] + km

        tau_mean = sel_part[:,9].mean()
        tau_std = sel_part[:,9].std()
        tau = abs(sp.random.normal(tau_mean,tau_std,n_particles).reshape(n_particles,1))
        tau = np.repeat(tau,len(data),axis = 1)
        particles[:,9,:] = particles[:,9,:] + tau

        particles[:,8,:] = particles[:,8,:]*particles[:,9,:]

        TT = np.zeros((n_particles,len(data)))
        for ii in np.arange(0,len(data)):
            TT[:,ii] = sample_beta(scale[ii],shift[ii],n_particles)
            TT = np.sort(TT,axis = 0)
        particles[:,10,:] = TT

    Sim,Perturb = simulate_particles_global(mock,particles)
    print 'Simulations: DONE'

    Dist,Total_dist = distance_all_global(data,Sim)

    print 'Distance: DONE'

    print 'Min start distance:', np.round(Total_dist.min(), 3)
    print 'Mean start distance:', np.round(Total_dist.mean(), 3)
    print 'Max start distance:', np.round(Total_dist.max(), 3)
    print 'STD of start distance:', np.round(Total_dist.std(), 3)
    end = time.time()
    print 'Duration',np.round((end - run)/60,2),'min'
    return sort_particles_global(particles,Sim,Perturb,Dist,Total_dist)

def simulate_particles_global(mock,particles,alpha = alpha_global):
    Sim = []
    Perturb = []

    for ii in np.arange(0,len(mock)):
        tf = mock[ii].shape[0]*deltaT
        n_sim = mock[ii].shape[1]

        sim = simulate_multiple_particles(particles[:,:,ii],n_sim,tf)
        scale,shape = fit_noise_model(mock[ii])
        sims = add_noise_log_normal(sim[0],alpha[ii],scale,shape)
        Sim.append(sims)
        Perturb.append(sim[1])

        print 'Simulated ', ii+1 ,'/', len(mock)
    return Sim,Perturb

def distance_all_global(data,sims):
    Dist = []
    Total_dist = []
    for ii,dat in enumerate(data):
        dist = distance_all(dat,sims[ii])
        Dist.append(dist)
        Total_dist.append(dist[:,5])
    Total_dist = np.vstack(Total_dist)
    Total_dist = Total_dist.sum(axis = 0)
    return Dist,Total_dist

def sort_particles_global(particles,Sim,Perturb,Dist,Total_dist):
    ind_sort = np.argsort(Total_dist)
    particles = particles[ind_sort,:,:]
    sim = []
    per = []
    dist = []
    for ii in np.arange(0,len(Sim)):
        sim.append(Sim[ii][:,:,ind_sort])
        per.append(Perturb[ii][:,:,ind_sort])
        dist.append(Dist[ii][ind_sort,:])
    Total_dist = Total_dist[ind_sort]
    return particles,sim,per,dist,Total_dist

def select_best_particles_global(particles,sim,perturb,dist,total_dist,n_new):
    part_sort,sim_sort,per_sort,dist_sort,total_dist_sort = sort_particles_global(particles,sim,perturb,dist,total_dist)
    res_sim = []
    res_perturb = []
    res_dist = []
    for ii in np.arange(0,len(sim)):
        res_sim.append(sim_sort[ii][:,:,0:n_new])
        res_perturb.append(per_sort[ii][:,:,0:n_new])
        res_dist.append(dist_sort[ii][0:n_new,:])
    return part_sort[0:n_new,:,:],res_sim,res_perturb,res_dist,total_dist_sort[0:n_new]

def proposal_b_tau_T_global(param,sig = sigmas_global):
    res = sp.random.lognormal(sp.log(param), sig)
    return res

def create_new_particles_global(accepted_particles,n_new,sel):
    weights = accepted_particles[:,0,0]/accepted_particles[:,0,0].sum()
    particle_index = np.arange(0,accepted_particles.shape[0])
    sel_index = np.random.choice(particle_index,n_new,1,weights)
    res = accepted_particles[sel_index,:,:]

    for ii in np.arange(0,n_new):
        sigma_km = res[ii,5,0]
        sigma_km_prop = propose_sigma(sigma_km)
        res[ii,5,:] = sigma_km_prop

        if sel == 'T':
            burst = res[ii,8,0]
            tau = res[ii,9,0]
            T = res[ii,10,:]

            param = np.hstack((burst,tau,T))
            b_tau_T_prop = proposal_b_tau_T_global(param)

            res[ii,8,:] = b_tau_T_prop[0]
            res[ii,9,:] = b_tau_T_prop[1]
            res[ii,10,:] = -np.sort(-b_tau_T_prop[2:])

        if sel == 'burst':
            burst = res[ii,8,:]
            tau = res[ii,9,0]
            T = res[ii,10,0]

            param = np.hstack((burst,tau,T))
            b_tau_T_prop = proposal_b_tau_T_global(param)
            n_data = accepted_particles.shape[2]
            res[ii,8,:] = np.sort(b_tau_T_prop[0:n_data])
            res[ii,9,:] = b_tau_T_prop[n_data]
            res[ii,10,:] = b_tau_T_prop[-1]

        if sel == 'both':
            burst = res[ii,8,:]
            tau = res[ii,9,0]
            T = res[ii,10,:]

            param = np.hstack((burst,tau,T))
            b_tau_T_prop = proposal_b_tau_T_global(param)

            n_data = accepted_particles.shape[2]
            res[ii,8,:] = np.sort(b_tau_T_prop[0:n_data])
            res[ii,9,:] = b_tau_T_prop[n_data]
            res[ii,10,:] = -np.sort(-b_tau_T_prop[n_data+1:])

        off = res[ii,3,0]
        if off > 1:
            mu = res[ii,12,0]
            res[ii,12,:] = proposal_u_i(mu)

    return res

def weight_particles_global(old_particles,new_particles):
    weights = np.zeros((new_particles.shape[-1],new_particles.shape[0]))
    for ii in np.arange(0,old_particles.shape[-1]):
        weights[0] = weigh_particles(old_particles[:,:,ii],new_particles[:,:,ii])[:,0]
    weights = weights.sum(axis = 0)
    weights = np.repeat(weights.reshape(new_particles.shape[0],1),new_particles.shape[-1],axis = 1)
    new_particles[:,0,:] = weights
    return new_particles

def stack_simulations(sim_old,sim_new):
    res = []
    for ii in np.arange(0,len(sim_old)):
        sims = np.dstack((sim_old[ii],sim_new[ii]))
        res.append(sims)
    return res

def stack_distances(dist_old,dist_new):
    res = []
    for ii in np.arange(0,len(dist_old)):
        dist = np.vstack((dist_old[ii],dist_new[ii]))
        res.append(dist)
    return res

def create_new_particles_below_max_dist_global(data,mock,selected_particles,max_dist,n_new,sel,
                                               deltaT = deltaT):
    max_tries = 20
    tries = 0
    n_accepted = 0
    n_created = 0
    factor = 1

    while tries < max_tries:

        if (n_new - n_accepted) < 10:
            factor = 2
        new = factor * (n_new - n_accepted)

        if tries == 0 or n_accepted == 0:
            new_particles = create_new_particles_global(selected_particles,new,sel)
        else:
            new_particles = create_new_particles_global(np.vstack((selected_particles,accepted_particles)),new,sel)
        n_created = n_created + new_particles.shape[0]

        sim,perturb = simulate_particles_global(mock,new_particles)
        dist,total_dist = distance_all_global(data,sim)

        new_particles_sort,sim_sort,per_sort,dist_sort,total_dist_sort = sort_particles_global(new_particles,
                                                                                               sim,perturb,dist,
                                                                                               total_dist)
        accepted_particle_index = np.where(total_dist_sort <= max_dist)[0]

        if len(accepted_particle_index) == 0:
            print 'Try:', tries, 'No particles accepted'

        elif len(accepted_particle_index) > 0 and n_accepted == 0:
            print 'Try:', tries, 'First particles accepted:', len(accepted_particle_index)
            accepted_particles = weight_particles_global(selected_particles,
                                                         new_particles_sort[accepted_particle_index,:,:])
            n_accepted = n_accepted + len(accepted_particle_index)

        elif len(accepted_particle_index) > 0 and n_accepted > 0:
            new_particles[accepted_particle_index,:,:] = weight_particles_global(accepted_particles,
                                                                        new_particles[accepted_particle_index,:,:])
            accepted_particles = np.vstack((accepted_particles, new_particles[accepted_particle_index,:,:]))
            n_accepted = n_accepted + len(accepted_particle_index)
            print 'Try:', tries, 'More particles accepted:', n_accepted

        if tries == 0:
            created_particles = new_particles_sort
            created_perturbations = per_sort
            created_simulations = sim_sort
            created_distances = dist_sort
            created_total_distances = total_dist_sort
        else:
            particles = np.vstack((created_particles,new_particles))
            perturbations = stack_simulations(created_perturbations,per_sort)
            simulations = stack_simulations(created_simulations,sim_sort)
            distances = stack_distances(created_distances,dist_sort)
            total_distances = np.hstack((created_total_distances,total_dist))

            created_particles,created_simulations,created_perturbations,created_distances,created_total_distances = select_best_particles_global(particles,
                                                                                                                                             simulations,
                                                                                                                                             perturbations,
                                                                                                                                             distances,
                                                                                                                                             total_distances,
                                                                                                                                             n_new)

        if n_accepted >= n_new:
            print 'Enough particles accepted:', n_accepted
            print '# Created particles:', n_created
            return [created_particles,
                    created_simulations,
                    created_distances,
                    created_total_distances,
                    created_perturbations,
                    n_accepted,
                    n_created]

        if tries == 10 and n_accepted < float(n_new) / 2:
            max_tries = 15
            print 'Slow creation of new particles, reduced number of tries to:', max_tries

        tries = tries + 1
    print 'Not enough particles accepted:', n_accepted
    print '# Created particles:', n_created
    return [created_particles,
            created_simulations,
            created_distances,
            created_total_distances,
            created_perturbations,
            n_accepted,
            n_created]


def smc_abc_global(data,mock,start_population,iterations,stop,alpha = alpha, save = [False], sel = 'T'):
    old_particles = start_population[0]
    old_simulations = start_population[1]
    old_distances = start_population[3]
    old_total_distances = start_population[4]
    dist_max_old = old_total_distances.max()
    old_perturbations = start_population[2]

    print '# particles better than stop:', np.sum(old_total_distances < stop)
    print 'Min start distance:', np.round(old_total_distances.min(), 3)
    print 'Mean start distance:', np.round(old_total_distances.mean(), 3)
    print 'Max start distance:', np.round(old_total_distances.max(), 3)
    print 'STD of start distance:', np.round(old_total_distances.std(), 3)
    print 'Mean start burst and tau:',np.round(old_particles[:,8:10,0].mean(axis = 0),3)
    print 'STD of start burst and tau:', np.round(old_particles[:,8:10,0].std(axis = 0),3)
    print 'Mean startT:',np.round(old_particles[:,10,:].mean(axis = 0),3)
    print 'STD of start T:',np.round(old_particles[:,10,:].std(axis = 0),3)

    Max_distances = []
    N_accepted = []
    N_created = []

    percentile = 0.2
    n_particles = old_particles.shape[0]
    n_old = int(percentile * n_particles)
    n_new = old_particles.shape[0] - n_old
    index_max_distance = int(percentile * n_particles)

    Run = time.time()
    ii = 0
    while ii < iterations:
        run = time.time()
        print 'Iteration:', ii + 1
        max_distance = old_total_distances[index_max_distance]
        print 'Threshold distance', np.round(max_distance, 3)
        Max_distances.append(max_distance)

        selected_particles = old_particles[0:index_max_distance, :]
        new_particles = create_new_particles_below_max_dist_global(data,mock,selected_particles,max_distance,n_new,sel)

        N_accepted.append(new_particles[5])
        N_created.append(new_particles[6])

        cand_particles = np.vstack((old_particles,new_particles[0]))
        cand_simulations = stack_simulations(old_simulations,new_particles[1])
        cand_distances = stack_distances(old_distances,new_particles[2])
        cand_total_distances = np.hstack((old_total_distances,new_particles[3]))
        cand_perturbations = stack_simulations(old_perturbations,new_particles[4])

        particles,simulations,perturbations,distances,total_distances = select_best_particles_global(cand_particles,
                                                                                                     cand_simulations,
                                                                                                     cand_perturbations,
                                                                                                     cand_distances,
                                                                                                     cand_total_distances,
                                                                                                     n_particles)
        print '# particles better than stop:', np.sum(total_distances < stop)
        print 'Min distance of current population:', np.round(total_distances.min(), 3)
        print 'Mean distance of current population:', np.round(total_distances.mean(), 3)
        print 'Max distance of current population:', np.round(total_distances.max(), 3)
        print 'Mean of burst and tau:',np.round(particles[:,8:10,0].mean(axis = 0),3)
        print 'STD of burst and tau:', np.round(particles[:,8:10,0].std(axis = 0),3)
        print 'Mean T:',np.round(particles[:,10,:].mean(axis = 0),3)
        print 'STD ofT:',np.round(particles[:,10,:].std(axis = 0),3)
        print 'Duration of', ii + 1, 'th iteration', np.round((time.time() - run) / 60, 2), 'min'

        if save[0] == True:
            np.save(save[1]+save[2]+str(ii)+'.npy',[particles,simulations,distances,total_distances,perturbations,
                                                   N_accepted,N_created])

        stop_distance = total_distances.max()
        if stop_distance <= stop:
            print 'STOP: minimal distance reached'
            print 'Duration of iterations:', np.round((time.time() - Run)/60,2), 'min'
            return [particles,simulations,distances,total_distances,perturbations,N_accepted,N_created]

        if ii > 1:
            if stop_distance > 0.99*dist_max_old:
                print 'STOP: no improvement gained, stopped after', ii, 'iterations'
                print 'Duration of iterations:', np.round((time.time() - Run)/60,2), 'min'
                return [particles,simulations,distances,total_distances,perturbations,N_accepted,N_created]
        dist_max_old = stop_distance

        old_particles = particles
        old_simulations = simulations
        old_distances = distances
        old_total_distances = total_distances
        old_perturbations = perturbations

        ii = ii + 1

    print 'Last iteration finished, max distance:',np.round(old_total_distances.max(),3)
    print 'Duration of iterations:', np.round((time.time() - Run)/60,2), 'min'
    return [particles,simulations,distances,total_distances,perturbations,N_accepted,N_created]