import scipy as sp
import numpy as np
from random import uniform
import scipy.stats as stats

# -----------------------------------------
#period length in min
tau = 40.

#-----------------------------------------
def propensities(state_vector,rates_Promoter,active_states,birth,rate_RNA):
    prop = np.zeros(len(state_vector)+1)
    prop[0] = rates_Promoter[state_vector[0] - 1]

    if (state_vector[0] >= active_states[0]) & (state_vector[0] <= active_states[-1]):
        prop[1] = birth
    #control = state_vector > 0
    prop[2:] = rate_RNA * state_vector[1:]

    return prop

def GillespieDelaymRNA_numba(nPromoter,rates_Promoter,active_states,birth,rate_RNA,initial_state,tf):
    state_vector = initial_state
    state_path = state_vector
    time = []
    tt = 0
    time.append(tt)

    while tt < tf:
        r1 = uniform(0,1)
        r2 = uniform(0,1)
        props = propensities(state_vector,rates_Promoter,active_states,birth,rate_RNA)
        a0 = props.sum()
        tt = tt + 1. / a0 * sp.log(1. / r1)
        time.append(tt)

        prop_fraction = r2 * a0
        prop_cumsum = props.cumsum()
        value = prop_cumsum[prop_cumsum > prop_fraction][0]
        reaction = [i for i,j in enumerate(prop_cumsum) if j == value]

        # make editable in parallel mode
        state_vector = state_vector.copy()

        if reaction < len(props) - 1:
            state_vector[reaction] = state_vector[reaction] + 1
            if reaction > 1:
                state_vector[reaction - 1] = state_vector[reaction - 1] - 1
        else:
            state_vector[-1] = state_vector[-1] -1

        if state_vector[0] > nPromoter:
            state_vector[0] = 1

        state_path = np.vstack((state_path,state_vector))
    return time,state_path

def GillespieDelaymRNA(nPromoter,rates_Promoter,active_states,birth,rate_RNA,initial_state,tf):
    state_vector = initial_state
    state_path = state_vector
    time = []
    tt = 0
    time.append(tt)
    time_rna = []
    time_rna.append(0)

    while tt < tf:
        rr = sp.random.uniform(0,1,2)
        props = propensities(state_vector,rates_Promoter,active_states,birth,rate_RNA)
        a0 = props.sum()
        tt = tt + 1. / a0 * sp.log(1. / rr[0])
        time.append(tt)

        prop_fraction = rr[1] * a0
        prop_cumsum = props.cumsum()
        reaction = np.where(prop_cumsum > prop_fraction)[0][0]

        # make editable in parallel mode
        state_vector = state_vector.copy()

        if reaction < len(props) - 1:
            state_vector[reaction] = state_vector[reaction] + 1
            if reaction > 1:
                state_vector[reaction - 1] = state_vector[reaction - 1] - 1
            if reaction == 1:
                time_rna.append(tt)
        else:
            state_vector[-1] = state_vector[-1] -1

        if state_vector[0] > nPromoter:
            state_vector[0] = 1

        state_path = np.vstack((state_path,state_vector))
    return time_rna,time,state_path

def GillespieDelaymRNA_Multi(nSim,nPromoter,rates_Promoter,active_states,birth,rate_RNA,initial_state,tf,view):
    N_Promoter = np.zeros(nSim) + nPromoter
    Rates_Promoter = np.zeros((nSim,len(rates_Promoter))) + rates_Promoter
    Active_states = np.zeros((nSim,len(active_states))) + active_states
    Birth = np.zeros(nSim) + birth
    Rate_RNA = np.zeros(nSim) + rate_RNA
    Initial_State = np.zeros((nSim,len(initial_state))) + initial_state
    TF = np.zeros(nSim) + tf

    Res = view.map(GillespieDelaymRNA,N_Promoter,Rates_Promoter,Active_states,Birth,Rate_RNA,Initial_State,TF)

    return Res


def GillespieStateDetTranscription(ms,mu,overlap,E2state,startBlock,endBlock,startstate,km,tf,rate,rateS,rateU):
    '''
    Corrected version of the cyclic stochastic simulation without RNA decay, only the RNA production events are of
    interest
    :param ms:
    :param mu:
    :param overlap:
    :param E2state:
    :param startBlock:
    :param endBlock:
    :param startstate:
    :param km:
    :param tf:
    :param rate:
    :param rateS:
    :param rateU:
    :return:
    '''
    active = np.arange(startBlock,endBlock+1)

    nStates = ms + mu - overlap

    branchIn = E2state - overlap
    if branchIn < 0:
        branchIn = branchIn + ms

    state = startstate
    State = []
    State.append(state)

    rna = 0
    mRNA = []
    mRNA.append(rna)

    Time = []
    tt = 0
    Time.append(tt)

    a1 = rate
    a2 = km

    while tt < tf:
        r = sp.random.uniform(0,1,2)

        if state in active:
            a0 = a1 + a2
            tt = tt + 1./ a0 * sp.log(1. / r[0])

            if r[1] * a0 <= a1:
                state = state + 1
            else:
                rna = rna + 1
        else:
            a0 = a1
            tt = tt + 1. / a0 * sp.log(1. / r[0])
            state = state + 1

        lock = 0
        if state == E2state:
            r = sp.random.uniform(0,1,1)
            if r >= rateS / (rateS + rateU):
                state = ms + 1
                lock = 1
        if state == nStates:
            state = branchIn
            lock = 0

        if state == ms and lock == 0:
            state = 1
        Time.append(tt)
        mRNA.append(rna)
        State.append(state)

    return np.array([Time,State,mRNA])

# function to stochastically simulate progression through promoter states and mRNA production and decay
def GillespieStateRNA(ms, mu, overlap, E2state, startBlock, endBlock, startState, mRNA0, birth, death, tf, rate, rateS,
                      rateU):
    '''
	Gillespie realisation of the progression of the promoter through a cyclic network of states.

	:param ms: states in the long cycle
	:param mu: states in the short cycle
	:param overlap: length of the overlap region between both cycles
	:param E2state: estrogen dependent state, i.e. the state where the decision which cycle to follow is made
	:param rate: reaction rate of all but the estrogen dependent state
	:param rateS: rate to follow the long (stimulated) cycle
	:param rateU:  rate to follow the short (unstimulated cycle)
	:param startBlock: Start of the transcriptionally active block
	:param end:  End of the transcriptionally active block
	:param mRNA0: Starting number of mRNA molecules
	:param birth: Birth rate of mRNA molecules during the on states
	:param death: Decay rate of the mRNA molecules
	:param tf: Ending time of the simulation
	:return: Array containing the [0] time, [1] the promoter state, [2] the number of mRNA molecules
	'''
    #active states
    active = np.arange(startBlock, endBlock + 1)

    # number of states
    nstates = ms + mu - overlap
    #print nstates
    branchIn = E2state - overlap
    if branchIn < 0:
        branchIn = branchIn + ms
    #print branchIn

    #initialisation
    state = startState
    State = list()
    State.append(state)

    rna = mRNA0
    mRNA = list()
    mRNA.append(mRNA0)

    trna = list()
    trna.append(0)

    # propensities
    a1 = rate
    a2 = birth

    tt = 0
    while tt < tf:
        # two random numbers
        r = sp.random.uniform(low = 0, high = 1, size = 2)
        # decay propensity
        a3 = rna * death

        # no mRNA present
        if rna == 0:
            # in an active state
            if state in active:
                a0 = a1 + a2
                tt = tt + 1 / a0 * np.log(1 / r[0])
                if r[1] * a0 < a2:
                    state = state + 1
                else:
                    rna = rna + 1

            # not in an active state
            else:
                a0 = a1
                tt = tt + 1 / a0 * np.log(1 / r[0])
                state = state + 1

        # mRNA is present
        else:
            # in an active state
            if state in active:
                a0 = a1 + a2 + a3
                tt = tt + 1 / a0 * np.log(1 / r[0])
                if r[1] * a0 > a1 + a2:
                    rna = rna - 1
                if ((r[1] * a0 > a1) & (r[1] * a0 < a1 + a2)):
                    rna = rna + 1
                else:
                    state = state + 1
            # not in an active state
            else:
                a0 = a1 + a3
                tt = tt + 1 / a0 * np.log(1 / r[0])
                if r[1] * a0 > a1:
                    rna = rna - 1
                else:
                    state = state + 1
        # include the short cut from banchOut to branchIn
        lock = 0
        if (state == E2state):
            r = sp.random.uniform(low = 0, high = 1, size = 1)
            if (r >= rateS / (rateS + rateU)):
                state = ms + 1
                lock = 1
        if state == nstates:
            state = branchIn
            lock = 0
        # close the promoter cycle
        if (state == ms + 1 and lock == 0):
            state = 1
        # append the time, state and rna variable to the corresponding lists
        trna.append(tt)
        #print rna
        mRNA.append(rna)
        State.append(state)
    return np.array([trna, State, mRNA])


# function to extract the on and off times from the simulated promoter states
def ExtractOnOff(out, start, end):
    '''
	Extract the promoter on and off times from time courses simulated using the gillespie function
	:param out: array containing [0] time, [1] promoter states, [2] (here not necessary) mRNA number
	:param start: start of the active block
	:param end: end of the active block
	:return:
	'''
    index = np.where((out[1, :] >= start) & (out[1, :] <= end))
    onStates = np.zeros(len(out[1, :]))
    onStates[index] = 1

    switches = np.diff(onStates)
    onSwitches = np.where(switches == 1)[0]
    offswitches = np.where(switches == -1)[0]

    if onStates[-1] == 0:
        on = out[0][offswitches + 1] - out[0][onSwitches + 1]
        off = out[0][onSwitches[1:] + 1] - out[0][offswitches[:-1] + 1]
    else:
        on = out[0][offswitches + 1] - out[0][onSwitches[:-1] + 1]
        off = out[0][onSwitches[1:] + 1] - out[0][offswitches + 1]
    return on, off

def Extract_ON_OFF_Multi(data,active_states):
    ON = []
    OFF  = []
    start = active_states[0]
    end = active_states[-1]
    for out in data:
        on,off = ExtractOnOff(out,start,end)
        ON.append(on)
        OFF.append(off)
    ON = np.hstack(ON)
    OFF = np.hstack(OFF)
    return ON,OFF

def Extract_mRNA_distribution(data,visible):
    mrna = data[0][1][:,visible+1].sum(axis = 1)
    ind = np.arange(1,len(data))
    for ii in ind:
        mrna = np.hstack((mrna,data[ii][1][:,visible+1].sum(axis = 1)))
    return mrna

# function that samples an 'experimental' time course from a simulated one
def ExperimentalTC(out, deltaT):
    '''
	Extract from a simulated mRNA time course the signal that would have been seen in an experiment with a time
	resolution deltaT

	:param out:  array containing [0] time, [1] promoter states, [2] (here not necessary) mRNA number
	:param deltaT: experimental time resolution in minutes
	:return: mRNA number and time vector
	'''
    texp = np.arange(0, np.max(out[0]), deltaT)
    res = np.zeros(len(texp))
    count = 0
    while count < len(texp):
        ind = np.where(out[0] <= texp[count])
        res[count] = out[2][max(ind[0])]
        count = count + 1
    return res, texp

def ExperimentalTC_Multi(data,deltaT,tf,visible):
    texp = np.arange(0,tf,deltaT)
    res = np.zeros((len(texp),len(data)))
    ind = 0
    while ind < len(data):
        tt = data[ind][0]
        states = data[ind][1]
        out = np.zeros((3,len(tt)))
        out[0] = tt
        out[2] = states[:,visible+1].sum(axis = 1)
        tc,t = ExperimentalTC(out, deltaT)
        index = np.where(t == texp)
        res[:,ind] = tc[0:len(texp)]
        ind = ind + 1
    return res,texp

# runs faster than the function above, the results are identical
def AutoCorrelate(signal, deltaT):
    '''
	Calculate the autocorrelation function
	:param signal: 1d array, input signal
	:param deltaT: , float, experimental time resolution
	:return: 2D array, 1st column: lag, 2nd column: acf(lag)
	'''
    signal = signal - np.mean(signal)
    res = np.correlate(signal, signal, mode = 'full')
    res = res / np.max(res)
    out = np.zeros((2, len(signal)))
    out[0] = np.arange(0, len(signal)) * deltaT

    out[1] = res[np.argmax(res):]
    return out

def AutocorrelateAllData(DataTC, deltaT):
    '''
    Calculate the autocorrelations for all extracted signals
    :param DataTC: 2D numpy array with the time courses in columns
    :param deltaT: imaging time interval
    :return: 2d numpy array with the calculated autocorrelations in columns
    '''
    res = np.empty((DataTC.shape))
    count = 0
    while count < DataTC.shape[1]:
        res[:, count] = AutoCorrelate(DataTC[:, count], deltaT)[1]
        count = count + 1
    return res


def AutoCorrelateMulti(data):
    autocorrelations = np.zeros(data.shape)
    autocorrelations[:,0] = data[:,0]
    deltaT = 1#data[1,0] - data[0,0]
    count = 0
    while count < data.shape[1]:
        autocorrelations[:,count] = AutoCorrelate(data[:,count],deltaT)[1]
        count = count + 1

    return autocorrelations

def CrossCorrelate(signal1, signal2, delaT):
    '''
	Calculate the crosscorrelation between the signal of two cells
	:param signal1: 1d array, signal from cell 1
	:param signal2: 1d array, signal from cell 2
	:param delatT: float, time resolution of the experiment
	:return:
	'''
    signal1 = signal1 - np.mean(signal1)
    signal2 = signal2 - np.mean(signal2)
    res = np.correlate(signal1, signal2, mode = 'full')
    res = res / signal1.std() / signal2.std()
    out = np.zeros((2, len(res)))
    out[0] = np.arange(-len(signal1) + 1, len(signal2)) * delaT
    out[1] = res
    return out


# function so simulate multiple promoter state and mRNA time courses in parallel
def GillespiStateRNAparallel(branchOut, branchIn, start, end, mRNA0, birth, death, tf, rates, rateShort, m, sims):
    # write the Gillespie inputs into arrays of length sims
    TF = np.zeros(sims) + tf
    BRANCHin = np.zeros(sims) + branchIn
    BRANCHout = np.zeros(sims) + branchOut
    START = np.zeros(sims) + start
    END = np.zeros(sims) + end
    MRNA0 = np.zeros(sims) + mRNA0
    BIRTH = np.zeros(sims) + birth
    DEATH = np.zeros(sims) + death
    RATES = np.zeros(sims) + rates
    RATEshort = np.zeros(sims) + rateShort
    M = np.zeros(sims) + m

    out = view.map(GillespieStateRNA, BRANCHin, BRANCHout, START, END, MRNA0, BIRTH, DEATH, TF, RATES, RATEshort, M)

    return out


# function to sort the simulated states and mRNA values into bins and average
def MeanRNAState(out):
    S = np.zeros((3, len(out), 100000))
    count = 0
    while count < len(out):
        S[:, count, 0:out[count].shape[1]] = out[count]
        count = count + 1

    bins = sp.linspace(0, tf, len(tChIP))
    MeanTime = np.zeros(len(tChIP) - 1)
    SDtime = np.zeros(len(tChIP) - 1)
    MeanState = np.zeros(len(tChIP) - 1)
    SDstate = np.zeros(len(tChIP) - 1)
    MeanRNA = np.zeros(len(tChIP) - 1)
    SDrna = np.zeros(len(tChIP) - 1)
    count = 0
    while count < len(bins) - 1:
        index = np.where((S[0] > bins[count]) & (S[0] <= bins[count + 1]))
        MeanTime[count] = sp.mean(S[0][index])
        SDtime[count] = sp.std(S[0][index])
        MeanState[count] = 100 * sp.mean(S[1][index]) / m
        SDstate[count] = 100 * sp.std(S[1][index]) / m
        MeanRNA[count] = sp.mean(S[2][index])
        SDrna[count] = sp.std(S[2][index])
        count = count + 1
    return np.array([MeanTime, SDtime, MeanRNA, SDrna, MeanState, SDstate])


# ideal signal generated by one polymerase creating a transcript
def polymerase(speed, promoterLength, stemStart, nStems, deltaT, splice, signalScale):
    '''
    Generate a hybrid deterministic-stochastic signal corresponding to one single transcript
    :param speed: float, speed of the RNA polymerase in bp/min
    :param promoterLength: int, total length of the promoter in bp
    :param stemStart: int, positon where in the promoter the stem loops start
    :param nStems: int, number of stem loops
    :param deltaT: float, experimental time resolution
    :param splice: float, splicing rate after the first exon
    :return: two 1d arrays, 1st: time vector, 2nd: fluorescence signal
    '''

    PolTime = promoterLength / speed
    PolTimeVector = np.linspace(0, PolTime, promoterLength)
    promoter = np.zeros(promoterLength)
    stemEnd = stemStart + nStems * 60
    intron  = stemEnd + 7680
    promoter[stemStart:stemEnd + 1] = 1
    fluorescence = np.cumsum(promoter)
    fluorescence[-1] = 0

    #splicing
    rr = sp.random.uniform(low = 0, high = 1, size = 1)
    #print rr
    bases = sp.arange(intron,promoterLength) - intron
    weights = sp.exp(-splice*bases)
    #pl.plot(bases,weights)
    ind = np.where(weights <= rr)[0]
    fluorescence[ind+intron] = 0

    #coarse time sampling
    diffT = PolTimeVector[1] - PolTimeVector[0]
    sample = int(deltaT / diffT)
    index = np.arange(0, len(PolTimeVector), sample)
    #print index
    return PolTimeVector[index], signalScale * fluorescence[index]


# function to extract experimental time course data
def SumPolymerase(deltaT, times, signal):
    Signals = np.zeros(times.shape) + signal
    texp = np.arange(0, times.max(), deltaT)
    diffT = times[0, 1] - times[0, 0]
    res = np.zeros((2, len(texp)))
    res[0] = texp
    count = 0
    for tt in texp:
        upper = times <= tt
        lower = times >= tt - diffT
        ind = upper * lower
        res[1, count] = sum(Signals[ind])
        count = count + 1
    #print (sum(ind), tt)
    return res


def ExperimentalFluorescenceTC(GillOut, speed, length, stemStart, nStems, deltaT, tf, tau, splice,signalScale):
    '''
    Use the stochastic transcription initiation events from the gillespie simulation to calculate the actual 'experimental'
    time course. The signal of one transcript is generated by the function polymerase.
    :param GillOut: 2d array with 3 rows. 3rd row is the mRNA time course
    :param speed: float, speed of the polymerase 2.5 kb/min usually
    :param length: float, length of the reporter
    :param stemStart: float, start position of the stem loops in the reporter
    :param nStems: int, number of stem loops
    :param deltaT: float, experimental time resolution
    :param tf: float, termination time of the gillespie simulation
    :param tau: float, period of one promoeter cycle
    :return: list, 1st: 1d array, time vector, 2nd: 1d array, summed signal, 3rd: 2d array signal of each single transcript
    '''
    #polTime, signal = polymerase(speed, length, stemStart, nStems, deltaT, splice,signalScale)

    # extract the transcription initiation events form the stochastic mRNA time courses
    tr = np.diff(GillOut[2])
    ind = np.where(tr == 1)
    tevents = GillOut[0][ind[0] - 1]

    times = np.arange(0, tf + 2*tau, deltaT)

    Signals = np.zeros((len(tevents), len(times)))
    cc = 0
    for te in tevents:
        ttdiv = te / deltaT
        ttmod = te % deltaT
        if ttmod < deltaT / 2.:
            ttadd = int(ttdiv)
        else:
            ttadd = int(ttdiv) + 1

        polTime, signal = polymerase(speed, length, stemStart, nStems, deltaT, splice,signalScale)
        Signals[cc, ttadd:(ttadd + len(signal))] = signal
        cc = cc + 1
    return [times, Signals.sum(axis = 0), Signals]


def RandomTelegraph(startState, mRNA0, kon, koff, km, death, tf):
    '''
    Gillespie implementation of a transcriptional random telegraph model
    :param startState: int, 1 or 0, initial state
    :param mRNA0: int, initial amount of mRNA
    :param kon: float, switching rate from off to on
    :param koff: float, switching rate from on to off
    :param km: float, transcription rate
    :param death: float, decay rate of mRNA
    :param tf: float, termination time of the simulation
    :return: 2d array, 1st time, 2nd promoter state, 3rd mRNA amount
    '''
    #initiallise
    mRNA = []
    rna = mRNA0
    mRNA.append(rna)
    State = []
    state = startState
    State.append(state)
    Time = []
    tt = 0
    Time.append(tt)

    # propensities
    a1 = kon
    a2 = koff
    a3 = km

    while tt < tf:
        # two random numbers
        r = sp.random.uniform(low = 0, high = 1, size = 2)
        # decay propensity
        a4 = rna * death

        # no mRNA present
        if rna == 0:
            # in inactive state
            if state == 0:
                a0 = a1
                tt = tt + 1 / a0 * np.log(1 / r[0])
                state = 1
            # in an active state
            else:
                a0 = a2 + a3
                tt = tt + 1 / a0 * np.log(1 / r[0])
                # select reaction
                if r[1] * a0 < a2:
                    #print 'true'
                    state = 0
                else:
                    rna = rna + 1

        # RNA already present
        else:
            if state == 1:
                a0 = a2 + a3 + a4
                tt = tt + 1 / a0 * np.log(1 / r[0])
                #select reaction
                if r[1] * a0 < a2:
                    state = 0
                elif ((r[1] * a0 > a2) & (r[1] * a0 < a2 + a3)):
                    rna = rna + 1
                else:
                    rna = rna - 1
            else:
                a0 = a1 + a4
                tt = tt + 1 / a0 * np.log(1 / r[0])
                if r[1] * a0 < a1:
                    state = 1
                else:
                    rna = rna - 1
        Time.append(tt)
        State.append(state)
        mRNA.append(rna)
    return np.array([Time, State, mRNA])

def RandomTelegraphDetTranscription(startState,kon,koff,km,tf):
    #initialise
    rna = 0
    mRNA = []
    mRNA.append(rna)
    State = []
    state = startState
    State.append(state)
    Time = []
    tt = 0
    Time.append(tt)
    Events = []

    #propensities
    a1 = kon
    a2 = koff
    a3 = km

    while tt < tf:
        # two random number from [0,1]
        r = sp.random.uniform(0,1,2)

        if state == 1:
            a0 = a2 + a3
            tt = tt + 1. / a0 + sp.log(1. / r[0])

            if r[1] * a0 <= a2:
                state = 0
            else:
                rna = rna + 1
                Events.append(tt)
        else:
            a0 = a1
            tt = tt + 1. / a0 + sp.log(1. / a0)
            state = 1
        Time.append(tt)
        mRNA.append(rna)
        State.append(state)
    return np.array([Time,State,mRNA]),Events

def AutoCorrHalfLife(auto, deltaT):
    '''
    Extract the lag time where the autocorrelation drops to 0.5
    :param auto:
    :param deltaT:
    :return:
    '''
    index = np.where(auto >= 0.5)[0]
    x1 = index[-1]*deltaT
    y1 = auto[index[-1]]
    x2 = x1 + deltaT
    y2 = auto[index[-1]+1]
    m = (y2 - y1)/(x2 - x1)
    b = y1 - m*x1
    HL = (0.5 - b)/m
    return HL

def ACF_HL_multi(ACF):
    res = np.zeros(ACF.shape[1])
    index = np.arange(0,ACF.shape[1])
    for ii in index:
        res[ii] = AutoCorrHalfLife(ACF[:,ii],1)
    return res

def cell_to_cell_kde(data,mock):
    score = (data - mock.mean())/mock.std()
    x = sp.linspace(-4,50,100)
    res = np.zeros((data.shape[1],len(x)))
    index = np.arange(0,data.shape[1])
    for ii in index:
        dd = score[:,ii]
        pdf = sp.stats.gaussian_kde(dd)
        res[ii,:] = pdf(x)

    return res,x

def max_density(pdf,x):
    maxima = pdf.max(axis = 1)
    res = np.zeros(len(maxima))
    for ii,mm in enumerate(maxima):
        ind = np.where(pdf[ii,:] == mm)[0]
        res[ii] = x[ind]
    return res
