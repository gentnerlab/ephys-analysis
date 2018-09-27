'''
from UltraMegaSort2000 by Hill DN, Mehta SB, & Kleinfeld D  - 07/12/2010
ported to Python by Justin Kiggins - Apr 2016

imported GaussianMixture as GMM, not sure what sklearn version was originally
but if this doesn't work that might be why. - Marvin Thielk 9/18
'''
import numpy as np
from scipy.stats import norm, chi2
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture as GMM


def censored(tau_c, M, T):
    '''
    returns the expected false negative rate due to censoring


    Description
    -----
    Estimates the percent of a cluster that is lost due to censoring.  Every
    detected spike event turns off spike detection for a period `tau_c`.  This
    simple function estimates the percent of false negative errors in a cluster
    by calculating what fraction of the entire data set was censored by
    detected events that are external to this cluster.

    If `tau_c` is the duration of the censored period, `M` is the number of other
    detected events, and `T` is the duration of the recording period, then the
    fraction of the experiment that was censored is simply `tau_c * M / t`.

    Parameters
    ----- 
    tau_c : float
        censored period (ms)
    M : int  
        number of events detected outside of cluster of interest
    T : float 
        total duration of experiment (s)

    Returns
    ------
    c 
        estimated false negative fraction from censoring
    '''
    return (tau_c / 1000.0) * M / T


def gaussian_overlap(w1, w2):
    '''
    estimate cluster overlap from a 2-mean Gaussian mixture model

    Description  
    -------
    Estimates the overlap between 2 spike clusters by fitting with two
    multivariate Gaussians.  Implementation makes use of scikit learn 'GMM'. 

    The percent of false positive and false negative errors are estimated for 
    both classes and stored as a confusion matrix. Error rates are calculated 
    by integrating the posterior probability of a misclassification.  The 
    integral is then normalized by the number of events in the cluster of
    interest. See description of confusion matrix below.

    NOTE: The dimensionality of the data set is reduced to the top 99% of 
    principal components to increase the time efficiency of the fitting
    algorithm.

    Parameters
    --------
    w1 : array-like [Event x Sample ] 
        waveforms of 1st cluster
    w2 : array-like [Event x Sample ] 
        waveforms of 2nd cluster

    Returns
    ------
    C 
        a confusion matrix

    C[0,0] - False positive fraction in cluster 1 (waveforms of neuron 2 that were assigned to neuron 1)
    C[0,1] - False negative fraction in cluster 1 (waveforms of neuron 1 that were assigned to neuron 2)
    C[1,0] - False negative fraction in cluster 2 
    C[1,1] - False positive fraction in cluster 2
    '''
    # reduce dimensionality to 98% of top Principal Components
    N1 = w1.shape[0]
    N2 = w2.shape[0]

    X = np.concatenate((w1, w2))
    pca = PCA()
    pca.fit(X)
    Xn = pca.transform(X)

    cutoff = 0.98
    num_dims = (np.cumsum(pca.explained_variance_ratio_) < cutoff).sum()

    w1 = Xn[:N1, :num_dims]
    w2 = Xn[N1:, :num_dims]

    # fit 2 multivariate gaussians
    gmm = GMM(n_components=2)
    gmm.fit(np.vstack((w1, w2)))

    # get posteriors
    pr1 = gmm.predict_proba(w1)
    pr2 = gmm.predict_proba(w2)

    # in the unlikely case that the cluster identities were flipped during the
    # fitting procedure, flip them back
    if pr1[:, 0].mean() + pr2[:, 1].mean() < 1:
        pr1 = pr1[:, [1, 0]]
        pr2 = pr2[:, [1, 0]]

    # create confusion matrix
    confusion = np.zeros((2, 2))

    confusion[0, 0] = pr1[:, 1].mean()   # probability that a member of 1 is false
    # relative proportion of spikes that were placed in cluster 2 by mistake
    confusion[0, 1] = pr2[:, 0].sum() / N1
    confusion[1, 1] = pr2[:, 0].mean()   # probability that a member of 2 was really from 1
    # relative proportion of spikes that were placed in cluster 1 by mistake
    confusion[1, 0] = pr1[:, 1].sum() / N2

    return confusion


def poissfit(data, alpha=0.05):
    '''
    estimates the parameters of a poisson distribution

    Parameters
    ---------
    data : array-like
        the data to fit. if 2D, then parameters will be estimated for each column
    alpha : float (optional)
        the alpha value of the confidence intervals

    Returns
    -------
    lambdahat : float or np.array
        the estimate of lambda
    lambdaci : np.array 
        the confidence intervals 

    '''

    r = np.array(data)

    lambdahat = r.mean(axis=0)

    def lower(lh):
        return chi2.ppf(alpha / 2, 2 * lh * r.shape[0]) / (2 * r.shape[0])
    lower = np.vectorize(lower)

    def upper(lh):
        return max(0.0, chi2.ppf(1 - alpha / 2, 2 * lh * r.shape[0] + 2) / (2 * r.shape[0]))
    upper = np.vectorize(upper)

    lambdaci = np.array((
        lower(lambdahat),
        upper(lambdahat)
    ))

    return lambdahat, lambdaci


def rpv_contamination(N, T, RP, RPV, alpha=0.05):
    '''
    get range of contamination


    Description
    -----------

    Estimates contamination of a cluster based on refractory period
    violations (RPVs).  Estimate of contamination assumes that the 
    contaminating spikes are statistically independent from the other spikes
    in the cluster.  Estimate of the confidence interval assumes Poisson
    statistics.

    Refractory period violations must arise from spikes that were not 
    generated by the neuron that a cluster represents.  We calculate the rate
    of these "rogue" events by dividing the total number of RPV's by the total
    period in which an RPV can occur.  For every true spike in a cluster, if
    a rogue spike occurs immediately before or after, this causes a RPV.
    Therefore, for every true spike in a cluster, there is a period

        tau_rpv = 2*(tau_r - tau_c) 

    when refactory period violations can occur if a rogue spike is present, 
    where tau_r is the user-defined refractory period, and tau_c is the 
    user-defined shadow (censored) period. Therefore, the total time in which
    an RPV can occur is 

        T_rpv = N(1-P)tau_rpv 

    where N is the number of spikes in the cluster and P is the probability 
    that a spike is a "rogue" spike.  Finally, we estimate the rate of 
    contamination as

       lambda_rogue = RPV / T_rpv

    where RPV is the total number of observed RPV's.  Finally, noting that

       lambda_rogue =  p * N / T

    where T is duration of the experiment, we can perform substitution to
    solve for p and plugging in the values for N, T, RPV, tau_r, and tau_c.

    Parameters
    ---------- 
    N : int
        Number of spike events in cluster
    T : float 
        Duration of recording (s)
    RP : float 
        Duration of useable refractory period, tau_rp - tau_c (s) (Remember to subtract censor period!)
    RPV : int 
        Number of observed refractory period violations in cluster

    Returns
    -------
    ev : float
        expected value of % contamination,
    lb : float
        lower bound on % contamination, using alpha confidence interval
    ub : float
        upper bound on % contamination, using alpha confidence interval
    '''

    lambda_ = N / T  # mean firing rate for cluster

    # get Poisson confidence interval on number of expected RPVs
    dummy, interval = poissfit(RPV, alpha)

    # convert contamination from number of RPVs to a percentage of spikes
    lb = convert_to_percentage(interval[0], RP, N, T, lambda_)
    ub = convert_to_percentage(interval[1], RP, N, T, lambda_)
    ev = convert_to_percentage(RPV, RP, N, T, lambda_)

    return ev, lb, ub


def convert_to_percentage(RPV, RP, N, T, lambda_):
    # converts contamination from number of RPVs to a percentage of spikes

    RPVT = 2 * RP * N  # total amount of time in which an RPV could occur
    RPV_lambda = RPV / RPVT  # rate of RPV occurence
    p = RPV_lambda / lambda_  # estimate of % contamination of cluster

    # force p to be a real number in [0 1]
    if np.isnan(p):
        p = 0
    elif p > 1:
        p = 1
    return p


def undetected(w, threshes, criteria_func='auto'):
    '''
    UltraMegaSort2000 by Hill DN, Mehta SB, & Kleinfeld D  - 07/12/201

    undetected - estimate fraction of events that did not reach threshold

    Usage:
        [p,mu,stdev,n,x] = undetected(waveforms,threshes,criteria_func)

    Description:  
    Estimates fraction of events that did not reach threshold by applying 
    the detection metric to each waveform and then fitting it with a Gaussian 
    that has a missing tail.

    The distribution of detection metric values is turned into a histogram.  
    A Gaussian is fit to the historgram to minimize the absolute error 
    between the Gaussian and the histogram for values above threshold.  
    The integral of this Gaussian that is below threshold is the estimate of 
    the fraction of missing events.

    Note that values are normalized so that the threshold is +/- 1.  The function
    attempts to preserve the sign of the original threshold, unless thresholds
    on different channels had different signs. In the case of multiple channels, 
    each channel is normalized so that the threshold has a magnitude of 1.  Then, 
    for each event, only the channel with the most extreme value of the detection 
    metric is used. 

    By default, this function assumes that a simple voltage crossing was used 
    for detection, but see "criteria_fun" below for alternatives. In the case 
    of a simple voltage threshold, note that the threshold is interpreted as 
    responding to crossings away from zero, i.e., negative thresholds 
    imply negative-going crossings and positive thresholds imply 
    positive-going crossings. 


    Input:
        waveforms  - [Events X Samples X Channels] the waveforms of the cluster
        threshes   - [1 X Channels] the threshold for each channel
        criteria_func - Used to determine what the detection metric is on each
                       waveform.  If this is the string "auto" or "manual" then
                       it is assumed that a simple voltage threshold was used. 
                       The detection criterion then is to divide each channel
                       by its threhsold and use the maximum value.  Otherwise
                       the criteria_func is assumed to be a function handle that
                       takes in waveforms and threshes and returns the detection
                       metric for each event [Events x 1].  The function will
                       be called as
                          criteria = criteria_func( waveforms, threshes)
                       It is assumed that the values of criteria are normalized 
                       to use a threshold value of + 1.

    Output:
        p            - estimate of probability that a spike is missing because it didn't reach threshhold
        mu           - mean estimated for gaussian fit
        stdev        - standard deviation estimated for gaussian fit
        n            - bin counts for histogram used to fit Gaussian
        x            - bin centers for histogram used to fit Gaussian

    '''

    # constant bin count
    bins = 75

    # check for detection method
    if (criteria_func == 'auto') or (criteria_func == 'manual'):
        # normalize all waveforms by threshold
        w /= threshes

        # get maximum value on each channel
        criteria = w.max(axis=2)
    else:
        criteria = criteria_func(w, threshes)

    # create the histogram values
    global_max = criteria.max()
    mylims = np.linspace(1, global_max, bins + 1)
    x = mylims + (mylims[1] - mylims[0]) / 2
    n = histc(criteria, mylims)

    # fit the histogram with a cutoff gaussian
    # use mode instead of mean, since tail might be cut off
    m = mode_guesser(criteria, .05)
    stdev, mu = stdev_guesser(criteria, n, x, m)  # fit the standard deviation as well

    # Now make an estimate of how many spikes are missing, given the Gaussian and the cutoff
    p = norm.cdf(1, mu, stdev)

    # attempt to keep values negative if all threshold values were negative
    if (threshes < 0).all():
        mu *= -1
        x *= -1

    return p, mu, stdev, n, x


def stdev_guesser(thresh_val, n, x, m):
    '''
    fit the standard deviation to the histogram by looking for an accurate
    match over a range of possible values
    '''
    # initial guess is juts the RMS of just the values below the mean
    init = m + np.sqrt(np.mean((m - (n * x)[thresh_val <= m])**2))
    print init

    num = 100
    factor = 10
    st_guesses = np.linspace(max(init / factor, 0.001), init * factor / 5, num)
    m_guesses = np.linspace(m - init, max(m + init, 1), num)
    error = np.empty((m_guesses.shape[0], st_guesses.shape[0]))
    for j in range(len(m_guesses)):
        for k in range(len(st_guesses)):
            b = stats.norm.pdf(x, m_guesses[j], st_guesses[k])
            b = b * np.sum(n) / np.sum(b)
            error[j, k] = np.sum(np.abs(b - n))

    plt.xlabel('stdev')
    plt.ylabel('mean')

    # which one has the least error?
    pos = error.argmin()
    jpos, kpos = np.unravel_index(pos, error.shape)

    # refine mode estimate
    stdev = st_guesses[kpos]
    m = m_guesses[jpos]

    return stdev, m
