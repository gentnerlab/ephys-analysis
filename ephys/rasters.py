from spiketrains import get_spiketrain
import core
import events
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt 


def do_raster(raster_data, times, ticks, ax=None, spike_linewidth=1.5, 
              spike_color='k', tick_linewidth=1.5, tick_color='r'):
    '''
    Generalized raster plotting function

    Parameters
    ------
    raster_data : list of lists of floats
        List of lists.  Each sublist corresponds to one row of events
        Each element of a sublist is an event times
    times : list of floats
        The beginning and end times to plot 
    ticks : list of floats
        Will add a vertical tick across the whole plot for each time in list
    ax : Matplotlib axes handle, optional 
        Axes on which to produce raster. Default gca.
    spike_linewidth : float, optional   
        width in points of ticks for spikes
    spike_color : str 
        color of ticks for spikes 
    tick_linewidth : float 
        width in points of ticks for events 
    tick_color  : str 
        color of ticks for events

    Returns
    ------
    raster_plot : 
        Handle to the raster plot 
    '''
    ntrials = len(raster_data)
    if ax is None:
        ax = plt.gca()
    ax.set_xlim(times)
    ax.set_ylim((1, ntrials+1))
    for trial, trialdata in enumerate(raster_data):
        ypts = [1+trial, 2+trial]
        for spiketime in trialdata:
            ax.plot([spiketime, spiketime], ypts, spike_color, 
                    lw=spike_linewidth)
    for pltticks in ticks:
        ax.plot([pltticks, pltticks], [1, ntrials+1], tick_color, 
                lw=tick_linewidth)
    return ax


def plot_raster_cell_stim(spikes, trials, clusterID, 
                          stim, period, fs, **kwargs):
    '''
    Plots a spike raster for a single cell and stimulus 
    Finds all the recordings and trials of a single stimulus 
    No need to specify specific recordings

    Parameters
    ------
    spikes : pandas dataframe
        spike dataframe from core 
    trials : pandas dataframe
        trials dataframe from events
    clusterID : int
        ID number of the cluster you wish to make the raster for 
    stim : str 
        Name of the stimulus you wish to plot cluster's activity for 
    period : list of floats 
        Time window for the raster:  
        [Seconds_pre_stimulus_onset, Seconds_post_stimulus_end]
    fs : float 
        Sampling rate
    plot_params : dict
        Drawing parameters:
        'spike_linewidth' - linewidth of ticks for spikes 
        'tick_linewidth' - linewidth of ticks for event markers
        'spike_color' - color of spike ticks 
        'tick_color' - color of event ticks 
    ax : Matplotlib axes handle, optional
        Axes on which to produce the raster.  Default is to use gca 
    ''' 
    stim_trials = trials[trials['stimulus']==stim]
    ntrials = len(stim_trials)
    stim_starts = stim_trials['time_samples'].values
    stim_ends = stim_trials['stimulus_end'].values
    stim_end_seconds = np.unique((stim_ends - stim_starts)/fs)[0]
    window = [period[0], stim_end_seconds+period[1]]
    stimrecs = stim_trials['recording'].values
    raster_data = []
    for trial, [start, rec] in enumerate(zip(stim_starts, stimrecs)):
        sptrain = get_spiketrain(rec, start, clusterID, spikes, window, fs)
        raster_data.append(sptrain)
    do_raster(raster_data, window, [0, stim_end_seconds], **kwargs)

def gaussian_psth_func(times, spike_data, sigma):
    '''
    Generates a gaussian psth from spike data 

    Parameters
    ------
    times : numpy array 
        times to generate psth for 
    spike_data : list of floats 
        times of each spike 
    sigma : float   
        standard deviation of the gaussian 
    '''
    output = np.zeros(len(times))
    for spike_time in spike_data:
        output = output+np.exp(-1.0*np.square(times-spike_time)/(2*sigma**2))
    return output

def calc_avg_gaussian_psth(spikes, trials, clusterID, stim, period, rec, fs, sigma=0.05, alpha=0.95):
    ''' 
    Calculates a gaussian smoothed average psth over all trials of stim for a given cluster.

    Parameters 
    ------
    spikes : dataframe 
        spike data
    trials : dataframe 
        trial data 
    clusterID : int 
        cluster ID to compute 
    stim : str 
        Stim name 
    period : list of floats 
        period in seconds before/after stim 
    rec : int 
        recording id 
    fs : float 
        sampling rate 
    sigma : float 
        stand deviation for gaussian 
    alpha : float 
        confidence level 

    Returns 
    ------
    avg_psth : numpy array 
        the average gaussian psth 
    std_psth :
        standard deviation of the psth 
    conf_ints :
        confidence intervals 
    times : 
        times for the signals 
    '''

    stim_trials = trials[trials['stimulus']==stim]
    ntrials = len(stim_trials)
    stim_starts = stim_trials['time_samples'].values
    stim_ends = stim_trials['stimulus_end'].values

    stim_end_seconds = np.unique((stim_ends - stim_starts)/fs)[0]
    window = [period[0], stim_end_seconds+period[1]]
    npts = np.floor(1.0*(window[1]-window[0])*fs)
    times = np.linspace(window[0], window[1], npts)
    psths = np.zeros((ntrials, npts))
    for trial, start in enumerate(stim_starts):
        sptrain = get_spiketrain(rec, start, clusterID, spikes, window, fs)
        psths[trial, :] = gaussian_psth_func(times, sptrain, sigma)
    avg_psth = np.mean(psths, 0)
    std_psth = np.std(psths, 0)
    conf_ints = stats.t.interval(alpha, df=ntrials-1, loc=avg_psth, scale=std_psth/np.sqrt(ntrials))
    return (avg_psth, std_psth, conf_ints, times)

def plot_unit_raster(spikes, trials, clusterID, raster_window, rec, fs, subplot_xy, figsize, plot_params=None, fontsize=20):
    ''' 
    Plots a raster of all trials of all stimuli from a given unit 
    '''

    stims = trials['stimulus'].unique()

    f, pltaxes = plt.subplots(subplot_xy[0], subplot_xy[1], sharey=True, figsize=figsize)
    for ind, stim in enumerate(stims):
        ax = pltaxes.flatten()[ind]
        plot_raster_cell_stim(spikes, trials, clusterID, stim, 
                              raster_window, rec, fs, plot_params=plot_params, ax=ax)
        ax.set_title('Unit: {} Stim: {}'.format(str(clusterID), stim))
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Repetition')
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(fontsize)
    return f

def plot_all_cells(spikes, trials, clusters, quality, raster_window, fs):
    '''
    Plots rasters for all cells of a given quality.
    The rasters are organized by stimuli; each subplot showing all trials from a given stimulus

    Parameters 
    ------
    spikes : dataframe 
        spike data
    trials : dataframe 
        trial data 
    clusters : dataframe
        cluster data 
    quality : str 
        quality identifier string ('Good', 'MUA', etc) 
    raster_window : list 
        Time before stimulus start and after stimulus end to include in plot (e.g. [-2, 2] for 2 secs before and after)
    fs : int 
        sampling rate 
    '''

    clusToPlot = clusters[clusters['quality']==quality]
    stims = trials['stimulus'].unique()
    nstims = len(stims)
    for clu in clusToPlot['cluster'].values:
        NplotRows = int(np.ceil(nstims/4.0))
        f, pltaxes = plt.subplots(NplotRows, 4, sharey=True, figsize=(22,18))
        f.tight_layout()
        for ind, stim in enumerate(stims):
            ax = pltaxes.flatten()[ind]
            rasters.plot_raster_cell_stim(spikes, trials, clu, stim, 
                                  raster_window, fs, ax=ax)
            ax.set_title('Cell: {} Stim: {}'.format(clu, stim))

def plot_avg_gaussian_psth_cell_stim(spikes, trials, clusterID, stim, raster_window, rec, fs, ax=None):
    return 0

def plot_unit_gaussian_psth(spikes, trials, clusterID, raster_window, rec, fs, subplot_xy, figsize):
    ''' 
    Plots average psth gaussian smoothed
    '''

    stims = trials['stimulus'].unique()
    f, pltaxes = plt.subplots(subplot_xy[0], subplot_xy[1], sharey=True, figsize=figsize)


