from __future__ import absolute_import
from __future__ import print_function
from ephys.spiketrains import get_spiketrain
from ephys import core
from ephys import events
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from six.moves import zip


def do_raster(raster_data, times, ticks, ntrials, ax=None, spike_linewidth=1.5,
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
    raster_data = np.array(raster_data)
    if ax is None:
        ax = plt.gca()
    ax.set_xlim(times)
    ax.set_ylim((-0.5, ntrials-0.5))
    ax.set_yticks(range(0, ntrials))
    ax.eventplot(raster_data, linewidths=spike_linewidth, colors=spike_color)
    for pltticks in ticks:
        ax.axvline(pltticks, color=tick_color)
    return ax


def plot_raster_cell_stim(spikes, trials, clusterID,
                          stim, period, rec, fs, ax=None, stim_ref='stim', **kwargs):
    '''
    Plots a spike raster for a single cell and stimulus

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
    rec : int
        Recording ID
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
    kwargs :
        keyword arguments are passed to the do_raster method
    '''
    stim_trials = trials[trials['stimulus'] == stim]
    ntrials = len(stim_trials)
    stim_starts = stim_trials['time_samples'].values
    stim_ends = stim_trials['stimulus_end'].values
    stim_end_seconds = np.unique((stim_ends - stim_starts) / fs)[0]
    if stim_ref == 'stim':
        window = [period[0], stim_end_seconds + period[1]]
    elif stim_ref == 'abs':
        window = [period[0], period[1]]
    raster_data = []
    for trial, start in enumerate(stim_starts):
        sptrain = get_spiketrain(rec, start, clusterID, spikes, window, fs)
        raster_data.append(sptrain)
    ax = do_raster(raster_data, window, [0, stim_end_seconds], ntrials, ax, **kwargs)
    return ax

def plot_all_rasters(block_path):
    ''' Plots all the rasters from all units for all stimuli 
        Places them in a blockpath/rasters folder
    '''
    rasters_folder = os.path.join(block_path, 'rasters/')
    spikes = core.load_spikes(block_path)
    trials = core.load_trials(block_path)
    fs = core.load_fs(block_path)
    stims = np.unique(trials['stimulus'].values)

    os.makedirs(rasters_folder, exist_ok=True)
    for cluster in tqdm.tqdm(clusters):
        os.makedirs(os.path.join(rasters_folder, '{}/'.format(cluster), exist_ok=True))
        for stim in tqdm.tqdm(stims):
            fig = plt.figure()
            ax = plot_raster_cell_stim(spikes, trials, cluster, stim, [-2, 2], 0, fs)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Trial Number')
            ax.set_title('Unit: {}  Stimulus: {}'.format(cluster, stim))
            plt.savefig(os.path.join(rasters_folder, '/{}/unit-{}_stim-{}.pdf'.format(cluster, cluster, stim)))
            plt.close(fig)


def plot_raster_cell_stim_emily(spikes, trials, clusterID,
                                stim, period, rec, fs, ax=None, **kwargs):
    '''
    Plots a spike raster for a single cell and stimulus

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
    rec : int
        Recording ID
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
    kwargs :
        keyword arguments are passed to the do_raster method
    '''
    stim_trials = trials[trials['stimulus'] == stim]
    stim_recs = stim_trials['recording'].values
    ntrials = len(stim_trials)
    stim_starts = stim_trials['time_samples'].values
    stim_ends = stim_trials['stimulus_end'].values
    stim_end_seconds = np.unique((stim_ends - stim_starts) / fs)[0]
    window = [period[0], stim_end_seconds + period[1]]
    raster_data = []
    for trial, stpl in enumerate(zip(stim_starts, stim_recs)):
        start = stpl[0]
        srec = stpl[1]
        sptrain = get_spiketrain(srec, start, clusterID, spikes, window, fs)
        raster_data.append(sptrain)
    do_raster(raster_data, window, [0, stim_end_seconds], ax, **kwargs)


def plot_raster_stim_trial(spikes, trials, clusters, 
                          stim, trial, period, rec, fs, plot_params=None, ax=None):
    '''
    Plots a spike raster for all cells for a single trial of a single stimulus
    '''
    nclus = len(clusters)
    cluIDs = clusters['cluster'].values
    stim_trials = trials[trials['stimulus']==stim]
    this_trial = stim_trials.iloc[trial]
    stim_start = this_trial['time_samples']
    stim_end = this_trial['stimulus_end']
    stim_end_seconds = np.unique((stim_end - stim_start)/fs)[0]
    window = [period[0], stim_end_seconds+period[1]]
    raster_data = []
    for clu_num, clu in enumerate(cluIDs):
        sptrain = get_spiketrain(rec, stim_start, clu, spikes, window, fs)
        raster_data.append(sptrain)
    if plot_params == None:
        do_raster(raster_data, window, [0, stim_end_seconds], ax) 
    else:
        do_raster(raster_data, window, [0, stim_end_seconds], ax, 
                  spike_linewidth=plot_params['spike_linewidth'],
                  spike_color=plot_params['spike_color'],
                  tick_linewidth=plot_params['tick_linewidth'],
                  tick_color=plot_params['tick_color'])



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

    Return
    ------
    output : numpy array
        gaussian peristimulus time histogram
    '''
    output = np.zeros(len(times))
    for spike_time in spike_data:
        output = output + np.exp(-1.0 * np.square(times - spike_time) / (2 * sigma ** 2))
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
        Name of stimulus to calculate psth for
    period : list of floats
        Time window for the raster:
        [Seconds_pre_stimulus_onset, Seconds_post_stimulus_end]
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

    stim_trials = trials[trials['stimulus'] == stim]
    ntrials = len(stim_trials)
    stim_starts = stim_trials['time_samples'].values
    stim_ends = stim_trials['stimulus_end'].values

    stim_end_seconds = np.unique((stim_ends - stim_starts) / fs)[0]
    window = [period[0], stim_end_seconds + period[1]]
    npts = np.floor(1.0 * (window[1] - window[0]) * fs)
    times = np.linspace(window[0], window[1], npts)
    psths = np.zeros((ntrials, npts))
    for trial, start in enumerate(stim_starts):
        sptrain = get_spiketrain(rec, start, clusterID, spikes, window, fs)
        psths[trial, :] = gaussian_psth_func(times, sptrain, sigma)
    avg_psth = np.mean(psths, 0)
    std_psth = np.std(psths, 0)
    conf_ints = stats.t.interval(alpha, df=ntrials - 1, loc=avg_psth, scale=std_psth / np.sqrt(ntrials))
    return (avg_psth, std_psth, conf_ints, times)


def plot_unit_raster(spikes, trials, clusterID, raster_window, rec, fs, subplot_xy, figsize, fontsize=20, **kwargs):
    '''
    Plots a raster of all trials of all stimuli from a given unit

    Parameters
    ------
    spikes : pandas dataframe
        spike dataframe from core
    trials : pandas dataframe
        trials dataframe from events
    clusterID : int
        ID number of the cluster you wish to make the raster for
    raster_window :
        TODO
    rec : int
        Recording ID
    fs : float
        Sampling rate
    subplot_xy : (int, int)
        x and y indices of the working subplot
    figsize : integer tuple
        The figure plot size in tuple of integers, (width, height) in inches
    fontsize : int
        font size of the plot labels
    kwargs :
        keyword arguments are passed to the do_raster method

    Return
    ------
    figure :
        matplotlib Figure
    '''

    stims = trials['stimulus'].unique()

    f, pltaxes = plt.subplots(subplot_xy[0], subplot_xy[1], sharey=True, figsize=figsize)
    for ind, stim in enumerate(stims):
        ax = pltaxes.flatten()[ind]
        plot_raster_cell_stim(spikes, trials, clusterID, stim,
                              raster_window, rec, fs, ax=ax, **kwargs)
        ax.set_title('Unit: {} Stim: {}'.format(str(clusterID), stim))
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Repetition')
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                     ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(fontsize)
    return f


def plot_unit_raster_emily(spikes, trials, clusterID, raster_window, rec, fs, subplot_xy, figsize, fontsize=20,
                           **kwargs):
    '''
    Plots a raster of all trials of all stimuli from a given unit

    Parameters
    ------
    spikes : pandas dataframe
        spike dataframe from core
    trials : pandas dataframe
        trials dataframe from events
    clusterID : int
        ID number of the trial you wish to make the raster for
    raster_window : list of floats
        Time window for the raster
    rec : int
        Recording ID
    fs : float
        Sampling rate
    figsize : integer tuple
        The figure plot size in tuple of integers, (width, height) in inches
    fontsize : int
        font size of the plot labels
    kwargs :
        keyword arguments are passed to the do_raster method

    Returns
    -------
    figure : matplotlib.figure.Figure
        the final figure with the plotted rasters
    '''

    stims = np.unique(trials['stimulus'].values)
    # stims = stims[~np.isnan(stims)]

    f, pltaxes = plt.subplots(subplot_xy[0], subplot_xy[1], sharey=True, figsize=figsize)
    for ind, stim in enumerate(stims):
        if str(stim) == 'nan':
            continue
        print(stim)
        stimrecs = trials[trials['stimulus'] == stim]['recording']
        ax = pltaxes.flatten()[ind]
        plot_raster_cell_stim_emily(spikes, trials, clusterID, stim,
                                    raster_window, rec, fs, ax=ax, **kwargs)
        ax.set_title('Unit: {} Stim: {}'.format(str(clusterID), stim))
        # ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Repetition')
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                     ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(fontsize)
    return f


def plot_trial_raster_emily(spikes, trials, clusters, trialID, stim, period, rec, fs, ax=None, **kwargs):
    '''
    Plots TODO

    Parameters
    ------
    spikes : pandas dataframe
        spike dataframe from core
    trials : pandas dataframe
        trials dataframe from events
    clusters :
        TODO
    trialID : int
        ID number of the trial you wish to make the raster for
    stim : str
        Name of the stimulus to plot raster for
    period : list of floats
        Time window for the raster:
        [Seconds_pre_stimulus_onset, Seconds_post_stimulus_end]
    fs : float
        Sampling rate
    ax : Matplotlib axes handle, optional
        Axes on which to produce raster. Default gca.
    kwargs :
        keyword arguments are passed to the do_raster method
    '''

    stim_trials = trials[trials['stimulus'] == stim]
    stim_recs = stim_trials['recording'].values
    ntrials = len(stim_trials)

    stim_starts = stim_trials['time_samples'].values
    stim_ends = stim_trials['stimulus_end'].values

    stim_start = stim_starts[trialID]
    stim_end = stim_ends[trialID]
    stim_end_seconds = (stim_end - stim_start) / fs
    srec = stim_recs[trialID]

    clusterIDs = clusters['cluster'].values
    window = [period[0], stim_end_seconds + period[1]]
    raster_data = []
    for cluster in clusterIDs:
        sptrain = get_spiketrain(srec, stim_start, cluster, spikes, window, fs)
        raster_data.append(sptrain)
    do_raster(raster_data, window, [0, stim_end_seconds], ax, **kwargs)


def plot_avg_gaussian_psth_cell_stim(spikes, trials, clusterID, stim, raster_window, rec, fs, ax=None):
    return 0


def plot_unit_gaussian_psth(spikes, trials, clusterID, raster_window, rec, fs, subplot_xy, figsize):
    '''
    Plots average psth gaussian smoothed

    Parameters
    ------
    trials : pandas dataframe
        trials dataframe from events
    subplot_xy : tuple
        location of the subplot to modify
    figsize : integer tuple
        The figure plot size in tuple of integers, (width, height) in inches
    '''

    stims = trials['stimulus'].unique()
    f, pltaxes = plt.subplots(subplot_xy[0], subplot_xy[1], sharey=True, figsize=figsize)
