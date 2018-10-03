from __future__ import absolute_import
import numpy as np
from six.moves import zip


def get_spiketrain(rec, samps, clu, spikes, window, fs):
    '''
    Returns a numpy array of spike times for a single cluster 
        within a window locked to a sampling time.
    
    Parameters
    ------
    rec : int
        the recording to look in
    samps : int
        the time to lock the spiketrain to in samples
    clu : int
        the cluster identifier to get spikes from
    spikes : pandas dataframe
        the pandas dataframe containing spikes (see core)
    window : tuple or list of floats
        the window around the event in seconds to sample spikes
    fs : float
        sampling rate of the recording
    
    Returns
    ------
    spike_train : numpy array of spike times in seconds
    '''
    bds = [w * fs + samps for w in window]

    window_mask = (
            (spikes['time_samples'] > bds[0])
            & (spikes['time_samples'] <= bds[1])
    )

    perievent_spikes = spikes[window_mask]

    mask = (
            (perievent_spikes['recording'] == rec)
            & (perievent_spikes['cluster'] == clu)
    )
    return (perievent_spikes['time_samples'][mask].values.astype(np.float_) - samps) / fs


def calc_spikes_in_window(spikes, window):
    '''
    Returns a spike DataFrame containing all spikes within a given window

    Parameters
    ------
    spikes : pandas DataFrame
        Contains the spike data (see core)
    window : tuple
        The lower and upper bounds of the time window for extracting spikes in samples

    Returns
    ------
    spikes_in_window : pandas DataFrame
        DataFrame with same layout as input spikes but containing only spikes 
        within window 
    '''
    mask = ((spikes['time_samples'] < window[1]) &
            (spikes['time_samples'] >= window[0]))
    return spikes[mask]


def calc_spike_vector(spikes, time_bins):
    '''
    Returns a vector containing the number of spikes in each time bin contained in time_bins

    Parameters
    ------
    spikes : pandas dataframe 
        dataframe containing spike information
    time_bins : list 
        list of lists containing time bin definitions [t_low, t_high]

    Returns
    ------
    spike_vector : numpy array 
        Array of number of spikes in each bin 
    '''
    spike_vector = np.zeros(len(time_bins))
    for ind, bin in enumerate(time_bins):
        nspikes = len(calc_spikes_in_window(spikes, bin))
        spike_vector[ind] = nspikes
    return spike_vector


def calc_time_bins(bounds, fs, dt):
    '''
    Returns a list of time bins in samples between bounds of width dt (ms)

    Parameters
    ------
    bounds : list
        boundary of window list in samples 
    fs : float 
        sampling rate 
    dt : float 
        the length of each subwindow in ms 

    Returns
    ------
    time_bins : list 
        list of lists of time bins in samples 
    '''

    dt_samps = np.round(float(dt) / float(fs))
    T = bounds[1] - bounds[0]
    T_seconds = T / float(fs)
    nwin = np.round(T_seconds / (dt / 1000.))
    ticks = np.round(np.linspace(bounds[0], bounds[1], nwin))
    starts = ticks[0:-1]
    ends = ticks[1:]
    time_bins = []
    for start, end in zip(starts, ends):
        time_bins.append([start, end])
    return time_bins
