from __future__ import absolute_import
from __future__ import print_function
import os
import glob
import numpy as np
from scipy.interpolate import UnivariateSpline
from .core import file_finder, load_probe, load_fs, load_clusters, load_spikes
from .core import find_info, find_kwd, find_kwik, find_kwx
import h5py as h5
import json
from six.moves import range


@file_finder
def find_mean_waveforms(block_path, cluster, cluster_store=0, clustering='main'):
    '''
    Returns the mean waveform file for a given cluster found in the block path

    Parameters
    ------
    block_path : str
        path to the block
    cluster : int
        the cluster identifier
    cluster_store : int
        the cluster store identifier
    clustering : str, optional
        ID of clustering

    Returns
    ------
    mean_waveforms_file : full path name to mean_waveforms file
    '''
    return os.path.join(block_path,
                        '*.phy',
                        'cluster_store',
                        str(cluster_store),
                        clustering,
                        '{}.mean_waveforms'.format(cluster)
                        )


@file_finder
def find_mean_masks(block_path, cluster, cluster_store=0, clustering='main'):
    '''
    Returns the mean masks file for a given cluster found in the block path

    Parameters
    ------
    block_path : str
        path to the block
    cluster : int
        the cluster identifier
    cluster_store : int
        the cluster store identifier
    clustering : str, optional
        ID of clustering

    Returns
    ------
    mean_masks_file : full path name to mean_waveforms file
    '''
    return os.path.join(block_path,
                        '*.phy',
                        'cluster_store',
                        str(cluster_store),
                        clustering,
                        '{}.mean_masks'.format(cluster)
                        )


def mean_masks_w(block_path, cluster):
    '''
    Weights are equivalent to the mean_mask values for the channel.

    Parameters
    ------
    block_path : str
        the path to the block
    cluster : int
        the cluster identifier

    Returns
    ------
    w : weight vector
    '''
    mean_masks = find_mean_masks(block_path, cluster)
    mean_masks_arr = np.fromfile(mean_masks, dtype=np.float32)
    return mean_masks_arr


def max_masks_w(block_path, cluster):
    '''
    Places all weight on the channel(s) which have the largest mean mask values.

    If more than one channel have a mean_mask value equal to the max, these
    channels will be weighted equally.

    Parameters
    ------
    block_path : str
        the path to the block
    cluster : int
        the cluster identifier

    Returns
    ------
    w : weight vector
    '''
    w = mean_masks_w(block_path, cluster)
    return w == w.max()


def get_cluster_coords(block_path, cluster, weight_func=None):
    '''
    Returns the location of a given cluster on the probe in x,y coordinates
        in whatever units and reference the probe file uses.

    Parameters
    ------
    block_path : str
        the path to the block
    cluster : int
        the cluster identifier
    weight_func : function
        function which takes `block_path` and `cluster` as args and returns a weight
        vector for the coordinates. default: max_masks_w

    Returns
    ------
    xy : numpy array of coordinates
    '''
    if weight_func is None:
        weight_func = max_masks_w
    w = weight_func(block_path, cluster)

    prb_info = load_probe(block_path)
    channels = prb_info.channel_groups[0]['channels']
    geometry = prb_info.channel_groups[0]['geometry']
    coords = np.array([geometry[ch] for ch in channels])

    return np.dot(w, coords) / w.sum()


# spike shapes

def upsample_spike(spike_shape, fs, new_fs=1000000.0):
    '''
    upsamples a spike shape to prepare it for computing the spike width

    Parameters
    ------
    spike_shape : numpy array
        the spike shape
    fs : float
        the sampling rate of the spike shape
    new_fs : float
        sampling rate to upsample to (default=200Hz)

    Returns
    ------
    time : numpy array
        array of sample times in seconds
    new_spike_shape :
        upsampled spike shape
    '''
    t_max = spike_shape.shape[0] / fs
    t = np.arange(0, t_max, 1 / fs)[:spike_shape.shape[0]]
    spl = UnivariateSpline(t, spike_shape)
    ts = np.arange(0, t_max, 1 / new_fs)
    return ts, spl(ts)


def get_troughpeak(time, spike_shape):
    '''
    Grabs the time of the trough and peak

    Parameters
    ------
    time : numpy array
        time series of spike data
    spike_shape : numpy array
        the spike shape

    Returns
    ------
    trough_time : float
        time of trough in seconds
    peak_time : float
        time of peak in seconds
    '''
    trough_i = spike_shape.argmin()
    peak_i = spike_shape[trough_i:].argmax() + trough_i
    return time[trough_i], time[peak_i]


def get_width_half_height(time,spike_shape):
    '''
    grabs the time between the zero crossings around trough after normalize to min height and offset by 0.5
    
    Parameters
    ------
    time : numpy array
    spike_shape : numpy array %should be up-sampled to at least 100000 before calculating
    
    Returns
    ------
    width_half_height : float
        time between crossings in seconds

    '''
    width = np.nan;
    
    trough,peak = get_troughpeak(time,spike_shape)
    ind = np.where(time == trough)
    troughind = ind[0][0]
    
    spike_shape /= -spike_shape.min()
    spike_shape = spike_shape + 0.5
    zero_crossings = np.where(np.diff(np.sign(spike_shape)))[0]
    
    i = zero_crossings < troughind
    j = zero_crossings > troughind

    if (True in i) & (True in j):
        # zero crossings before trough
        i = (np.where(i))
        pre_ind = zero_crossings[max(i[0])]
        
        # zero crossings after trough
        j = (np.where(j))
        post_ind = zero_crossings[min(j[0])]
        
        width = time[post_ind] - time[pre_ind]

    return width


def get_width(block_path, cluster, new_fs=1000000.0):
    '''
    grabs the time of the trough and peak

    Parameters
    ------
    block_path : str
        the path to the block
    cluster : int
        the cluster identifier
    new_fs : float
        sampling rate to upsample to (default=200Hz)

    Returns
    ------
    width : float
        the width of the spike in seconds
    '''
    fs = load_fs(block_path)
    exemplar = get_spike_exemplar(block_path, cluster)

    trough, peak = get_troughpeak(*upsample_spike(exemplar, fs, new_fs=new_fs))

    return peak - trough


def get_mean_waveform_array(block_path, cluster):
    '''
    returns the mean spike shape on all channels

    Parameters
    ------
    block_path : str
        the path to the block
    cluster : int
        the cluster identifier

    Returns
    ------
    mean_waveform_array : numpy array
        mean waveform on principal channel. shape: (time_samples,channels)
    '''
    prb_info = load_probe(block_path)
    mean_waveform = find_mean_waveforms(block_path, cluster)
    shape = (-1, len(prb_info.channel_groups[0]['channels']))
    return np.fromfile(mean_waveform, dtype=np.float32).reshape(shape)


def get_spike_exemplar(block_path, cluster):
    '''
    Returns an exemplar of the spike shape on the principal channel

    Parameters
    ------
    block_path : str
        the path to the block
    cluster : int
        the cluster identifier

    Returns
    ------
    exemplar : numpy array
        mean waveform on principal channel
    '''

    mean_waveform = find_mean_waveforms(block_path, cluster)
    arr = get_mean_waveform_array(block_path, cluster)

    mean_masks = find_mean_masks(block_path, cluster)
    mean_masks_arr = np.fromfile(mean_masks, dtype=np.float32)

    return arr[:, mean_masks_arr.argmax()]


def get_wide_narrow(block_path, cluster_list, thresh):
    '''
    Return lists of clusters

    Parameters
    ------
    block_path : str
        the path to the block
    cluster_list : array
        a list of cluster identifiers
    thresh : float
        minimum duration of spike to be considered wide

    Returns
    ------
    wide : list of clusters with width greater than the threshold
    narrow : list of clusters with width less than the threshold
    '''
    wide = []
    narrow = []
    for cluster in cluster_list:
        sw = get_width(block_path, cluster)
        if sw >= thresh:
            wide.append(cluster)
        else:
            narrow.append(cluster)
    return (wide, narrow)


def make_phy_folder(block_path):
    '''
    Create a directory for phy clusters

    Parameters
    ------
    block_path : str
        the path to the block

    Returns
    ------
    phy folder: string
        path to phy directory
    '''
    kwikf = find_kwik(block_path)
    kwikfname = os.path.split(kwikf)[1]
    kwikname = os.path.splitext(kwikfname)[0]
    phy_fold = os.path.join(block_path, kwikname + '.phy')
    phy_fold = os.path.abspath(os.path.join(phy_fold, 'cluster_store/0/main/'))
    if not os.path.exists(phy_fold):
        os.makedirs(phy_fold)
    return phy_fold


def spikeindices(block_path, cluster, channel_group=0, clustering='main'):
    '''
    Return a list of indices of spikes for a cluster

    Parameters
    ------
    block_path : str
        the path to the block
    cluster : int
        the cluster identifier
    channel_group :
        the channel group identifier
    clustering : str, optional
        ID of clustering

    Returns
    ------
    indices : numpy array
        indices of spikes in a specified cluster
    '''
    with h5.File(find_kwik(block_path), 'r') as kwikf:
        sptimes = kwikf[
            '/channel_groups/{}/spikes/clusters/{}'.format(channel_group, clustering)][:]
    return (sptimes == cluster)


def compute_cluster_waveforms(block_path):
    '''
    legacy method for computing cluster waveforms

    Parameters
    ------
    block_path : str
        the path to the block
    '''
    with open(find_info(block_path), 'rb') as infofile:
        info = json.load(infofile)
        prespike = info['params']['prespike']
        postspike = info['params']['postspike']
        nchans = info['params']['nchan']
    spikes = load_spikes(block_path)
    clusters = spikes['cluster'].unique()
    phy_fold = make_phy_folder(block_path)

    for cluster in clusters:
        print("Cluster: {}".format(cluster))
        cluspikes = spikes[spikes['cluster'] == cluster]
        cluspiketimes = cluspikes['time_samples'].values
        mean_waveform = np.zeros((prespike + postspike, nchans))

        waveforms = np.zeros((len(cluspiketimes), prespike + postspike, nchans))
        with h5.File(find_kwd(block_path), 'r') as kwdf:
            for ind, sptime in enumerate(cluspiketimes):
                test = np.zeros((prespike + postspike, nchans))
                start_ind = max((int(sptime - prespike)), 0)
                start_ind2 = abs(min(int(sptime - prespike), 0))
                test[start_ind2:] = kwdf['/recordings/0/data'][start_ind:int(sptime + postspike), :]
                waveforms[ind, :, :] = test
                mean_waveform += test

        waveforms = waveforms.flatten()
        mean_waveform /= len(cluspiketimes)
        mean_waveform = mean_waveform.flatten()
        with h5.File(find_kwx(block_path), 'r') as kwxf:

            cluster_spike_inds = spikeindices(block_path, cluster)
            nspike = np.count_nonzero(cluster_spike_inds)
            masks = kwxf['/channel_groups/0/features_masks'][cluster_spike_inds, :, 1]
            masks = np.reshape(masks, (nspike, nchans, -1))
            masks = np.mean(masks, axis=2)
            mean_masks = np.mean(masks, axis=0)
            features = kwxf['/channel_groups/0/features_masks'][cluster_spike_inds, :, 0]
            mean_features = np.mean(features, axis=0)
            features = features.flatten()
            masks = masks.flatten()

            # make phy folder

        waveforms.tofile(os.path.join(phy_fold, '{}.waveforms'.format(cluster)))
        mean_waveform.tofile(os.path.join(phy_fold, '{}.mean_waveforms'.format(cluster)))
        masks.tofile(os.path.join(phy_fold, '{}.masks'.format(cluster)))
        mean_masks.tofile(os.path.join(phy_fold, '{}.mean_masks'.format(cluster)))
        mean_features.tofile(os.path.join(phy_fold, '{}.mean_features'.format(cluster)))
        features.tofile(os.path.join(phy_fold, '{}.features'.format(cluster)))


def compute_cluster_waveforms_fast(block_path, spikes, before=10, after=30, n_chans=-1):
    '''
    compute spike waveforms for unique clusters

    Parameters
    ------
    block_path : str
        the path to the block
    spikes : pandas dataframe
        spike dataframe from core
    before : int
        number of samples before the wave
    after : int
        number of samples after the wave
    n_chans : int
        number of recording channels observed

    Returns
    ------
    waveforms : numpy array
        array of waveforms for plotting. 3-dimensional of cluster id,
        wavelength width, and recorded spike data
    cluster_map : numpy array
        array of indexed clusters of unique spikes
    '''
    wave_length = before + 1 + after

    kwd = find_kwd(block_path)
    with h5.File(kwd, 'r') as kwd_f:
        if n_chans == -1:
            recordings = np.sort(
                np.array(list(kwd_f['recordings'].keys()), dtype=int)).astype('unicode')
            for recording in recordings:
                assert n_chans == -1 or n_chans == kwd_f['recordings'][recording]['data'].shape[1]
                n_chans = kwd_f['recordings'][recording]['data'].shape[1]

        num_clusters = len(spikes['cluster'].unique())
        counts = np.zeros(num_clusters)
        waveforms = np.zeros((num_clusters, wave_length, n_chans))
        cluster_map = spikes['cluster'].unique()
        cluster_map.sort()
        cluster_map = {cluster: idx for idx, cluster in enumerate(cluster_map)}

        for recording, recording_group in spikes.groupby('recording'):
            recording_data = kwd_f['recordings'][str(recording)]['data'][:, :]
            for cluster, cluster_spikes in recording_group.groupby('cluster'):
                starts = cluster_spikes['time_samples'].values - before
                starts = starts[starts > 0]
                starts = starts[starts + wave_length < recording_data.shape[0]]
                counts[cluster_map[cluster]] += len(starts)

                for i in range(wave_length):
                    waveforms[cluster_map[cluster], i,
                              :] += np.sum(recording_data[starts + i, :], axis=0)

    waveforms /= counts.reshape((num_clusters, 1, 1))
    return waveforms, cluster_map
