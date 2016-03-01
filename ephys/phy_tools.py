import os, sys
import glob
import imp
import h5py as h5
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

EXP_DIR = os.path.dirname(os.path.realpath(__file__))
KLUSTA_DIR = 'klusta'
TABLE_DIR = 'Tables'
FIGURE_DIR = 'Figures'


def get_kwik(block):
    return os.path.join(EXP_DIR,KLUSTA_DIR,block,block.split('__')[-1]+'.kwik')

def get_kwd(block):
    return os.path.join(EXP_DIR,KLUSTA_DIR,block,block.split('__')[-1]+'.raw.kwd')

def csv(block,name):
    return os.path.join(EXP_DIR,TABLE_DIR,'{}_{}.csv'.format(block,name))

## top-level functions to read data from kwik files

def read_events(block,event_type):
    '''
    Reads events from the kwik file associated with a block
    
    Parameters
    ------
    block : str
        identifier of block
    event_type : str
        the type of event
    
    Returns
    ------
    events : pandas DataFrame
    
    '''
    with h5.File(get_kwik(block),'r') as kf:
        events = {}
        for col in kf['/event_types'][event_type]:
            events[col] = kf['/event_types'][event_type][col][:]
    return pd.DataFrame(events)

def get_fs(block):
    with h5.File(get_kwik(block),'r') as kf:
        fs = kf['/recordings/0'].attrs['sample_rate']
    return fs


QUAL_LOOKUP = {0: 'Noise',
               1: 'MUA',
               2: 'Good',
               3: 'unsorted',
              }


def get_qual(block,cluster):
    with h5.File(get_kwik(block),'r') as kf:
        qual = kf['/channel_groups/0/clusters/main/']["%i" % cluster].attrs['cluster_group']
    return QUAL_LOOKUP[qual]

def get_clusters(block):    
    with h5.File(get_kwik(block),'r') as kf:
        observed_clusters = np.unique(kf['/channel_groups/0/spikes/clusters/main'][:])
        clusters = pd.DataFrame({'cluster': observed_clusters})
        clusters['quality'] = clusters['cluster'].apply(lambda clu: get_qual(block,clu))
    return clusters

def get_spikes(block):
    with h5.File(get_kwik(block),'r') as kf:
        spikes = pd.DataFrame(dict(cluster=kf['/channel_groups/0/spikes/clusters/main'][:],
                                   recording=kf['/channel_groups/0/spikes/recording'][:],
                                   time_samples=kf['/channel_groups/0/spikes/time_samples'][:]))
    return spikes

# functions to process stim & trial info from digmarks

def find_stim_end(digmark_row):
    return digmark_row['codes'] in '>#'
    
class FindEnd():
    def __init__(self):
        self.keep = True
    def check(self,code):
        if code in '()FfTt]N>#':
            self.keep = False
        return self.keep
    
def get_stim_start(stim_end_row,digmarks):
    rec,ts = stim_end_row['recording'],stim_end_row['time_samples']
    mask = (
        (digmarks['recording']==rec)
        & (digmarks['time_samples'] < ts)
        & ~digmarks['codes'].str.contains('[RCL]')
        )
    this_trial_mask = (
        digmarks[mask].iloc[::-1]['codes'].apply(FindEnd().check).iloc[::-1]
        & digmarks[mask]['codes'].str.contains('<')
    )
    this_trial = digmarks[mask][this_trial_mask]
    return this_trial.iloc[0]

def is_not_floatable(string):
    ''' returns True if string can be converted to float else False
    '''
    try: 
        float(string)
        return False
    except ValueError:
        return True


def get_stim_info(trial_row,stimulus,fs):
    rec,samps = trial_row['recording'], trial_row['time_samples']
    stim_mask = (
        (stimulus['recording']==rec)
        & (stimulus['time_samples']>(samps-0.1*fs))
        & (stimulus['time_samples']<(samps+fs))
        & ~stimulus['text'].str.contains('date')
        & stimulus['text'].apply(is_not_floatable)
        )
    
    if stim_mask.sum()>0:
        return stimulus[stim_mask].iloc[0]
    else:
        return dict(codes=np.nan,time_samples=np.nan,recording=np.nan,text=np.nan)


def get_stim_end(trial_row,digmarks,fs,window=60.0):
    rec,samps = trial_row['recording'], trial_row['time_samples']
    resp_mask = (
        (digmarks['recording']==rec)
        & (digmarks['time_samples']>samps)
        & (digmarks['time_samples']<(samps+fs*window))
        & digmarks['codes'].str.contains('[>#]')
        )
    assert digmarks[resp_mask].shape[0]>0
    return digmarks[resp_mask].iloc[0]

def get_response(trial_row,digmarks,fs,window=5.0):
    rec,samps = trial_row['recording'], trial_row['time_samples']
    try:
        stim_dur = trial_row['stimulus_end'] - trial_row['time_samples']
    except KeyError:
        stim_dur = get_stim_end(rec,samps,fs)['time_samples']-samps
    resp_mask = (
        (digmarks['recording']==rec)
        & (digmarks['time_samples']>(samps+stim_dur))
        & (digmarks['time_samples']<(samps+stim_dur+fs*window))
        & digmarks['codes'].str.contains('[RLN]')
        )
    if digmarks[resp_mask].shape[0]>0:
        return digmarks[resp_mask].iloc[0]
    else:
        return dict(codes=np.nan,time_samples=np.nan,recording=np.nan)

def get_consequence(trial_row,digmarks,fs,window=2.0):
    rec,samps = trial_row['recording'], trial_row['time_samples']
    rt = trial_row['response_time']
    bds = rt, rt+fs*window
    resp_mask = (
        (digmarks['recording']==rec)
        & (digmarks['time_samples']>bds[0])
        & (digmarks['time_samples']<bds[1])
        & digmarks['codes'].str.contains('[FfTt]')
        )
    if digmarks[resp_mask].shape[0]>0:
        return digmarks[resp_mask].iloc[0]
    else:
        return dict(codes=np.nan,time_samples=np.nan,recording=np.nan)

def is_correct(consequence):
    return consequence if consequence is np.nan else consequence in 'Ff'

def get_trials(block):
    digmarks = read_events(block,'DigMark')
    stimulus = read_events(block,'Stimulus')
    fs = get_fs(block)
    
    stim_end_mask = digmarks.apply(find_stim_end,axis=1)
    trials = digmarks[stim_end_mask].apply(lambda row: get_stim_start(row,digmarks),axis=1)[:]
    trials.reset_index(inplace=True)
    del trials['index']
    del trials['codes']
    trials['stimulus'] = trials.apply(lambda row: get_stim_info(row,stimulus,fs)['text'],axis=1)
    trials['stimulus_end'] = trials.apply(lambda row: get_stim_end(row,digmarks,fs)['time_samples'],axis=1)
    trials['response'] = trials.apply(lambda row: get_response(row,digmarks,fs)['codes'],axis=1)
    trials['response_time'] = trials.apply(lambda row: get_response(row,digmarks,fs)['time_samples'],axis=1)
    trials['consequence'] = trials.apply(lambda row: get_consequence(row,digmarks,fs)['codes'],axis=1)
    trials['correct'] = trials['consequence'].apply(is_correct)
    return trials

## functions which "join" multiple tables

def get_spiketrain(rec,samps,clu,spikes,window,fs):
    bds = [w*fs+samps for w in window]
    motif_mask = (
        (spikes['recording']==rec)
        & (spikes['time_samples']>bds[0])
        & (spikes['time_samples']<=bds[1])
        )
    clu_mask = spikes['cluster']==clu
    return (spikes['time_samples'][motif_mask & clu_mask].values.astype(np.float_) - samps) / fs

## cluster metrics

def find_mean_waveforms(block,cluster):
    return os.path.join(EXP_DIR,KLUSTA_DIR,block,block.split('__')[-1]+'.phy',
                        'cluster_store','0','main','{}.mean_waveforms'.format(cluster))
    
def find_mean_masks(block,cluster):
    return os.path.join(EXP_DIR,KLUSTA_DIR,block,block.split('__')[-1]+'.phy',
                        'cluster_store','0','main','{}.mean_masks'.format(cluster))

def find_prb(block):
    '''
    Returns the path of the block's probe file.
    '''
    ls = glob.glob(
        os.path.join(EXP_DIR,KLUSTA_DIR,block,'*.prb')
        )
    assert len(ls)==1
    return ls[0]

def get_cluster_loc(mean_masks,prb,weighted=False):
    prb_info = imp.load_source('prb',prb)
    
    channels = prb_info.channel_groups[0]['channels']
    geometry = prb_info.channel_groups[0]['geometry']

    coords = np.array([geometry[ch] for ch in channels])
    
    mean_masks_arr = np.fromfile(mean_masks,dtype=np.float32)

    if weighted:
        return np.dot(mean_masks_arr,coords) / mean_masks_arr.sum()
    else:
        return coords[mean_masks_arr.argmax(),:]

def plot_mean_waveform(mean_waveform,prb,chan_alpha=None,scale_factor=0.05,color='0.5',**plot_kwargs):
    prb_info = imp.load_source('prb',prb)
    
    channels = prb_info.channel_groups[0]['channels']
    geometry = prb_info.channel_groups[0]['geometry']

    coords = np.array([geometry[ch] for ch in channels])
    
    arr = np.fromfile(mean_waveform,dtype=np.float32).reshape((-1,len(channels)))
    
    if chan_alpha is None:
        chan_alpha = np.ones(len(coords))
    
    for waveform,xy,alpha in zip(arr.T,coords,chan_alpha):
#         print mask
        plt.plot(xy[0]+np.arange(len(waveform))-len(waveform)/2,
                 waveform*scale_factor+xy[1],
                 color=color,
                 alpha=alpha,
                 **plot_kwargs
                )
        try:
            label = plot_kwargs.pop('label')
        except KeyError:
            pass

def get_cluster_coords(block,clu,weighted=False):
    mean_masks = find_mean_masks(block,clu)
    prb = find_prb(block)
    return get_cluster_loc(mean_masks,prb,weighted=weighted)


def plot_cluster(block,clu,color='0.5',**plot_kwargs):
    mean_waveforms = find_mean_waveforms(block,clu)
    mean_masks_arr = np.fromfile(find_mean_masks(block,clu),dtype=np.float32)
    prb = find_prb(block)

    plot_mean_waveform(mean_waveforms,prb,chan_alpha=mean_masks_arr,color=color,**plot_kwargs)
    
import seaborn as sns

def plot_all_clusters(block,clusters=None):
    
    if clusters is None:
        clusters = get_clusters(block)
        clusters = clusters[clusters.quality.isin(['Good','MUA'])].sort_values(['quality','cluster']).reset_index()

    palette = sns.color_palette("hls", len(clusters))

    f = plt.figure(figsize=(6, 12))
    for idx,cluster_row in clusters.iterrows():

        clu = cluster_row.cluster
        qual = cluster_row.quality

        plot_cluster(block,clu,color=palette[idx],scale_factor=0.05,label="{}({})".format(clu,qual))

    plt.axis('equal')
    leg = plt.legend(loc='center left')
    [plt.setp(label, alpha = 1.0) for label in leg.get_lines()]
    sns.despine(bottom=True,left=True)
    plt.xticks([])
    plt.yticks([])
    
## spike shapes
from scipy.interpolate import UnivariateSpline

def upsample_spike(spike_shape,fs,factor=10):
    t = np.arange(0,spike_shape.shape[0]/fs,1/fs)[:spike_shape.shape[0]]
    spl = UnivariateSpline(t,spike_shape)
    ts = np.arange(0,len(spike_shape)/fs,1/(fs*factor))
    return ts, spl(ts)
    
def get_troughpeak(time,spike_shape):
    trough_i = spike_shape.argmin()
    peak_i = spike_shape[trough_i:].argmax()+trough_i
    return time[trough_i],time[peak_i]

def get_raw_shape(block,clu):
    prb = find_prb(block)
    prb_info = imp.load_source('prb',prb)
    
    mean_waveform = find_mean_waveforms(block,clu)
    arr = np.fromfile(mean_waveform,dtype=np.float32).reshape((-1,len(prb_info.channel_groups[0]['channels'])))
    
    mean_masks = find_mean_masks(block,clu)
    mean_masks_arr = np.fromfile(mean_masks,dtype=np.float32)
    
    return arr[:,mean_masks_arr.argmax()]

def get_width(block,clu):
    fs = get_fs(block)
    raw_shape = get_raw_shape(block,clu)
    
    trough,peak = get_troughpeak(*upsample_spike(raw_shape,fs))
    return peak-trough

def plot_spike_shape(block,clu,normalize=False,**kwargs):
    fs = get_fs(block)
    raw_shape = get_raw_shape(block,clu)
    
    time,shape = upsample_spike(raw_shape,fs)
    time -= time[shape.argmin()]
    
    if normalize==True:
        shape /= -shape[shape.argmin()]
    
    plt.plot(time,shape,**kwargs)
    