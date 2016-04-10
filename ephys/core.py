import sys, os
import glob
import imp
import numpy as np
import h5py as h5
import pandas as pd
from functools import wraps

try: import simplejson as json
except ImportError: import json

def file_finder(find_file_func):
    '''
    Decorator to help find files.

    This wraps a function that yields a single value

    Example use:

    >>>@file_finder
    >>>def find_my_text_file():
    >>>    return '/home/ephys/*.txt'
    >>>find_my_text_file()
    /home/ephys/the_only_text_file_here.txt

    '''
    @wraps(find_file_func)
    def decorated(*args,**kwargs):
        ls = glob.glob(find_file_func(*args,**kwargs))
        assert len(ls)==1, ls
        return ls[0]
    return decorated

@file_finder
def find_kwik(block_path):
    '''
    Returns the kwik file found in the block path
    
    Parameters
    ------
    block_path : str
        path to the block
    
    Returns
    ------
    kwik : full path name to kwik file
    '''
    return os.path.join(block_path,'*.kwik')

@file_finder
def find_kwd(block_path):
    '''
    Returns the raw.kwd file found in the block path
    
    Parameters
    ------
    block_path : str
        path to the block
    
    Returns
    ------
    kwd : full path name to raw.kwd file
    '''
    return os.path.join(block_path,'*.raw.kwd')


@file_finder
def find_kwx(block_path):
    '''
    Returns the kwx file found in the block path
    
    Parameters
    ------
    block_path : str
        path to the block
    
    Returns
    ------
    kwx : full path name to kwx file
    '''
    return os.path.join(block_path,'*.kwx')

@file_finder
def find_prb(block_path):
    '''
    Returns the *.prb file found in the block path
    
    Parameters
    ------
    block_path : str
        path to the block
    
    Returns
    ------
    prb : full path name to *.prb file
    '''
    return os.path.join(block_path,'*.prb')

@file_finder
def find_info(block_path):
    '''
    Returns the raw.kwd file found in the block path
    
    Parameters
    ------
    block_path : str
        path to the block
    
    Returns
    ------
    kwd : full path name to _info.json file
    '''
    return os.path.join(block_path,'*_info.json')

def load_probe(block_path):
    '''
    Returns the probe info for the block
    
    Parameters
    ------
    block_path : str
        path to the block
    
    Returns
    ------
    probe_info : dictionary of probe channels, geometry, and adjacencies
    '''
    return imp.load_source('prb',find_prb(block_path))

def load_events(block_path,event_type):
    '''
    Reads events from the kwik file associated with a block

    example use:

    >>> digmarks = load_events(block_path,'DigMark')
    >>> stimulus = load_events(block_path,'Stimulus')
    
    Parameters
    ------
    block_path : str
        path to the block
    event_type : str
        the type of event
    
    Returns
    ------
    events : pandas DataFrame
        Dataframe format
        Depends on event type
        codes : the code of the event 
        time_samples : time in samples of the event
        recording : the recording ID of the event 
        text : text associated with the event (Stimulus only)
    
    '''
    with h5.File(find_kwik(block_path),'r') as kf:
        events = {}
        for col in kf['/event_types'][event_type]:
            events[col] = kf['/event_types'][event_type][col][:]
    return pd.DataFrame(events)

def load_fs(block_path):
    '''
    Reads sampling rate in Hz from the kwik file associated with a block
    
    Parameters
    ------
    block_path : str
        path to the block
    
    Returns
    ------
    fs : sampling rate in Hz
    
    '''
    with h5.File(find_kwik(block_path),'r') as kf:
        fs = kf['/recordings/0'].attrs['sample_rate']
    return fs


QUAL_LOOKUP = {0: 'Noise',
               1: 'MUA',
               2: 'Good',
               3: 'unsorted',
              }


def load_qual(block_path,cluster):
    '''
    Returns labeled cluster quality ('Noise', 'MUA', 'Good', 'unsorted')
        for a given cluster
    
    Parameters
    ------
    block_path : str
        path to the block
    cluster : int
        the cluster id
    
    Returns
    ------
    quality : string
        one of the following: ('Noise', 'MUA', 'Good', 'unsorted')
    
    '''
    with h5.File(find_kwik(block_path),'r') as kf:
        qual = kf['/channel_groups/0/clusters/main/']["%i" % cluster].attrs['cluster_group']
    return QUAL_LOOKUP[qual]

def load_clusters(block_path,channel_group=0,clustering='main'):
    '''
    Returns a dataframe of clusters observed in kwik file

    Parameters
    ------
    block_path : str
        path to the block
    channel_group : int, optional
        shank ID
    clustering : str, optional
        ID of clustering
    
    Returns
    ------
    clusters : pandas DataFrame
        Dataframe format
        cluster : cluster ID
        quality : cluster quality from manual sort ('Noise', 'MUA', 'Good', 'unsorted')

    '''    
    with h5.File(find_kwik(block_path),'r') as kf:
        observed_clusters = np.unique(
            kf['/channel_groups/{}/spikes/clusters/{}'.format(channel_group,clustering)][:]
            )
        clusters = pd.DataFrame({'cluster': observed_clusters})
        clusters['quality'] = clusters['cluster'].map(lambda clu: load_qual(block_path,clu))
    return clusters

def load_spikes(block_path,channel_group=0,clustering='main'):
    '''
    Returns a pandas dataframe of spikes observed in kwik file
    
    Parameters
    ------
    block_path : str
        path to the block
    channel_group : int, optional
        shank ID
    clustering : int, optional
        ID of clustering
        
    Returns
    ------
    spikes : pandas DataFrame
        Dataframe format
        cluster : cluster ID of the spike 
        recording : recording ID of the spike
        time_samples : time stamp (samples) of the spike

    '''    
    with h5.File(find_kwik(block_path),'r') as kf:
        spikes = pd.DataFrame(
            dict(cluster=kf['/channel_groups/{}/spikes/clusters/{}'.format(channel_group,clustering)][:],
                 recording=kf['/channel_groups/{}/spikes/recording'.format(channel_group)][:],
                 time_samples=kf['/channel_groups/{}/spikes/time_samples'.format(channel_group)][:],
                 )
            )
    return spikes

def load_info(block_path):
    with open(find_info(block_path)) as f:
        info = json.load(f)
    return info