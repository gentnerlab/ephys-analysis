import numpy as np
import pandas as pd
import os, sys
import subprocess

import events
import core

def get_spikes_in_window(spikes, window):
    '''
    Returns a spike DataFrame containing all spikes within a given window

    Parameters
    ------
    spikes : pandas DataFrame
        Contains the spike data (see core)
    window : tuple
        The lower and upper bounds of the time window for extracting spikes 

    Returns
    ------
    spikes_in_window : pandas DataFrame
        DataFrame with same layout as input spikes but containing only spikes 
        within window 
    '''
    mask = ((spikes['time_samples']<=window[1]) & 
            (spikes['time_samples']>=window[0]))
    return spikes[mask]


def mean_fr_decorator(mean_fr_func):

    def decorated(cluster_row, *args, **kwargs):
        print(cluster_row)
        try:
            int(cluster_row)
            mean_fr = mean_fr_func(cluster_row, *args, **kwargs)
            print(pd.DataFrame({'mean_fr': mean_fr}))
            return pd.DataFrame({'mean_fr': mean_fr})
        except ValueError:
            mean_fr = mean_fr_func(cluster_row['cluster'], *args, **kwargs)
            print(pd.DataFrame({'mean_fr': mean_fr}))
            return pd.DataFrame({'mean_fr': mean_fr})

    return decorated


def calc_mean_fr(cluster_row, spikes, window):
    '''
    Computes the mean firing rate of a unit within the given spikes DataFrame

    Parameters
    ------
    row : pandas dataframe row
        row of cluster dataframe containing clusterID to compute
    spikes : pandas dataframe 
        Contains the spike data.  
        Firing rate computed from this data over the window
    window : tuple
        time (samples) over which the spikes in 'spikes' occur.

    Returns
    ------
    mean_fr : float 
        Mean firing rate over the interval
    '''
    cluster = cluster_row['cluster']
    spikes = spikes[spikes['cluster']==cluster]
    spikes = get_spikes_in_window(spikes, window)
    nspikes = len(spikes.index)
    dt = window[1] - window[0]
    mean_fr = (1.0*nspikes) / dt
    retframe = pd.DataFrame({'mean_fr': [mean_fr]})
    return retframe.iloc[0]


def calc_mean_fr_int(cluster, spikes, window):
    ''' Does the same as above, but for ints.  This is bad.  Fix this
    '''
    spikes = spikes[spikes['cluster']==cluster]
    spikes = get_spikes_in_window(spikes, window)
    nspikes = len(spikes.index)
    dt = window[1] - window[0]
    mean_fr = (1.0*nspikes) / dt
    return mean_fr  


def create_subwindows(segment, subwin_len, n_subwin_starts):
    ''' Create list of subwindows for cell group identification 

    Parameters
    ------
    segment : list
        Beginning and end of the segment to subdivide into windows
    subwin_len : int
        number of samples to include in a subwindows
    n_subwin_starts : int
        number of shifts of the subwindows

    Returns
    ------
    subwindows : list 
        list of subwindows
    '''
    print('Building subwindows...')
    starts_dt = np.floor(subwin_len / n_subwin_starts)
    starts = np.arange(segment[0], segment[0]+subwin_len, starts_dt)

    subwindows = []
    for start in starts:
        subwin_front = np.arange(start, segment[1], subwin_len)
        for front in subwin_front:
            subwin_end = front + subwin_len
            subwindows.append([front, subwin_end])

    return subwindows


def calc_population_vectors(spikes, clusters, windows, thresh):
    '''
    Builds population vectors according to Curto and Itskov 2008

    Parameters
    ------
    spikes : pandas DataFrame
        DataFrame containing spike data. 
        Must have 'fr_mean' column containing mean firing rate
    clusters : pandas DataFrame
        Dataframe containing cluster information
    windows : tuple
        The set of windows to compute population vectors for 
    thresh : float
        how many times above the mean the firing rate needs to be

    Returns 
    ------
    popvec_list : list
        population vector list.  
        Each element is a list containing the window and the population vector.
        The population vector is an array containing cluster ID and firing rate. 
    '''
    print('Building population vectors...')
    total_win = len(windows)
    popvec_list = []
    for win_num, win in enumerate(windows):
        if np.mod(win_num, 500)==0:
            print("Window {} of {}".format(str(win_num), str(total_win)))
            sys.stdout.flush()
        popvec = np.zeros([len(clusters.index), 3])
        for ind, cluster in enumerate(clusters['cluster'].values):
            fr = calc_mean_fr_int(cluster, spikes, win)
            popvec[ind, 1] = fr
            popvec[ind, 0] = cluster
            popvec[ind, 2] = fr>(1.0*thresh*clusters[
                                clusters['cluster']==cluster]['mean_fr']).values
        popvec_list.append([win, popvec])
    return popvec_list


DEFAULT_CG_PARAMS = {'cluster_group': None, 'subwin_len': 100, 'threshold': 6.0,
                     'n_subwin': 5}
def calc_cell_groups(spikes, segment, clusters, cg_params=DEFAULT_CG_PARAMS):
    '''
    Creates cell group dataframe according to Curto and Itskov 2008

    Parameters
    ------
    spikes : pandas dataframe
        Dataframe containing the spikes to analyze
    segment : tuple or list of floats
        time window for which to create cell groups 
    clusters : pandas DataFrame
        dataframe containing cluster information
    cg_params : dict, optional
        Parameters for cell group creation.  Includes: 
        cluster_group : str
            Quality of the clusters to include in the analysis (Good, MUA, etc)
        subwin_len : int 
            length (samples) for each subwin
        threshold : float, optional 
            Multiples above baseline firing rate to include activity
        n_subwin : int 
            Number of subwindows to use to generate population vectors

    Returns
    ------
    cell_groups : list
        list where each entry is a list containing a time window 
        and the ID's of the cells in that group
    '''

    print('Creating cellgroups...')
    cluster_group = cg_params['cluster_group']
    subwin_len    = cg_params['subwin_len']
    threshold     = cg_params['threshold']
    n_subwin      = cg_params['n_subwin']

    spikes = get_spikes_in_window(spikes, segment)
    if cluster_group != None:
        mask = np.ones(len(clusters.index)) < 0
        for grp in cluster_group:
            mask = np.logical_or(mask, clusters['quality'] == grp)
        clusters = clusters[mask]
        spikes = spikes[spikes['cluster'].isin(clusters['cluster'].values)]

    topology_subwindows = create_subwindows(segment, subwin_len, n_subwin)

    clusters['mean_fr'] = clusters.apply(lambda row: calc_mean_fr(row,
                                         spikes,segment)['mean_fr'], axis=1)
    # Build population vectors
    population_vector_list = calc_population_vectors(spikes, clusters, 
                                                     topology_subwindows, 
                                                     threshold)
    cell_groups = []
    for population_vector_win in population_vector_list:
        win          = population_vector_win[0]
        popvec       = population_vector_win[1]
        active_cells = popvec[(popvec[:, 2].astype(bool)), 0].astype(int)
        cell_groups.append([win, active_cells])
        
    return cell_groups

def build_perseus_input(cell_groups, savefile):
    ''' 
    Formats cell group information as input 
    for perseus persistent homology software

    Parameters
    ------
    cell_groups : list 
        cell_group information returned by calc_cell_groups
    savefile : str 
        File in which to put the formatted cellgroup information

    Yields
    ------
    savefile : text File
        file suitable for running perseus on
    '''
    print('Building Perseus input...')
    with open(savefile, 'w+') as pfile:
        #write num coords per vertex
        pfile.write('1\n')
        for win_grp in cell_groups:
            grp = list(win_grp[1])
            #debug_print('Cell group: ' + str(grp) +'\n')
            grp_dim = len(grp) - 1
            if grp_dim < 0:
                continue
            vert_str = str(grp)
            vert_str = vert_str.replace('[', '')
            vert_str = vert_str.replace(']', '')
            vert_str = vert_str.replace(' ', '')
            vert_str = vert_str.replace(',', ' ')
            out_str  = str(grp_dim) + ' ' + vert_str + ' 1\n'
            #debug_print('Writing: %s' % out_str)
            pfile.write(out_str)

    return savefile

def run_perseus(pfile):
    ''' 
    Runs perseus persistent homology software on the data in pfile

    Parameters
    ------
    pfile : str 
        file on which to compute homology

    Returns
    ------
    betti_file : str
        file containing resultant betti numbers

    '''
    print('Running Perseus...')
    of_string, ext = os.path.splitext(pfile)
    perseus_command = "/home/btheilma/bin/perseus" 
    perseus_args = "nmfsimtop {} {}".format(pfile, of_string)

    perseus_return_code = subprocess.call([perseus_command, 'nmfsimtop', pfile, 
                                           of_string])
    assert (perseus_return_code == 0), "Peseus Error!"
    betti_file = of_string+'_betti.txt'
    betti_file = os.path.join(os.path.split(pfile)[0], betti_file)
    return betti_file

def calc_bettis(spikes, segment, clusters, pfile, cg_params=DEFAULT_CG_PARAMS):
    ''' Calculate betti numbers for spike data in segment

    Parameters
    ------
    spikes : pandas DataFrame
        dataframe containing spike data
    segment : list
        time window of data to calculate betti numbers
    clusters : pandas Dataframe
        dataframe containing cluster data
    pfile : str
        file in which to save simplex data
    cg_params : dict
        Parameters for CG generation

    Returns
    ------
    bettis : list
        betti numbers.  Each member is a list with the first member being 
        filtration time and the second being betti numbers
    '''
    print('In calc_bettis')
    cell_groups = calc_cell_groups(spikes, segment, clusters, cg_params)

    build_perseus_input(cell_groups, pfile)
    betti_file = run_perseus(pfile)

    bettis = []
    with open(betti_file, 'r') as bf:
        for bf_line in bf:
            if len(bf_line)<2:
                continue
            betti_data      = bf_line.split()
            nbetti          = len(betti_data)-1
            filtration_time = int(betti_data[0])
            betti_numbers   = map(int, betti_data[1:])
            bettis.append([filtration_time, betti_numbers])
    return bettis


DEFAULT_SEGMENT_INFO = {'period': 'stim'}
def get_segment(trial_bounds, fs, segment_info):
    '''
    Use segment info to determine segment to compute topology for 

    Parameters
    ------
    trial_bounds : list
        List containing trial start and trial end in samples
    fs : int 
        The sampling rate 
    segment_info : dict
        Dictionary containing:
        'period' (stim or ?)
        'segstart' : time in ms of segment start relative to trial start 
        'segend' : time in ms of segment end relative to trial start

    Returns
    ------
    segment : list 
        bounds for the segment to compute topology for, in samples 

    '''

    if segment_info['period'] == 'stim':
        return trial_bounds
    else:
        seg_start = trial_bounds[0] + np.floor(segment_info['segstart']*(fs/1000.))
        seg_end = trial_bounds[0] + np.floor(segment_info['segend']*(fs/1000.))
    return [seg_start, seg_end]


def calc_bettis_on_dataset(block_path, cluster_group=None, windt_ms=50., 
                           segment_info=DEFAULT_SEGMENT_INFO):
    '''
    Calculate bettis for each trial in a dataset and report statistics

    Parameters
    ------
    block_path : str 
        Path to directory containing data files 
    cluster_group : list
        list of cluster qualities to include in analysis 
    windt_ms : float, optional
        window width in milliseconds
    segment_info : dict 
        dictionary containing information on which segment to compute topology
        'period' (stim or ?)
        'segstart' : time in ms of segment start relative to trial start 
        'segend' : time in ms of segment end relative to trial start
    
    Yields
    ------
    betti_savefile : file
        File containing betti numbers for each trial for a given stimulus
        For all stimuli 
    '''
    maxbetti      = 10
    kwikfile      = core.find_kwik(block_path)
    kwikname, ext = os.path.splitext(os.path.basename(kwikfile))

    spikes   = core.load_spikes(block_path)
    clusters = core.load_clusters(block_path)
    trials   = events.get_trials(block_path)
    fs       = core.get_fs(block_path)

    windt_samps = np.floor(windt_ms*(fs/1000.))

    stims = set(trials['stimulus'].values)
    for stim in stims:
        print('Calculating bettis for stim: {}'.format(stim))
        stim_trials = trials[trials['stimulus']==stim]
        nreps       = len(stim_trials.index)
        stim_bettis = np.zeros([nreps, maxbetti])

        betti_savefile = kwikname + '_stim{}'.format(stim) + '_betti.csv'
        betti_savefile = os.path.join(block_path, betti_savefile)
        for rep in range(nreps):
            pfile = kwikname + '_stim{}'.format(stim) + \
                    '_rep{}'.format(int(rep)) + '_simplex.txt'
            pfile = os.path.join(block_path, pfile)

            trial_start = stim_trials.iloc[rep]['time_samples']
            trial_end   = stim_trials.iloc[rep]['stimulus_end']

            cg_params                   = DEFAULT_CG_PARAMS
            cg_params['subwin_len']     = windt_samps
            cg_params['cluster_group']  = cluster_group

            segment = get_segment([trial_start, trial_end], fs, segment_info)
            print('Trial bounds: {}  {}'.format(str(trial_start), 
                                                str(trial_end)))
            print('Segment bounds: {}  {}'.format(str(segment[0]), 
                                                  str(segment[1])))
            
            bettis = calc_bettis(spikes, segment, 
                                 clusters, pfile, cg_params)
            assert (len(bettis) == 1), "Too many filtrations"
            trial_bettis                         = bettis[0][1]
            stim_bettis[rep, :len(trial_bettis)] = trial_bettis

        stim_bettis_frame = pd.DataFrame(stim_bettis)
        stim_bettis_frame.to_csv(betti_savefile, index_label='rep')