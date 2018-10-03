from __future__ import absolute_import
from __future__ import print_function
import os
import subprocess
import tempfile
import shutil
import numpy as np
import h5py as h5
import pandas as pd
from .core import load_clusters, load_spikes, find_kwx
from six.moves import range


def make_isotools_features(block_path, features_file, do_noise=False):
    '''
    TODO comment

    Parameters
    ------
    block_path : str
        path to the block
    features_file : str
        path to
    do_noise : boolean


    Returns
    ------
    cluster values :

    '''

    clusters = load_clusters(block_path)

    if do_noise:
        neurons = clusters.sort_values(['quality', 'cluster']).reset_index()
    else:
        neurons = clusters[clusters.quality != 'Noise'].sort_values(['quality', 'cluster']).reset_index()

    spikes = load_spikes(block_path)
    spike_index = np.where(spikes.cluster.isin(neurons.cluster).values == True)[0]

    clu_vals = np.unique(spikes.loc[spike_index].cluster.values)
    lookup = {clu: idx + 1 for idx, clu in enumerate(clu_vals)}

    kwx = find_kwx(block_path)
    with h5.File(kwx, 'r') as kf, open(features_file, 'w') as f:

        n_features = kf['channel_groups/0/features_masks'].shape[1]

        assert spikes.cluster.values.shape[0] == kf['channel_groups/0/features_masks'].shape[0]

        f.write(' '.join(['cluster_identifier'] + ['feature_' + str(ii) for ii in range(n_features)]) + '\n')

        for indx in spike_index:
            feat = kf['channel_groups/0/features_masks'][indx, :, 0]
            cl = spikes.loc[indx]['cluster']
            f.write(' '.join([str(lookup[cl])] + [str(ii) for ii in feat]) + '\n')

    return clu_vals


def run_isorat(features_file, isorat_output):
    '''
    TODO comment

    Parameters
    ------
    features_file : str
        path to
    isorat_output :

    Returns
    ------
    returncode :

    '''

    cmd = ['/usr/bin/env', 'isorat', features_file, isorat_output]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    for line in p.stdout:
        print(line)
    p.wait()
    return p.returncode


def run_isoi(features_file, isoi_output):
    '''
    TODO comment

    Parameters
    ------
    features_file : str
        path to
    isorat_output : str

    Returns
    ------
    returncode :

    '''

    cmd = ['/usr/bin/env', 'isoi', features_file, isoi_output]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    for line in p.stdout:
        print(line)
    p.wait()
    return p.returncode


def load_isorat_results(isorat_output, clu_vals):
    '''
    TODO comment

    Parameters
    ------
    isorat_output : str
        path
    clu_vals :

    Returns
    ------
    results : pandas Dataframe
    '''

    results = pd.read_csv(isorat_output,
                          delim_whitespace=True,
                          header=None,
                          names=('isolation_distance', 'L_ratio'),
                          )
    results['cluster'] = results.index.map(lambda x: clu_vals[x])
    return results


def load_isoi_results(isoi_output, clu_vals):
    '''
    TODO comment

    Parameters
    ------
    isorat_output : str
        path
    clu_vals :

    Returns
    ------
    results : pandas Dataframe
    '''

    results = pd.read_csv(isoi_output,
                          delim_whitespace=True,
                          header=None,
                          names=('IsoI_BG', 'IsoI_NN', 'NN'),
                          )
    results['cluster'] = results.index.map(lambda x: clu_vals[x])
    results['NN'] = results['NN'].map(lambda x: clu_vals[x - 1])
    return results


def calc_isotools_results(block_path, isorat=True, isoi=True):
    '''
    TODO comment

    Parameters
    ------
    block_path : str
        path to the block
    isorat : boolean

    isoi : boolean

    Returns
    ------
    results : pandas Dataframe

    '''

    if (not isorat) and (not isoi):
        return None

    tmpdir = tempfile.mkdtemp(prefix='isotools-')

    features_file = os.path.join(tmpdir, 'isotools_features.txt')
    clu_vals = make_isotools_features(block_path, features_file)

    processes = []
    if isorat:
        isorat_output = os.path.join(tmpdir, 'isorat_output.txt')
        cmd = ['/usr/local/bin/isorat', features_file, isorat_output]
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, cwd=tmpdir)
        processes.append(p)
    if isoi:
        isoi_output = os.path.join(tmpdir, 'isoi_output.txt')
        cmd = ['/usr/local/bin/isoi', features_file, isoi_output]
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, cwd=tmpdir)
        processes.append(p)
    [p.wait() for p in processes]

    isorat_results = load_isorat_results(isorat_output, clu_vals)
    isoi_results = load_isoi_results(isoi_output, clu_vals)
    results = pd.merge(isorat_results, isoi_results, left_on='cluster', right_on='cluster')

    shutil.rmtree(tmpdir)

    return results