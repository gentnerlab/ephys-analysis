import os
import numpy as np
import h5py as h5
import pandas as pd
from .core import load_clusters, load_spikes, find_kwx


def make_isotools_features(block_path,features_file,do_noise=False):
    
    clusters = load_clusters(block_path)

    if do_noise:
        neurons = clusters.sort_values(['quality','cluster']).reset_index()
    else:
        neurons = clusters[clusters.quality!='Noise'].sort_values(['quality','cluster']).reset_index()
    
    spikes = load_spikes(block_path)    
    spike_index = np.where(spikes.cluster.isin(neurons.cluster).values==True)[0]
    
    clu_vals = np.unique(spikes.loc[spike_index].cluster.values)
    lookup = {clu:idx+1 for idx,clu in enumerate(clu_vals)}

    kwx = get_kwx(block_path)
    with h5.File(kwx,'r') as kf, open(features_file,'w') as f:
        
        n_features = kf['channel_groups/0/features_masks'].shape[1]
        
        assert spikes.cluster.values.shape[0]==kf['channel_groups/0/features_masks'].shape[0]
        
        f.write(' '.join(['cluster_identifier']+['feature_'+str(ii) for ii in range(n_features)])+'\n')
        
        for indx in spike_index:
            
            feat = kf['channel_groups/0/features_masks'][indx,:,0]
            cl = spikes.loc[indx]['cluster']
            f.write(' '.join([str(lookup[cl])]+[str(ii) for ii in feat])+'\n')
            
    return clu_vals


def run_isorat(features_file,isorat_output):
    cmd = ['isorat',features_file,isorat_output]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    for line in p.stdout:
        print line
    p.wait()
    return p.returncode

def run_isoi(features_file,isoi_output):
    cmd = ['isoi',features_file,isoi_output]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    for line in p.stdout:
        print line
    p.wait()
    return p.returncode
            
def get_isorat_results(isorat_output,clu_vals):
    results = pd.read_csv(isorat_output,
                          delim_whitespace=True,
                          header=None,
                          names=('isolation_distance','L_ratio'),
                         )
    results['cluster'] = results.index.map(lambda x: clu_vals[x])
    return results

def get_isoi_results(isoi_output,clu_vals):
    results = pd.read_csv(isoi_output,
                          delim_whitespace=True,
                          header=None,
                          names=('IsoI_BG','IsoI_NN','NN'),
                         )
    results['cluster'] = results.index.map(lambda x: clu_vals[x])
    results['NN'] = results['NN'].map(lambda x: clu_vals[x-1])
    return results

def get_isotools_results(block_path):
    
    features_file = os.path.join(block_path,'isotools_features.txt')
    clu_vals = make_isotools_features(block_path,features_file)

    isorat_output = os.path.join(block_path,'isorat_output.txt')
    run_isorat(features_file,isorat_output)

    isoi_output = os.path.join(block_path,'isoi_output.txt')
    run_isoi(features_file,isoi_output)

    isorat_results = get_isorat_results(isorat_output,clu_vals)
    isoi_results = get_isoi_results(isoi_output,clu_vals)

    return pd.merge(isorat_results,isoi_results,left_on='cluster',right_on='cluster')

