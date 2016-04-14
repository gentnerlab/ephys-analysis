

def make_isotools_features(block_path):
    
    clusters = get_clusters(block_path)
    neurons = clusters[clusters.quality.isin(['Good','MUA'])].sort_values(['quality','cluster']).reset_index()
    
    spikes = get_spikes(block)    
    spike_index = spikes.cluster.isin(neurons.cluster).values
    
    kwx = get_kwx(block_path)
    
    neuronal_spikes = spikes[spike_index]
    
    clu_vals = np.unique(spikes.cluster.values)
    lookup = {clu:idx+1 for idx,clu in enumerate(clu_vals)}

    with h5.File(kwx,'r') as kf, open('features.txt','w') as f:
        features = kf['channel_groups/0/features_masks'][spike_index,:,0]

        f.write(' '.join(['cluster_identifier']+['feature_'+str(ii) for ii in range(features.shape[1])])+'\n')

        assert spikes.cluster.values.shape[0]==features.shape[0]
        
        for cl,feat in zip(spikes.cluster.values,features):
            f.write(' '.join([str(lookup[cl])]+[str(ii) for ii in feat])+'\n')
    return clu_vals


def run_isorat(features_file,isorat_output):
    cmd = ['/home/jkiggins/Code/isotools/bin/isorat',features_file,isorat_output]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    for line in p.stdout:
        print line
    p.wait()
    return p.returncode

def run_isoi(features_file,isoi_output):
    cmd = ['/home/jkiggins/Code/isotools/bin/isoi',features_file,isoi_output]
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