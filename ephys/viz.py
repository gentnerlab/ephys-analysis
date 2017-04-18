import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from core import load_probe, load_fs, load_clusters
from clust import get_mean_waveform_array, upsample_spike, find_mean_masks
from clust import get_cluster_coords, mean_masks_w

def plot_cluster(block_path,clu,chan_alpha=None,scale_factor=0.05,color='0.5',**plot_kwargs):
    '''
    Plots the mean waveforms on each channel for a single cluster, in the 
        geometric layout from the probe file. 
    
    Parameters
    ------
    block_path : str
        the path to the block
    clu : int
        the cluster identifier
    chan_alpha : numpy array
        an array of alpha values to use for each channel
    scale_factor : float
        the factor to scale the waveforms (default: 0.05)
    color
        color to plot (default: '0.5')
    kwargs
        keyword arguments are passed to the plot function
    
    '''

    #load the probe coordinates
    prb_info = load_probe(block_path)
    channels = prb_info.channel_groups[0]['channels']
    geometry = prb_info.channel_groups[0]['geometry']
    coords = np.array([geometry[ch] for ch in channels])

    if chan_alpha is None:
        chan_alpha = np.ones(len(coords))

    mean_waveform_array = get_mean_waveform_array(block_path,clu)
    
    for waveform,xy,alpha in zip(mean_waveform_array.T,coords,chan_alpha):
        plt.plot(xy[0]+np.arange(len(waveform))-len(waveform)/2,
                 waveform*scale_factor+xy[1],
                 color=color,
                 alpha=alpha,
                 **plot_kwargs
                )
        try:
            # if we get a label, let's only apply it once
            label = plot_kwargs.pop('label')
        except KeyError:
            pass



def plot_all_clusters(block_path,clusters=None,quality=('Good','MUA'),**kwargs):
    '''
    Plots the mean waveforms for all clusters in a block, in the 
        geometric layout from the probe file. 
    
    Parameters
    ------
    block_path : str
        the path to the block
    clusters : pandas dataframe, optional
        cluster dataframe. default: load all clusters
    quality : tuple of quality values
        an array of alpha values to use for each channel. default: ('Good','MUA')
    kwargs
        keyword arguments are passed to the plot function
    
    '''
    
    if clusters is None:
        clusters = load_clusters(block_path)
        clusters = clusters[clusters.quality.isin(quality)]

    clusters = (
        clusters
        .sort_values(['quality','cluster'])
        .reset_index()
        )

    palette = sns.color_palette("hls", len(clusters))

    for idx,cluster_row in clusters.iterrows():

        lbl = "{}({})".format(cluster_row.cluster,cluster_row.quality)

        # use mean mask for alpha transparency
        mean_masks_array = np.fromfile(
            find_mean_masks(block_path,cluster_row.cluster),
            dtype=np.float32
            )

        plot_cluster(block_path,
                     cluster_row.cluster,
                     color=palette[idx],
                     chan_alpha=mean_masks_array,
                     label=lbl,
                     **kwargs
                     )

    plt.axis('equal')
    leg = plt.legend(loc='center left')
    [plt.setp(label, alpha = 1.0) for label in leg.get_lines()]
    sns.despine(bottom=True,left=True)
    plt.xticks([])
    plt.yticks([])



def plot_spike_shape(block_path,clu,normalize=True,**kwargs):
    '''
    Plots the upsampled spike shape, aligned to the trough. 
    
    Parameters
    ------
    block_path : str
        the path to the block
    clu : int
        the cluster identifier
    normalize : boolean
        scale the spike shape to the depth of the trough
    kwargs
        keyword arguments are passed to the plot function
    
    '''
    fs = load_fs(block_path)
    exemplar = get_spike_exemplar(block_path,clu)
    
    time,shape = upsample_spike(exemplar,fs)
    time -= time[shape.argmin()]
    
    if normalize==True:
        shape /= -shape[shape.argmin()]
    
    plt.plot(time,shape,**kwargs)

def plot_cluster_locations(block_path,clusters=None,quality=('Good','MUA'),bin_width=50.0,**kwargs):
    '''
    Plots the distribution of cluster locations.
    
    Parameters
    ------
    block_path : str
        the path to the block
    clusters : pandas dataframe, optional
        cluster dataframe. default: load all clusters
    quality : tuple of quality values
        an array of alpha values to use for each channel. default: ('Good','MUA')
    bin_width : float
        width of bins for marginal distribution plots
    kwargs
        keyword arguments are passed to the plot function
    
    '''
    
    if clusters is None:
        clusters = load_clusters(block_path)
        clusters = clusters[clusters.quality.isin(quality)]

    if 'x_probe' not in clusters.columns:
        clusters['x_probe'] = clusters['cluster'].map(
            lambda clu: get_cluster_coords(block_path,clu,weight_func=mean_masks_w)[0]
        )

    if 'y_probe' not in clusters.columns:
        clusters['y_probe'] = clusters['cluster'].map(
            lambda clu: get_cluster_coords(block_path,clu,weight_func=mean_masks_w)[1]
        )

    prb_info = load_probe(block_path)
    coords = np.array(prb_info.channel_groups[0]['geometry'].values())

    x_bds = coords[:,0].min(),coords[:,0].max()
    x_mid = sum(x_bds) / 2.0
    y_bds = coords[:,1].min(),coords[:,1].max()
    y_mid = sum(y_bds) / 2.0
    bins = np.arange(coords.min(),coords.max()+2*bin_width,bin_width)-bin_width/2
    half_span = (bins.max()-bins.min()) / 2

    loc_plot = sns.jointplot('x_probe','y_probe',data=clusters,
                             stat_func=None,
                             marginal_kws={'bins':bins},
                             **kwargs)

    loc_plot.ax_joint.set_aspect('equal')
    loc_plot.ax_joint.set_xlim(x_mid-half_span,x_mid+half_span)
    loc_plot.ax_joint.set_ylim(y_mid-half_span,y_mid+half_span)
    loc_plot.ax_joint.set_xticks(np.unique(coords[:,0]))
    loc_plot.ax_joint.set_yticks(np.unique(coords[:,1]))
