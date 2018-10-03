import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import rigid_pandas, core, events
from timeit import default_timer as timer
import os
from socket import gethostname

__test__ = True if 'lintu' in gethostname() else False

my_block = '/mnt/cube/ephys-example-data/long_acute_rec'
kristas_block = '/mnt/cube/ephys-example-data/Pen01_Lft_AP100_ML1300__Site03_Z2500__B1056_cat_P01_S03_1'
justins_block = '/mnt/cube/ephys-example-data/Pen01_Lft_AP2500_ML1350__Site10_Z2026__B997_cat_P01_S10_Epc10'

def test_acute(block_path):
    start = timer()
    try:
        spikes, stims = rigid_pandas.kwik2rigid_pandas(block_path)
    finally:
        end = timer()
        print "%s took %d seconds or %d minutes to load and align" % (os.path.split(block_path)[-1], end-start, (end-start)/60.0)
    # print spikes.head()
    # print stims.head()

    #spikes['morph_dim'] = spikes[~spikes.stim_name.str.contains('rec')].stim_name.str[0:2]
    #spikes['morph_pos'] = spikes[~spikes.stim_name.str.contains('rec')].stim_name.str[2:].astype(int)


    #rec_stims = good_spikes[good_spikes.stim_name.str.contains('rec')]

    clusters = core.load_clusters(block_path)
    fs = core.get_fs(block_path)
    stim_length = (stims.stim_end.head(1).values[0] - stims.stim_start.head(1).values[0]) / fs
    # print clusters.head()
    cluster = clusters[clusters['quality']=="Good"]['cluster'].iloc[0]
    rigid_pandas.raster_by_unit(spikes, cluster, fs)
    plt.savefig('test.png')

def test_chronic(block_path):
    start = timer()
    try:
        spikes = core.load_spikes(block_path)
        trials = events.get_trials(block_path)
        trials['stim_start'] = trials['time_samples']
        fs = core.get_fs(block_path)
        trials['stim_duration'] = trials['stimulus_end'] - trials['time_samples']
        rigid_pandas.timestamp2time(trials, fs, 'stim_duration')
    finally:
        end = timer()
        print "%s took %d seconds or %d minutes to load" % (os.path.split(block_path)[-1], end-start, (end-start)/60.0)
    start = timer()
    try:
        trials['stim_presentation'] = \
            trials['stimulus'].map(rigid_pandas._EventCounter().count)
        spikes = spikes.join(rigid_pandas.align_events(spikes, trials, columns2copy=['stimulus', 'stim_presentation',
                                                                        'stim_start', 'stim_duration'],
                                          start_label='time_samples', end_label='stimulus_end'))
        spikes['stim_aligned_time'] = (spikes['time_samples'].values.astype('int') -
                                       spikes['stim_start'].values)
        del spikes['time_samples']
        del spikes['stim_start']
        rigid_pandas.timestamp2time(spikes, fs, 'stim_aligned_time')
    finally:
        end = timer()
        print "%s took %d seconds or %d minutes to align" % (os.path.split(block_path)[-1], end-start, (end-start)/60.0)
    clusters = core.load_clusters(block_path)
    cluster = clusters[clusters['quality']=="Good"]['cluster'].iloc[0]
    spikes['stim_name'] = spikes['stimulus']
    rigid_pandas.raster_by_unit(spikes, cluster, fs)
    plt.savefig('test2.png')

def main():
    if 'lintu' in gethostname():
        test_acute(kristas_block)
        test_acute(my_block)
        test_chronic(justins_block)

if __name__ == '__main__':
    main()