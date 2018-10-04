from __future__ import absolute_import
import h5py
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from . import core


def kwik2rigid_pandas(block_path):
    '''
    loads data in a manner that is very useful for acute experiments

    Parameters
    ------
    block_path : str
        the path to the block

    Returns
    ------
    spikes : Pandas.DataFrame
        columns: cluster, recording, stim_name, stim_presentation, 
                 stim_duration, stim_aligned_time
    stims : Pandas.DataFrame
        columns: stim_start, stim_end, stim_name, stim_presentation, 
                 stim_duration

    '''
    spikes = core.load_spikes(block_path)
    stims = load_acute_stims(block_path)
    count_events(stims)
    fs = core.load_fs(block_path)
    stims['stim_duration'] = stims['stim_end'] - stims['stim_start']
    timestamp2time(stims, fs, 'stim_duration')
    spikes = spikes.join(align_events(spikes, stims))
    spikes['stim_aligned_time'] = (spikes['time_samples'].values.astype('int') -
                                   spikes['stim_start'].values)
    del spikes['time_samples']
    del spikes['stim_start']
    timestamp2time(spikes, fs, 'stim_aligned_time')
    return spikes, stims


def load_acute_stims(block_path):
    '''
    Fast code to load up stimuli information for an acute recording
    Makes fewer checks and more assumptions than core.load_trials()
    Doesn't include behavior only columns.
    ~8000x speedup on an example acute dataset.

    Parameters
    -------
    block_path : str
        the path to the block

    Returns
    ------

    trials : pandas dataframe

    Columns
    ------

    stim_start : int
        Time in samples of the start of a stimulus (trial)
    stim_name : str
        Name of the stimulus
    stim_end : int
        Time in samples of the end of the stimulus
    '''
    stims = core.load_events(block_path, 'DigMark')
    # assumes one start and one end for each trial
    stims.loc[stims['codes'] == b'<', 'stim_end'] = stims[
        stims['codes'] == b'>']['time_samples'].values
    stims = stims[stims['codes'] == b'<']
    # on some recs there are random date entries in the stim text field at the
    # start... removing them here
    stimdat = core.load_events(block_path, 'Stimulus')
    stims['stim_name'] = stimdat['text'][stimdat['time_samples'] > stims['time_samples'].min()][
        1::2].values
    stims.reset_index(drop=True, inplace=True)
    del stims['codes']
    stims.rename(columns={'time_samples': 'stim_start'}, inplace=True)
    return stims


def count_events(events, index='stim_name', target='stim_presentation'):
    '''
    Adds a column containing the event index

    Parameters
    -------
    events : pandas dataframe containing events to count,
                as from load_acute_stims()
    index : str
        Column to use for counter keys. Default: 'stim_name'
    target : str
        Column to drop event counts into. Default: 'stim_presentation'

    '''
    events[target] = events[index].map(_EventCounter().count)


def timestamp2time(df, sample_rate, time_stamp_label,
                   time_label=None, inplace=True):
    '''
    Converts a column from time samples to time in seconds

    Parameters
    ------
    df : Pandas.DataFrame
        DataFrame containing column indicated by time_stamp_label
        df will be modified and not returned
    sample_rate : int
        sample rate, from core.load_fs()
    time_stamp_label : str
        label of column with time stamp DataFrame
    time_label : str
        label of target column
        if None (default) leaves time data in df[time_stamp_label]
    inplace : boolean
        whether to overwrite df[time_stamp_label]
    '''
    if inplace:
        df[time_stamp_label] = df[time_stamp_label].values / sample_rate
        if time_label:
            df.rename(columns={time_stamp_label: time_label}, inplace=True)
    else:
        assert time_label, 'must provide time_label if not inplace'
        df[time_label] = df[time_stamp_label].values / sample_rate


def raster_by_unit(spikes, cluster, sample_rate, window_size=1, plot_by='stim_name', col_wrap=None):
    sns.set_context("notebook", font_scale=1.5,
                    rc={'lines.markeredgewidth': .1, 'patch.linewidth': 1})
    sns.set_style("white")
    num_repeats = np.max(spikes['stim_presentation'].values)
    num_stims = len(np.unique(spikes[plot_by]))
    if col_wrap is None:
        col_wrap = int(np.sqrt(num_stims))
    g = sns.FacetGrid(spikes[spikes['cluster'] == cluster],
                      col=plot_by, col_wrap=col_wrap)
    g.map(_raster, "stim_aligned_time", "stim_presentation", "stim_duration",
          window_size=window_size)
    g = g.set_titles("cluster %d, stim: {col_name}" % (cluster))


def _raster(stim_aligned_time, stim_presentation, stim_duration, window_size=1, **kwargs):
    plt.scatter(stim_aligned_time, stim_presentation, marker='|', **kwargs)
    num_repeats = np.max(stim_presentation)
    stim_length = stim_duration.iloc[0]
    plt.plot((0, 0), (0, num_repeats), c=".2", alpha=.5)
    plt.plot((stim_length, stim_length), (0, num_repeats), c=".2", alpha=.5)
    plt.xlim((-window_size, stim_length + window_size))
    plt.ylim((0, num_repeats))

from collections import Counter


class _EventCounter(Counter):

    def count(self, key):
        self[key] += 1
        return self[key] - 1


def align_events(spikes, events, columns2copy=['stim_name', 'stim_presentation',
                                               'stim_start', 'stim_duration'],
                 start_label='stim_start', end_label='stim_end'):
    '''
    Generates a dataframe that labels spikes as belonging to event windows
    Event windows must be non-overlapping
    Spikes that lie between event windows will be assigned to the closest window

    O(len(spikes) + len(events))

    Parameters
    -------
    spikes : Pandas.DataFrame
        from core.load_spikes
    events : Pandas.DataFrame
        such as from load_acute_stims or core.load_trials
        must have non-overlapping windows defined by start_label and end_label
    columns2copy : iterable of strings
        labels of columns of events that you want to populate spike_stim_info_df
        in the order you want them
    start_label : str
        label of the column of events corresponding to the start of the event
    end_label : str
        label of the column of events corresponding to the end of the event


    Returns
    ------
    spike_event_info_df : Pandas.DataFrame
        Spikes assigned to events
        This DataFrame is indexed by spikes.index
        Contains columns indicated by columns2copy

    '''
    data = []
    grouped_spikes = spikes.groupby('recording')
    for recording, event_recording_group in events.groupby('recording'):
        data.extend(grouped_spikes.get_group(recording)["time_samples"].map(
            _EventAligner(event_recording_group, output_labels=columns2copy,
                          start_label=start_label, end_label=end_label,
                          event_index=event_recording_group.index[0]).event_checker))
    return pd.DataFrame(data=data, columns=columns2copy, index=spikes.index)


class _EventAligner(object):
    # TODO: duplicate spikes that are <2 sec from 2 stimuli

    def __init__(self, events, output_labels, start_label='stim_start',
                 end_label='stim_end', event_index=0):
        self.event_index = event_index
        self.start_event_index = event_index
        self.events = events

        event_columns = list(events.keys().get_values())
        self.output_indices = \
            [event_columns.index(lbl) for lbl in output_labels]
        self.start_index = event_columns.index(start_label)
        self.end_index = event_columns.index(end_label)

        self.prev_event = None
        self.cur_event = self.events.loc[self.event_index].values
        self.next_event = self.events.loc[self.event_index + 1].values

    def event_checker(self, time_stamp):
        if time_stamp < self.cur_event[self.start_index]:
            if self.event_index == self.start_event_index or \
                    self.cur_event[self.start_index] - time_stamp < \
                    time_stamp - self.prev_event[self.end_index]:
                return self.cur_event[self.output_indices]
            else:
                return self.prev_event[self.output_indices]
        elif time_stamp < self.cur_event[self.end_index]:
            return self.cur_event[self.output_indices]
        else:
            if self.event_index + 1 < len(self.events):
                self.event_index += 1
                self.prev_event = self.cur_event
                self.cur_event = self.next_event
                if self.event_index + 1 < len(self.events):
                    self.next_event = self.events.loc[self.event_index + 1].values
                else:
                    self.next_event = None
                return self.event_checker(time_stamp)
            else:
                return self.cur_event[self.output_indices]
