# Spike funcs.  Routines to deal effectively with data frames containing spike data 
import pandas as pd
import numpy as np

cluster_groups = {'Good': 2, 'Noise': 0, 'MUA': 1, 'Unsorted': 3}

def spike_list_time_slice(spike_data, tl, th):
	# Return a dataframe with all spikes in given time window
	return spike_data[np.logical_and(spike_data['time_stamp'] >= tl, spike_data['time_stamp']<= th)]

def find_spikes_by_cluid(spike_data, cluid):
	# Return a dataframe with all spikes of a given cluid
	return spike_data[spike_data['cluster'] == cluid]

def find_spikes_by_clugroup(spike_data, clugroup):
	# Return a dataframe with all spikes of given cluster group
	clugroup_id = cluster_groups[clugroup]
	return spike_data[spike_data['cluster_group'] == clugroup_id]

def get_cluids(spike_data):
	#returns a list of unique cluster ids
	cluids = np.unique(spike_data['cluster'].values)
	return cluids

def get_clugroups(spike_data):
	return np.squeeze(np.unique(spike_data['cluster_group'].values))

def find_spikes_by_stim_name(spike_data, stim_name):
	# returns a list of all spikes with stim_name
	return spike_data[spike_data['stim_name'] == stim_name]

def get_stim_names(spike_data):
	return np.unique(spike_data['stim_name'].values)

def get_num_trials(spike_data, stim_name):
	# Get the number of presentations of the stim_name
	spike_data_stim = find_spikes_by_stim_name(spike_data, stim_name)
	presentations = spike_data_stim['stim_presentation'].values
	return np.max(presentations) + 1

def find_spikes_by_stim_trial(spike_data, stim_name, trialnum):
	# returns a list of spikes under given stimuli in given repetition of that stimuli
	# first, get spikes by stim name
	spike_data_stim = find_spikes_by_stim_name(spike_data, stim_name)
	return spike_data_stim[spike_data_stim['stim_presentation'] == trialnum]

def get_cluster_group(spike_data, win_l, win_h):
	# returns a list of all cluster ids that spiked within [win_l, win_h]
	spikes_in_win = spike_list_time_slice(spike_data, win_l, win_h)
	win_cluids = get_cluids(spikes_in_win)
	return win_cluids

def get_stim_times(spike_data, stim_name, trialnum):
	# Returns the stim start and stim end times for a given repetition of stim_name
	spike_data_stim = find_spikes_by_stim_trial(spike_data, stim_name, trialnum)
	stim_starts = np.unique(spike_data_stim['stim_time_stamp'].values.astype(int))
	assert np.size(stim_starts) == 1, "Too many stim starts!"
	stim_ends = np.unique(spike_data_stim['stim_end_time_stamp'].values.astype(int))
	assert np.size(stim_ends) == 1, "Too many stim ends!"

	stim_start = np.squeeze(stim_starts).tolist()
	stim_end = np.squeeze(stim_ends).tolist()
	return [stim_start, stim_end]

def get_stim_duration(spike_data, stim_name, trialnum):
	# Returns the duration of stim given by stim_name on trial trialnum
	spike_data_stim = find_spikes_by_stim_trial(spike_data, stim_name, trialnum)
	stim_starts = np.unique(spike_data_stim['stim_time_stamp'].values.astype(int))	
	stim_ends = np.unique(spike_data_stim['stim_end_time_stamp'].values.astype(int))
	stim_durations = np.unique(stim_ends - stim_starts)
	if (np.diff(stim_durations) < 5).all():
		return (np.squeeze(max(stim_durations)).tolist())
	else:
		print("Stimuli duration differences outside bounds\n")

def get_num_spikes(spike_data):
	return np.size(spike_data['time_stamp'].values)

def get_spike_times_samples(spike_data):
	return spike_data['time_stamp'].values

def get_spike_times_seconds_stim_aligned(spike_data):
	return spike_data['stim_aligned_time_stamp_seconds'].values