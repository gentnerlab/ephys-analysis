#!/usr/bin/env python
# cellgroups.py

import numpy as np
import pandas as pd 
import os
import sys
import argparse

sys.path.append('/home/btheilma/code/ECAnalysis/')
import spike_funcs as spf 

spikedatafile = "./somefile.pd"
bird = 'B999'

def debug_print(text):
	sys.stdout.write(str(text))
	sys.stdout.flush()

def get_args():
    parser = argparse.ArgumentParser(description='Make cellgroups with a certain window size')
    parser.add_argument('datafile', nargs='?', help='Path to directory with PANDAS DataFrame containing spike data')
    parser.add_argument('destdir', default='./', nargs='?', help='Directory in which to place raster plots')
    parser.add_argument('-t', dest='win_dt', type=float, default=50.0, help='Window size in milliseconds')
    parser.add_argument('-n', dest='numstarts', type=int, default=5, help='Number of window starts')
    parser.add_argument('-p', dest='prestim', type=float, default=2.0, help='Prestim time period in seconds')
    parser.add_argument('-f', dest='fs', type=float, default=31250.0, help='Sampling rate in Hertz')
    parser.add_argument('-c', dest='clu_group', type=str, default='gm', help='Cluster classes to included: g=Good, m=MUA, gm=Good + MUA')
    return parser.parse_args()

def main():

	args = get_args()
	spikedata = pd.read_pickle(args.datafile)

	prestim_dt = args.prestim # seconds pre stim to take
	win_n = args.numstarts
	win_dt = args.win_dt #milliseconds
	fs = args.fs
	clu_group = args.clu_group

	if clu_group == 'g':
		spikedata = spf.find_spikes_by_clugroup(spikedata, 'Good')
	elif clu_group == 'm':
		spikedata = spf.find_spikes_by_clugroup(spikedata, 'MUA')
	
	debug_print('Running make_cell_groups...\n')
	make_cell_groups(spikedata, win_dt, win_n, prestim_dt, fs, clu_group, args.destdir)


def make_cell_groups(spikedata, win_dt, win_n, prestim_dt, fs, clu_group, destdir):
	stim_names = spf.get_stim_names(spikedata)
	for stim in stim_names:
		ntrials = spf.get_num_trials(spikedata, stim)
		for trial in range(ntrials):
			debug_print('make_cell_groups: Stim %s, Trial %s...\n' % (stim, trial))
			stim_period_vert_list = set()
			prestim_vert_list = set()

			debug_print('- Extracting spikes\n')
			stimtimes = spf.get_stim_times(spikedata, stim, trial)
			trialdata = spf.find_spikes_by_stim_trial(spikedata, stim, trial)
			prestimwin = [stimtimes[0] - 2.0, stimtimes[0]]

			debug_print('- Creating Windows\n')
			# Subdivide a given time period into windows
			prestim_cg_win_list = win_subdivide(prestimwin, win_n, win_dt, fs)
			stim_cg_win_list = win_subdivide(stimtimes, win_n, win_dt, fs)

			debug_print('- Extracting stim period cell groups\n')
			for winl, winh in stim_cg_win_list:
				# debug_print('WinL: %s 	WinH: %s\n' % (winl, winh))
				# Get cell groups
				cgs = spf.get_cluster_group(trialdata, winl, winh)
				#debug_print(str(cgs) + '\n')
				# Convert to tuple and add to the vertex set list. 
				stim_period_vert_list.add(tuple(cgs))
				#debug_print('Vert list size: %s\n' % len(stim_period_vert_list))
			
			debug_print('- Extracting prestim period cell groups\n')
			for winl, winh in prestim_cg_win_list:
				cgs = spf.get_cluster_group(trialdata, winl, winh)
				prestim_vert_list.add(tuple(cgs))

			debug_print('- Writing perseus input files\n')	
			write_vert_list_to_perseus(stim_period_vert_list, destdir, stim, trial, bird, clu_group)
			write_vert_list_to_perseus(prestim_vert_list, destdir, 'pretrial'+stim, trial, bird, clu_group)
			debug_print('DONE\n')


def win_subdivide(win, nstarts, dt, fs):
	# given a large win, returns a list of subdivisions
	# nsubwin x 2 array:   Win1L Win1H
	#					   Win2L Win2H
	#					   Win3L Win3H
	dtsamps = np.floor(dt*fs/1000.)
	winL, winH = win
	a = range(nstarts)
	subwin_starts = np.multiply(a, np.floor(1.0*dtsamps/nstarts)) + winL
	subwin=[]
	for wins in subwin_starts:
	    subwin_s = np.arange(wins, winH, dtsamps)
	    for s in subwin_s:
	        e = s+dtsamps
	        winentry = [s, e]
	        subwin.append(winentry)
	return subwin

def write_vert_list_to_perseus(vert_list, destdir, stimn, trialnum, bird, clu_group):
	# first create the output file name:
	fname = bird + '_' + clu_group + '_' + stimn + '_' + str(trialnum) +'.pers'
	fname = os.path.join(destdir, fname)
	debug_print('Writing to... %s\n' % fname)

	with open(fname, 'w') as fd:
		#write num coords per vertex
		fd.write('1\n')
		for grp in vert_list:
			#debug_print('Cell group: ' + str(grp) +'\n')
			grp_dim = len(grp) - 1
			if grp_dim < 0:
				continue
			vert_str = str(grp)
			vert_str = vert_str.replace('(', '')
			vert_str = vert_str.replace(')', '')
			vert_str = vert_str.replace(' ', '')
			vert_str = vert_str.replace(',', ' ')
			out_str = str(grp_dim) + ' ' + vert_str + ' 1\n'
			#debug_print('Writing: %s' % out_str)
			fd.write(out_str)

if __name__ == '__main__':
	main()

