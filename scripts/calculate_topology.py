#!/usr/bin/env python
import sys
import os
import argparse
sys.path.append('/home/btheilma/code/ephys-analysis/')
from ephys import topology


def get_args():

	parser = argparse.ArgumentParser(description='Calculate full-segment' 
												 'topology of an ' 
												 'extracellular dataset')
	parser.add_argument('block_path', type=str, help='Path to folder'
													 'containing data files')
	parser.add_argument('windt', type=float, help='Window width in ms')
	parser.add_argument('period', type=str, help='either stim or prestim')
	parser.add_argument('segstart', type=float, help='Time in milliseconds of ' 
													 'start to include relative' 
													 ' to stimulus start')
	parser.add_argument('segend', type=float, help='Time in milliseconds of end'
												   'to include relative to '
												   ' stimulus start')
	return parser.parse_args()

def main():

	args = get_args()

	block_path = os.path.abspath(args.block_path)
	cluster_group = ['Good']
	segment_info = {'period': args.period, 
					'segstart': args.segstart, 
					'segend': args.segend}
	windt = args.windt
	
	topology.calc_bettis_on_dataset(block_path, 
									cluster_group=cluster_group, 
									windt_ms=windt, 
									segment_info=segment_info)


if __name__ == '__main__':
	main()