# Converts open ephys format files to kwd files
import os
import glob
import numpy as np
import h5py as h5

# OpenEphys constants
NUM_HEADER_BYTES = 1024
SAMPLES_PER_RECORD = 1024
BYTES_PER_SAMPLE = 2
RECORD_SIZE = 4 + 8 + SAMPLES_PER_RECORD*BYTES_PER_SAMPLE + 10
RECORD_RECORDING_OFFSET = 8+2


def find_continuous_files(data_dir):
    return glob.glob(os.path.join(data_dir, '*.continuous'))

def find_events_files(data_dir):
    return glob.glob(os.path.join(data_dir, '*.events'))

def get_n_channels(continuous_files):
    return len(continuous_files)

def initialize_kwd_file(data_dir, experiment_name):
    # Create an empty KWD file with the appropriate structur
    kwd_filename = experiment_name + '.raw.kwd'
    kwd_file = h5.File(os.path.join(data_dir, kwd_filename), "w-")
    kwd_file.create_group("recordings")

    continuous_files = find_continuous_files(data_dir)
    unique_recording_numbers = get_openephys_unique_recording_numbers(continuous_files)
    for rec_num in unique_recording_numbers:
        kwd_file.create_group("/recordings/{}".format(rec_num))
    return kwd_file

def read_openephys_header(oe_file):
    # return a dictionary of open ephys header values
    oe_header = {}
    oe_file.seek(0)
    h = oe_file.read(1024).decode().replace('\n', '').replace('header.','')
    for ind, header_item in enumerate(h.split(';')):
        if '=' in header_item:
            header_key = header_item.split(' = ')[0]
            header_value = header_item.split(' = ')[1]
            oe_header[header_key] = header_value
    return oe_header

def calculate_openephys_continuous_sizes(oe_file):
    # Calculate: Number of Records, Number of Samples
    n_file_bytes = os.fstat(oe_file.fileno()).st_size
    n_record_bytes = n_file_bytes - NUM_HEADER_BYTES
    
    # Check if file is consistent
    if n_record_bytes % RECORD_SIZE != 0:
        raise Exception("File size inconsistent: possible corrupt data file")

    n_records = n_record_bytes // RECORD_SIZE
    n_samples = n_records * SAMPLES_PER_RECORD
    return (n_records, n_samples)

def load_openephys_continuous_record(oe_file, record_num):
    # Loads a single record from an openephys continous file
    
    # Calculate start of record in bytes
    record_start = 1024 + record_num*RECORD_SIZE

    # Move to start of record
    oe_file.seek(record_start)

    # Get Timestamp
    record_timestamp = np.fromfile(oe_file, np.dtype('<i8'), 1)
    
    # Get Number of samples
    record_n_samples = np.fromfile(oe_file, np.dtype('<u2'), 1)[0]

    if record_n_samples != SAMPLES_PER_RECORD:
        raise Exception('Found corrupted record in ' + str(record_num))

    # Get recording number
    record_recording_number = np.fromfile(oe_file, np.dtype('>u2'), 1)[0]

    # Get raw record data
    record_raw_data = np.fromfile(oe_file, np.dtype('>i2'), record_n_samples)

    # Ignore record marker
    oe_file.read(10)  

    # Make array of timesamples
    sample_numbers = np.arange(record_timestamp, 
                               record_timestamp + record_n_samples)
    
    # Make data array
    data = np.zeros((record_n_samples, 2))
    data[:, 0] = sample_numbers
    data[:, 1] = record_raw_data
    recording_start_sample = int(data[0, 0])

    return (record_num, record_recording_number, recording_start_sample, data)
    
def get_openephys_continuous_recording_numbers(oe_file):
    # Returns an array of recording numbers within a .continuous file
    recordings = []
    (n_records, n_samples) = calculate_openephys_continuous_sizes(oe_file)
    for record in range(n_records):
        record_offset = 1024 + record*RECORD_SIZE
        recording_offset = record_offset + RECORD_RECORDING_OFFSET
        oe_file.seek(recording_offset)
        record_recording_number = np.fromfile(oe_file, np.dtype('>u2'), 1)[0]
        recordings.append(record_recording_number)
    return recordings

def get_openephys_continuous_record_recording_number(oe_file, record):
    offset = 1024 + record*RECORD_SIZE + RECORD_RECORDING_OFFSET
    oe_file.seek(offset)
    return np.fromfile(oe_file, np.dtype('>u2'), 1)[0]

def get_openephys_unique_recording_numbers(continuous_files):
    recordings = []
    for cf in continuous_files:
        with open(cf, 'rb') as f:
            cf_unique_recordings = get_openephys_continuous_recording_numbers(f)
            recordings = recordings + cf_unique_recordings
    return list(set(recordings))
            
def check_n_samples_consistency(continuous_files):
    n_samples_list = []
    for cf in continuous_files:
        with open(cf, "rb") as oe_file:
            n_samples = calculate_openephys_continuous_sizes(oe_file)[1]
            n_samples_list.append(n_samples)
    if len(set(n_samples_list)) != 1:
        raise Exception("Inconsistent number of samples across channels")
    return list(set(n_samples_list))[0]

def get_openephys_channel_number(continuous_file):
    cf_filename = os.path.split(continuous_file)[1]
    cf_filename = cf_filename.replace('.continuous', '')
    if 'CH' in cf_filename:
        channel_number = int(cf_filename.split('CH')[-1]) - 1
    if 'ADC' in cf_filename:
        channel_number = int(cf_filename.split('ADC')[-1]) - 12
    if 'AUX' in cf_filename:
        channel_number = int(cf_filename.split('AUX')[-1]) - 4
    return channel_number
    
def build_kwd_recording(kwd_file, continuous_files, recording_number):
    n_channels = get_n_channels(continuous_files)
    n_samples = check_n_samples_consistency(continuous_files)
    #data = np.zeros((n_samples, n_channels), dtype=np.int16)

    dset = kwd_file.create_dataset("/recordings/{}/data".format(recording_number),
                            (n_samples, n_channels), dtype='int16')

    for cf in continuous_files:
        # get channel number
        channel = get_openephys_channel_number(cf)
        print("Storing channel {} of {}".format(channel, n_channels))
        with open(cf, "rb") as oe_file:
            (n_records, n_oe_file_samples) = calculate_openephys_continuous_sizes(oe_file)
            for record in range(n_records):
                if get_openephys_continuous_record_recording_number(oe_file, record) != recording_number:
                    continue
                record_data_list = load_openephys_continuous_record(oe_file, record)
                record_data = record_data_list[3] 
                record_start = record_data_list[2]
                #data[record_data[0, :], channel] = record_data[1, :] 
                record_sample_lo = int(record_data[0, 0]) - record_start
                record_sample_hi = int(record_data[-1, 0]+1) - record_start
                #print(record_sample_lo, record_sample_hi)
                #print(channel, record, n_samples, n_records)
                dset[record_sample_lo:record_sample_hi, channel] = record_data[:, 1]
