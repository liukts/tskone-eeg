import load_local_neo_odml_elephant

import os

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as psig

import quantities as pq

from neo import Block, Segment
from elephant.signal_processing import butter

from reachgraspio import reachgraspio
from neo_utils import add_epoch, cut_segment_by_epoch, get_events

datapath = './data/multielectrode_grasp'
channels = 96

# Specify the path to the recording session to load, eg,
# '/home/user/l101210-001'
session_name = os.path.join(datapath,'datasets','i140703-001')
# session_name = os.path.join('..', 'datasets', 'l101210-001')
odml_dir = os.path.join(datapath,'datasets')

# Open the session for reading
session = reachgraspio.ReachGraspIO(session_name, odml_directory=odml_dir)

##################################################
# construct data in better format
##################################################

# extract and save displacement data
print(f'Movement channel')
print(f'Reading block...', end=' ')
data_block = session.read_block(
    nsx_to_load=2,
    n_starts=None, n_stops=None,
    channels=[143],units='all',
    load_events=True, load_waveforms=False, scaling='voltage',
    correct_filter_shifts=True
)
print(f'finished')
displ = data_block.segments[0].analogsignals[0].squeeze()

# extract analog neural data (lfp and spikes)
filtered_anasig = []
spiketimes = []
for i in range(1,channels+1):
    print(f'Neural channel {i}/{channels}')
    # Read data into memory as a block
    print(f'Reading block...', end=' ')
    data_block = session.read_block(
        nsx_to_load=2,
        n_starts=None, n_stops=None,
        channels=[i],units='all',
        load_events=True, load_waveforms=True, scaling='voltage',
        correct_filter_shifts=True
    )
    print(f'finished')
    # analog signals
    data_segment = data_block.segments[0]
    for i,anasig in enumerate(data_block.segments[0].analogsignals):
        # Use the Elephant library to filter the analog signal
        # f_anasig = butter(
        #         anasig,
        #         highpass_freq=None,
        #         lowpass_freq=300 * pq.Hz,
        #         order=4)
        # downsample to 1 kHz
        # ds_anasig = psig.resample(anasig.squeeze()[:int(30e6)],int(1000*30e6/30000),anasig.times[:int(30e6)])
        filtered_anasig.append(anasig.squeeze())
        # print(len(filtered_anasig[i-1]))

    # spike signals
    spiketimes.append(data_block.segments[0].spiketrains[0])
    for i,st in enumerate(data_block.segments[0].spiketrains):
        # print(i)
        pass
    # t = data_segment.analogsignals[0].times
    # v = data_segment.analogsignals[0].squeeze()
    # plt.plot(t,v)
    # # plt.xlim([0,1e3])
    # plt.savefig('test.png')
    # # print(data_segment.analogsignals)
