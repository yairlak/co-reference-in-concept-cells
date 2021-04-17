#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 14:00:47 2021

@author: yl254115
"""

import pandas as pd
import os, argparse
import numpy as np
import mne
from neo.io import NeuralynxIO
import matplotlib.pylab as plt
from utils import get_trialInfo, load_spike_data, get_event_id, epochs2cherries, update_paths
from plotting import generate_rasters

parser = argparse.ArgumentParser()
# EXPERIMENT
parser.add_argument('--patient', default=83, type=int)
parser.add_argument('--session', default=3, type=int)
parser.add_argument('--channel', default=[])
parser.add_argument('--block', default='syntactic', choices = ['pragmatic', 'syntactic', 'all_trials'])
# PLOTTING
parser.add_argument('--plot-spike-profile', default = False, action='store_true', help='Adds spike profile to plot.')
parser.add_argument('--times', default = [0, 2000], help='Plotting window')
parser.add_argument('--line-length', default = 0.9, type=int, help='Line length in rasters')
parser.add_argument('--smooth-raster', default = 100, type=int, help='Gaussian window for firing rate (msec)')
# PARADIGM
parser.add_argument('--soa', default = 700, type=int, help='(msec)')
parser.add_argument('--isi-pragmatic', default = 1000, type=int, help='(msec) break duration between sentence1 and sentence2 in the pragmatic block')
parser.add_argument('--isi-question', default = 2000, type=int, help='(msec) break duration between sentence1 and sentence2 in the pragmatic block')
parser.add_argument('--fixation-time', default = 300, type=int, help='(msec)')
# PATHS
parser.add_argument('--path2data', default='/home/yl254115/projects/coreference/')
parser.add_argument('--path2figures', default='/home/yl254115/projects/coreference/figures/')
parser.add_argument('--spike_format', choices = ['mat', 'h5'], default='mat')
# FLAGS
parser.add_argument('--lfp', action='store_true', default=False, help='Extract LFP data')

args = parser.parse_args()

args = update_paths(args)
if args.patient == 83:
    args.isi_pragmatic -= 500
print(args)


############
# METADATA #
############
TrialInfo = get_trialInfo(args.path2trialInfo, args)
metadata = pd.DataFrame(TrialInfo)


################
# READ RAW NCS #
################
reader = NeuralynxIO(dirname=args.path2data)
args.tmin, args.tmax = reader.global_t_start, reader.global_t_stop
args.sampling_freq_lfp = reader._sigs_sampling_rate
blks = reader.read(lazy=False)
channels = reader.header['signal_channels']
n_channels = len(channels)
channel_keys = [(channel[0], channel[1]) for channel in channels]
ncs_filenames = reader.ncs_filenames
CSCs = [os.path.basename(ncs_filenames[k]) for k in channel_keys]
CSC_nums = sorted([int(''.join(c for c in CSC if c.isdigit())) for CSC in CSCs])
if not args.channel:
    args.channel = CSC_nums
    args.ch_names = [k[0]+'_'+str(CSC_num) for k, CSC_num in zip(channel_keys, CSC_nums)]
print('Channels: ', CSC_nums)

##########
# EVENTS #
##########
event_id = get_event_id()
internal_event_ids = reader.internal_event_ids
IX2event_id = {IX:e_id for IX, (x, e_id) in enumerate(internal_event_ids)}

events_times, events_ids = [], []
for segment in blks[0].segments:
    event_times_mat = segment.events
    for IX, times in enumerate(event_times_mat):
        if IX2event_id[IX] in event_id.values():
            events_times.extend(times) # in SECONDS
            events_ids.extend([IX2event_id[IX]] * len(times))
events_ids = np.asarray(events_ids)    

##############
# SPIKE DATA #
##############
args.sampling_freq_spike = 1000
# CONSTRUCT EVENTS FOR MNE
events_times_spike = np.asarray(events_times) * args.sampling_freq_spike   
num_events = len(events_ids)
events = np.vstack([events_times_spike, np.zeros(num_events,), events_ids]).T.astype(int)
# mne.viz.plot_events(events, sfreq=args.sampling_freq_spike, event_id=event_id)
# PARSE TO SENTENCES (USE FIXATION EVENT NUMBER)
IXs_fixation = (events[:,2] == 8)
events = events[IXs_fixation, :]
# GENERATE EPOCHS
spike_data, unit_names = load_spike_data(args)
info_spike = mne.create_info(unit_names, sfreq=args.sampling_freq_spike)
raw_spike = mne.io.RawArray(spike_data, info_spike)
# raw_spike.plot(duration=2200)
epochs_spike = mne.Epochs(raw_spike, events, metadata=metadata, tmax=10, preload=True, reject=None, baseline=None)
print(epochs_spike)

########
# LFPs #
########

# TODO: FIX FILENAMES OF LFP FIGURES (REMOVE CLUSTER NUMBER AND ADD CORRECT CHANNEL NAME)
if args.lfp:
    print('Loading LFP data...')
    # GET LFP DATA
    channel_data = np.asarray(blks[0].segments[-1].analogsignals[0].T)
    for ch in range(channel_data.shape[0]):
        ch_data = channel_data[ch, :]
        lower, upper = np.percentile(ch_data, 1), np.percentile(ch_data, 99)
        channel_data[ch, :] = np.clip(ch_data, lower, upper)
    info_lfp = mne.create_info(n_channels, sfreq=args.sampling_freq_lfp, ch_types='seeg')
    raw_lfp = mne.io.RawArray(channel_data, info_lfp)
    new_sfreq_lfp = 512
    raw_lfp_resampled = raw_lfp.resample(new_sfreq_lfp)
    args.sampling_freq_lfp = new_sfreq_lfp
    raw_lfp_resampled.notch_filter(np.arange(50, 201, 50), filter_length='auto', phase='zero')
    # raw_lfp_resampled.filter(70,150)
    # CONSTRUCT EVENTS FOR MNE
    events_times_lfp = np.asarray(events_times) * args.sampling_freq_lfp   
    num_events = len(events_ids)
    events = np.vstack([events_times_lfp, np.zeros(num_events,), events_ids]).T.astype(int)
    events = events[events[:,0].argsort()] 
    # PARSE TO SENTENCES (USE FIXATION EVENT NUMBER)
    IXs_fixation = (events[:,2] == 8)
    events = events[IXs_fixation, :]
    epochs_lfp_resampled = mne.Epochs(raw_lfp_resampled, events, metadata=metadata, tmax=10, preload=True, reject=None, baseline=None)
    print(epochs_lfp_resampled)


# PLOT 
for unit, unit_name in enumerate(epochs_spike.ch_names):
    if args.block == 'all_trials':
        # ALL TRIALS
        curr_cherries, curr_TrialInfo = epochs2cherries(epochs_spike, args)
        args.ch_name = curr_cherries[unit]['channel_name']
        args.ch_num = int(curr_cherries[unit]['channel_num']) 
        args.cluster = int(curr_cherries[unit]['class_num'])
        # REORDER BASED ON SENTENCE STRING
        IXs_reorder = np.lexsort((curr_TrialInfo['target_words'], curr_TrialInfo['sentence_strings']))
        for k in curr_TrialInfo.keys():
            curr_TrialInfo[k] = curr_TrialInfo[k][IXs_reorder]
        for u in curr_cherries.keys():
            curr_cherries[u]['trial_data'] = curr_cherries[u]['trial_data'][IXs_reorder]
        fig_spike, axes = generate_rasters(unit, curr_cherries, curr_TrialInfo, [], [], args)
        # SAVE
        path2figures = os.path.join(args.path2figures, f'patient_{args.patient:03d}/session_{args.session}/{args.block}')
        if not os.path.exists(path2figures):
            os.makedirs(path2figures)
        fn_fig = f"{args.patient:03d}_{args.session}_{args.block}_{curr_cherries[unit]['channel_name']}_ch_{int(curr_cherries[unit]['channel_num'])}_cl_{int(curr_cherries[unit]['class_num'])}.png"
        plt.savefig(os.path.join(path2figures, 'raster_' + fn_fig))
        plt.close(fig_spike)
        if args.lfp and unit==0:
            IXs_reorder = np.lexsort((metadata['target_words'].values, metadata['sentence_strings'].values))
            figs_lfp = epochs_lfp_resampled.plot_image(picks=epochs_lfp_resampled.ch_names, order=IXs_reorder)
            [f.savefig(os.path.join(path2figures, f'LFP_{ch_name}_' + fn_fig)) for (f, ch_name) in zip(figs_lfp, epochs_lfp_resampled.ch_names)]
            [plt.close(f) for f in figs_lfp]
        print('Figures saved to: ', os.path.join(path2figures, 'LFP/raster_' + fn_fig))
        
                
    if args.block == 'syntactic':
        for target_word in list(set(TrialInfo['target_words'])):
            args.target_word = target_word 
            epochs_spike_block_target_word = epochs_spike[f'block_types == "{args.block}" and target_words=="{target_word}"']
            sentences = sorted([s for s in list(set(epochs_spike_block_target_word.metadata['sentence_strings'])) if target_word in s])
            if 'ref2target' in list(epochs_spike_block_target_word.metadata):
                # TAKE ONLY TARGET SENTENCE (NOT CONTROL)
                sentence_numbers = sorted(list(set(epochs_spike_block_target_word.metadata.query('ref2target==0')['sentence_numbers'])))
            else:
                sentence_numbers = sorted(list(set(epochs_spike_block_target_word.metadata['sentence_numbers'])))
            # LOOP OVER SENTENCES
            for i_sent, sentence_number in enumerate(sentence_numbers):
                if 'ref2target' in list(epochs_spike_block_target_word.metadata):
                    curr_epochs_spike = epochs_spike_block_target_word[f'sentence_numbers=={sentence_number} or ref2target=={sentence_number}']             
                else:
                    curr_epochs_spike = epochs_spike_block_target_word[f'sentence_numbers=={sentence_number}']             
                sentence = list(set(curr_epochs_spike.metadata.query(f'sentence_numbers=={sentence_number}')['sentence_strings']))
                assert len(sentence) == 1
                sentence = sentence[0]
                if curr_epochs_spike.get_data().shape[0] > 0:
                    curr_cherries, curr_TrialInfo = epochs2cherries(curr_epochs_spike, args)
                    args.ch_name = curr_cherries[unit]['channel_name'] 
                    args.ch_num = int(curr_cherries[unit]['channel_num']) 
                    args.cluster = int(curr_cherries[unit]['class_num']) 
                    fig_spike, axes = generate_rasters(unit, curr_cherries, curr_TrialInfo, sentence, sentence_number, args)
                    # SAVE
                    path2figures = os.path.join(args.path2figures, f'patient_{args.patient:03d}/session_{args.session}/{args.block}')
                    if not os.path.exists(path2figures):
                        os.makedirs(path2figures)
                    fn_fig = f"{args.patient:03d}_{args.session}_{args.block}_{curr_cherries[unit]['channel_name']}_ch_{int(curr_cherries[unit]['channel_num'])}_cl_{int(curr_cherries[unit]['class_num'])}_{target_word}_sentence_{i_sent+1}.png"
                    fig_spike.savefig(os.path.join(path2figures, 'raster_' + fn_fig))
                    plt.close(fig_spike)
                    # LFP
                    if args.lfp and unit==0:
                        epochs_lfp_block_target_word = epochs_lfp_resampled[f'block_types == "{args.block}" and target_words=="{target_word}"']
                        curr_epochs_lfp = epochs_lfp_block_target_word[f'sentence_strings=="{sentence}"']
                        figs_lfp = curr_epochs_lfp.plot_image(picks=curr_epochs_lfp.ch_names)
                        [f.savefig(os.path.join(path2figures, f'LFP_{ch_name}_' + fn_fig)) for (f, ch_name) in zip(figs_lfp, epochs_lfp_resampled.ch_names)]
                        [plt.close(f) for f in figs_lfp]
                    print('Figures saved to: ', os.path.join(path2figures, 'LFP/raster_' + fn_fig))                    
    elif args.block == 'pragmatic':
        for target_word in list(set(TrialInfo['target_words'])):
            args.target_word = target_word 
            curr_epochs_spike = epochs_spike[f'block_types == "{args.block}" and target_words=="{target_word}"']
            if curr_epochs_spike.get_data().shape[0] > 0:    
                curr_cherries, curr_TrialInfo = epochs2cherries(curr_epochs_spike, args)
                args.ch_name = curr_cherries[unit]['channel_name'] 
                args.ch_num = int(curr_cherries[unit]['channel_num']) 
                args.cluster = int(curr_cherries[unit]['class_num']) 
                fig_spike, axes = generate_rasters(int(unit), curr_cherries, curr_TrialInfo, [], [], args)
    
                # SAVE
                path2figures = os.path.join(args.path2figures, f'patient_{args.patient:03d}/session_{args.session}/{args.block}')
                if not os.path.exists(path2figures):
                    os.makedirs(path2figures)
                fn_fig = f"pt{args.patient:03d}_s{args.session}_{args.block}_{curr_cherries[unit]['channel_name']}_ch_{int(curr_cherries[unit]['channel_num'])}_cl_{int(curr_cherries[unit]['class_num'])}_{target_word}.png"
                plt.subplots_adjust(top=0.95, bottom=0.05)
                plt.savefig(os.path.join(path2figures, 'raster_' + fn_fig))
                plt.close(fig_spike)
                # LFP
                if args.lfp and unit==0:
                    curr_epochs_lfp = epochs_lfp_resampled[f'block_types == "{args.block}" and target_words=="{target_word}"']
                    figs_lfp = curr_epochs_lfp.plot_image(picks=curr_epochs_lfp.ch_names)
                    [f.savefig(os.path.join(path2figures, f'LFP_{ch_name}_' + fn_fig)) for (f, ch_name) in zip(figs_lfp, epochs_lfp_resampled.ch_names)]
                    [plt.close(f) for f in figs_lfp]
                print('Figures saved to: ', os.path.join(path2figures, 'LFP/raster_' + fn_fig))
                    
    

    