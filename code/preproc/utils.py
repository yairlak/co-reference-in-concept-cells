#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 14:21:06 2021

@author: yl254115
"""

import os
import scipy.io as sio
import numpy as np
import h5py



def update_paths(args):
    session_folders = [os.path.basename(x[0]) for x in os.walk(os.path.join(args.path2data, 'Results', f'{args.patient:03d}'))]
    session_folder = [x for x in session_folders if x.startswith(f'{args.patient:03d}') and x.endswith(f'pa{args.session}')][0]
    args.path2trialInfo = os.path.join(args.path2data, 'Results', f'{args.patient:03d}', session_folder, 'trialInfos')
    args.path2data = os.path.join(args.path2data, 'Results', f'{args.patient:03d}', session_folder, 'CSCs', 'pa')
    return args


def load_spike_data(args):
    spike_data, unit_names = [], []
    end_time_msec = int(np.ceil((args.tmax - args.tmin)*1e3))
    for channel, ch_name in zip(args.channel, args.ch_names):
        if args.spike_format == 'mat':
            spike_times = sio.loadmat(os.path.join(args.path2data, f'times_CSC{channel}.mat'))
            cluster_numbers = spike_times['cluster_class'][:, 0]
            spike_times_msec = (spike_times['cluster_class'][:, 1] - (args.tmin*1e3)).astype(int)
            
            clusters = list(map(int, set(cluster_numbers)))
            for cluster in clusters:
                if cluster>0:
                    unit_names.append(f'{ch_name}_{cluster:d}')
                    IX_cluster = (cluster_numbers==cluster)
                    spike_times_curr_cluster = spike_times_msec[IX_cluster]
                    spike_data.append(spikeTimes_to_spikeTrains(spike_times_curr_cluster, end_time_msec))    
                
        elif args.spike_format == 'h5':
            spike_times, clusters = load_combinato_sorted_h5(args)
            spike_times_msec = [(s - (args.start_time*1e3)).astype(int) for s in spike_times]
            
            for IX_cluster, cluster in enumerate(clusters):
                if cluster>0:
                    unit_names.append(f'{ch_name}_{cluster:d}')
                    spike_times_curr_cluster = spike_times_msec[IX_cluster]
                    spike_data.append(spikeTimes_to_spikeTrains(spike_times_curr_cluster, end_time_msec))   
            
        
    return np.vstack(spike_data), unit_names


def spikeTimes_to_spikeTrains(spike_times, tmax):
    spike_train = np.zeros((1, tmax))
    for spike_time in spike_times:
        spike_train[0, spike_time-1] = 1
    
    return spike_train

def get_trialInfo(path2trialInfo, args):

    TrialInfo = {}

    if args.patient == 83 and args.session == 3:    
        fn_trialInfo_p = 'TrialInfos_pragmatic.mat'
        trial_info_p = sio.loadmat(os.path.join(path2trialInfo, fn_trialInfo_p), uint16_codec='latin1')
        fn_trialInfo_s = 'TrialInfos_syntactic.mat'
        trial_info_s = sio.loadmat(os.path.join(path2trialInfo, fn_trialInfo_s), uint16_codec='latin1')
        fn_trialOrder_p = 'TrialOrder_allInfo_pragmatic.mat'
        trial_order_p = sio.loadmat(os.path.join(path2trialInfo, fn_trialOrder_p), uint16_codec='latin1')
        fn_trialOrder_s = 'TrialOrder_allInfo_syntactic.mat'
        trial_order_s = sio.loadmat(os.path.join(path2trialInfo, fn_trialOrder_s), uint16_codec='latin1')
        
        trial_order, trial_info = {}, {}
        trial_order['TrialOrder'] = {}
        trial_info['trials'] = {}
        for k in ['sentence', 'sentence_type', 'responsive_word_Nr', 'responsive_word', 'grammatical_role', 'grammatical_Nr', 'sentence_Nr']:
            trial_order['TrialOrder'][k] = {}
            # Remove the last three elements from the syntactic block, 
            # since the pragmatic block begun but triggers weren't sent 
            # so Bita relaunched the pragmatic block seperatley
            trial_order['TrialOrder'][k][0,0] = np.hstack([trial_order_s['TrialOrder'][k][0,0][:, :-3], trial_order_p['TrialOrder'][k][0,0]])
            
        trial_info['trials']['word'] = {}
        trial_info['trials']['word'][0,0] = np.hstack([trial_info_s['trials']['word'][0,0], trial_info_p['trials']['word'][0,0]])
        trial_info['trials']['word'][0,0] = trial_info['trials']['word'][0,0]
        
    elif args.patient in [86, 87]: # with all_info
        fn_trialInfo = 'TrialInfos_allInfo.mat'
        trial_info = sio.loadmat(os.path.join(path2trialInfo, fn_trialInfo), uint16_codec='latin1')
        fn_trialOrder = 'TrialOrder.mat'
        trial_order = sio.loadmat(os.path.join(path2trialInfo, fn_trialOrder), uint16_codec='latin1')
    else:
        fn_trialInfo = 'TrialInfos.mat'
        trial_info = sio.loadmat(os.path.join(path2trialInfo, fn_trialInfo), uint16_codec='latin1')
        fn_trialOrder = 'TrialOrder.mat'
        trial_order = sio.loadmat(os.path.join(path2trialInfo, fn_trialOrder), uint16_codec='latin1')


    # Info about trials
    sentence_strings = trial_order['TrialOrder']['sentence'][0, 0]
    TrialInfo['sentence_strings'] = np.asarray([l[0] for s in sentence_strings for l in s if l])
    num_stimuli = len(TrialInfo['sentence_strings'])
    block_types = trial_order['TrialOrder']['sentence_type'][0, 0][0, :]
    TrialInfo['block_types'] = np.asarray([s[0] for s in block_types if s])
    target_word_nums = trial_order['TrialOrder']['responsive_word_Nr'][0, 0][0, :]
    TrialInfo['target_word_nums'] = np.asarray([s[0, 0] for s in target_word_nums if s])
    # word_strs = trial_info['trials']['word'][0, 0][0, :]
    
    try:
        TrialInfo['target_words'] = np.asarray([l[0] for l in trial_order['TrialOrder']['responsive_word'][0, 0][0, :] if l])
    except:
        print('"responsive_word" field is missing in TrialOrder')    
    # TrialInfo['word_strings'] = np.asarray([s[0] for s in word_strs if s])

        
    if args.patient not in [83, 84, 88]:    
        TrialInfo['grammatical_role'] = np.asarray([i[0] for i in trial_order['TrialOrder']['pronoun_role'][0, 0][0, :] if i])
        TrialInfo['grammatical_number'] = np.asarray([i[0] for i in trial_order['TrialOrder']['grammatical_number'][0, 0][0, :] if i])
        # Get stimulus numbers and target2ref
        sentence_numbers_per_word = trial_info['trials']['sentence_Nr'][0, 0][0, :]
        sentence_numbers_per_word = np.asarray([s[0,0] if s else s for s in sentence_numbers_per_word])
        ref2target_per_word = trial_info['trials']['ref2target'][0, 0][0, :]
    
        sentence_numbers, ref2target = -1*np.ones(num_stimuli), -1*np.ones(num_stimuli)
        cnt = 0
        for sent_num, r2t in zip(sentence_numbers_per_word, ref2target_per_word):
            if sent_num:
                sentence_numbers[cnt] = sent_num
                ref2target[cnt] = r2t
                #print(cnt, sent_num, r2t)
            else:
                cnt += 1
        TrialInfo['sentence_numbers'] = sentence_numbers
        TrialInfo['ref2target'] = ref2target
    elif args.patient == 83:    
        try:
            TrialInfo['grammatical_role'] = np.asarray([i[0] for i in trial_order['TrialOrder']['grammatical_role'][0, 0][0, :] if i])
            TrialInfo['grammatical_number'] = np.asarray([i for i in trial_order['TrialOrder']['grammatical_Nr'][0, 0][0, :] if i])
            TrialInfo['grammatical_number'] = np.asarray(['S' if n==1 else 'P' for n in TrialInfo['grammatical_number']])
        except:
            print('"grammatical_role" or "grammatical_Nr" fields are missing')
        TrialInfo['sentence_numbers'] = trial_order['TrialOrder']['sentence_Nr'][0, 0][0, :]
    elif args.patient == 84:
        TrialInfo['grammatical_role'] = np.asarray([i[0] for i in trial_order['TrialOrder']['pronoun_role'][0, 0][0, :] if i])
        TrialInfo['grammatical_number'] = np.asarray([i[0] for i in trial_order['TrialOrder']['grammatical_number'][0, 0][0, :] if i])
        
        sentence_numbers_per_word = trial_info['trials']['sentence_Nr'][0, 0][0, :]
        sentence_numbers_per_word = np.asarray([s[0,0] if s else s for s in sentence_numbers_per_word])
        sentence_numbers = -1*np.ones(num_stimuli)
        cnt = 0
        for sent_num in sentence_numbers_per_word:
            if sent_num:
                sentence_numbers[cnt] = sent_num
            else:
                cnt += 1
        TrialInfo['sentence_numbers'] = sentence_numbers
    
    # IXs_block = (TrialInfo['block_types'] == block_name)
    # for k in TrialInfo.keys():
    #     TrialInfo[k] = TrialInfo[k][IXs_block]

    return TrialInfo


def get_event_id():
    event_id = {}
    # event_id['start_of_paradigm'] = 1
    event_id['start_of_run'] = 4
    event_id['fixation_cross'] = 8
    event_id['word_onset'] = 16
    event_id['word_offset'] = 32
    event_id['question_onset'] = 64
    event_id['response_onset'] = 128
    # event_id['block_begin'] = 256
    return event_id



def load_combinato_sorted_h5(args):    
    spike_times = []; cluster_numbers = []

    filename = os.path.join(args.path2data, f'CSC{args.channel}', f'data_CSC{args.channel}.h5')
    f_all_spikes = h5py.File(filename, 'r')

    for sign in ['pos']:
        filename_sorted = os.path.join(args.path2data, f'CSC{args.channel}', 'sort_pos_bit', 'sort_cat.h5')
        f_sort_cat = h5py.File(filename_sorted, 'r')

        classes =  np.asarray(f_sort_cat['classes'])
        index = np.asarray(f_sort_cat['index'])
        # matches = np.asarray(f_sort_cat['matches'])
        groups = np.asarray(f_sort_cat['groups'])
        group_numbers = set([g[1] for g in groups])
        types = np.asarray(f_sort_cat['types']) # -1: artifact, 0: unassigned, 1: MU, 2: SU

        # For each group, generate a list with all spike times and append to spike_times
        for g in list(group_numbers):
            IXs = []
            type_of_curr_group = [t_ for (g_, t_) in types if g_ == g]
            if len(type_of_curr_group) == 1:
                type_of_curr_group = type_of_curr_group[0]
            else:
                raise ('issue with types: more than one group assigned to a type')
            if type_of_curr_group>0: # ignore artifact and unassigned groups

                # Loop over all spikes
                for i, c in enumerate(classes):
                    # check if current cluster in group
                    g_of_curr_cluster = [g_ for (c_, g_) in groups if c_ == c]
                    if len(g_of_curr_cluster) == 1:
                        g_of_curr_cluster = g_of_curr_cluster[0]
                    else:
                        raise('issue with groups: more than one group assigned to a cluster')
                    # if curr spike is in a cluster of the current group
                    if g_of_curr_cluster == g:
                        curr_IX = index[i]
                        IXs.append(curr_IX)

                curr_spike_times = np.asarray(f_all_spikes[sign]['times'])[IXs]
                spike_times.append(curr_spike_times)
                cluster_numbers.append(g)
    
    return spike_times, cluster_numbers


def epochs2cherries(epochs, args):
    cherries = {}
    unit_names = epochs.ch_names
    data = epochs._data
    for cluster, unit_name in zip(range(data.shape[1]), unit_names):
        cherries[cluster] = {}
        list_trials_spikes = []
        for trial in range(data[:, cluster, :].shape[0]):
            curr_trial_spike_train = data[trial, cluster, :]
            spike_times_msec = np.where(curr_trial_spike_train == 1)[0]
            list_trials_spikes.append(spike_times_msec)
        cherries[cluster]['trial_data'] = np.asarray(list_trials_spikes)
        cherries[cluster]['class_num'] = unit_name.split('_')[2]
        cherries[cluster]['channel_num'] = unit_name.split('_')[1]
        cherries[cluster]['channel_name'] = unit_name.split('_')[0]
        cherries[cluster]['site'] = '?'
        cherries[cluster]['kind'] = '?'
    
    TrialInfo = epochs.metadata.to_dict('list')
    for k in TrialInfo.keys():
        TrialInfo[k] = np.asarray(TrialInfo[k])
    
    return cherries, TrialInfo