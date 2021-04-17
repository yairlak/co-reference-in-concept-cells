import scipy.io as sio
import os
import numpy as np



def get_data(path2data, args, ending=''):

    TrialInfo = {}
    
    if args.patient == 83:
        trial_info = sio.loadmat(os.path.join(path2data, 'trialInfos', f'TrialInfos_{args.block}'), uint16_codec='latin1')
        trial_order = sio.loadmat(os.path.join(path2data, 'trialInfos', f'TrialOrder_allInfo_{args.block}'), uint16_codec='latin1')
    elif args.patient in [84, 88, 89]:
        trial_info = sio.loadmat(os.path.join(path2data, 'trialInfos', 'TrialInfos'), uint16_codec='latin1')
        trial_order = sio.loadmat(os.path.join(path2data, 'trialInfos', 'TrialOrder'), uint16_codec='latin1')
    else:
        trial_info = sio.loadmat(os.path.join(path2data, 'trialInfos', 'TrialInfos_allInfo'), uint16_codec='latin1')
        trial_order = sio.loadmat(os.path.join(path2data, 'trialInfos', 'TrialOrder'), uint16_codec='latin1')
    #fn_cherries = os.path.join(path2data, 'cherries', 'Manually clustered', f'cherries_{args.block}{ending}.mat')
    fn_cherries = os.path.join(path2data, 'cherries', f'cherries_{args.block}{ending}.mat')
    print(f'Loading cherries: {fn_cherries}')
    cherries = sio.loadmat(fn_cherries)


    # Info about trials
    #sentence_strings = trial_order['TrialOrder']['sentence'][0, 0]
    #TrialInfo['sentence_strings'] = np.asarray([l[0] for s in sentence_strings for l in s if l])
    #num_stimuli = len(TrialInfo['sentence_strings'])
    #block_types = trial_order['TrialOrder']['sentence_type'][0, 0][0, :]
    #TrialInfo['block_types'] = np.asarray([s[0] for s in block_types if s])
    #target_word_nums = trial_order['TrialOrder']['responsive_word_Nr'][0, 0][0, :]
    #TrialInfo['target_word_nums'] = np.asarray([s[0, 0] for s in target_word_nums if s])
    #word_strs = trial_info['trials']['word'][0, 0][0, :]

    #TrialInfo['target_words'] = np.asarray([l[0] for l in trial_order['TrialOrder']['responsive_word'][0, 0][0, :] if l])
    #if ending == '_sentence':
    #    TrialInfo['word_strings'] = np.asarray([s[0] if s else '' for s in word_strs])   
    #else:
    #    TrialInfo['word_strings'] = np.asarray([s[0] for s in word_strs if s])

    sentence_strings = trial_order['TrialOrder']['sentence'][0, 0]
    TrialInfo['sentence_strings'] = np.asarray([l[0] for s in sentence_strings for l in s])
    sentence_numbers = trial_order['TrialOrder']['sentence_Nr'][0, 0]
    TrialInfo['sentence_numbers'] = sentence_numbers[0]
    num_stimuli = len(TrialInfo['sentence_strings'])
    block_types = trial_order['TrialOrder']['sentence_type'][0, 0][0, :]
    TrialInfo['block_types'] = np.asarray([s[0] for s in block_types])
    target_word_nums = trial_order['TrialOrder']['responsive_word_Nr'][0, 0][0, :]
    TrialInfo['target_word_nums'] = np.asarray([s[0, 0] for s in target_word_nums])
    word_strs = trial_info['trials']['word'][0, 0][0, :]

    TrialInfo['target_words'] = np.asarray([l[0] for l in trial_order['TrialOrder']['responsive_word'][0, 0][0, :]])
    if ending == '_sentence':
        TrialInfo['word_strings'] = np.asarray([s[0] if s else '' for s in word_strs])   
    else:
        TrialInfo['word_strings'] = np.asarray([s[0] for s in word_strs if s])
    

    
    if args.patient not in [83, 84, 86, 88, 89]:    
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
    elif args.patient in [83]:    
        TrialInfo['grammatical_role'] = np.asarray([i[0] for i in trial_order['TrialOrder']['grammatical_role'][0, 0][0, :] if i])
        TrialInfo['grammatical_number'] = np.asarray([i for i in trial_order['TrialOrder']['grammatical_Nr'][0, 0][0, :] if i])
        TrialInfo['grammatical_number'] = np.asarray(['S' if n==1 else 'P' for n in TrialInfo['grammatical_number']])
        
    # Unit activity
    dict_cherries = {}
    for unit_num in range(cherries['cherries'].shape[1]):
        dict_cherries[unit_num + 1] = {}
        dict_cherries[unit_num + 1]['trial_data'] = cherries['cherries'][0, unit_num]['trial'][0, :]
        dict_cherries[unit_num + 1]['class_num'] = cherries['cherries'][0, unit_num]['classno'][0, 0]
        dict_cherries[unit_num + 1]['channel_num'] = cherries['cherries'][0, unit_num]['channr'][0, 0]
        dict_cherries[unit_num + 1]['channel_name'] = cherries['cherries'][0, unit_num]['chnname'][0]
        dict_cherries[unit_num + 1]['site'] = cherries['cherries'][0, unit_num]['site'][0]
        dict_cherries[unit_num + 1]['kind'] = cherries['cherries'][0, unit_num]['kind'][0]


    return TrialInfo, dict_cherries


def get_spike_profile(path2data, ch_num, class_num):
    curr_data = sio.loadmat(os.path.join(path2data, f'times_CSC{ch_num}'))
    cluster_class = curr_data['cluster_class']
    spikes = curr_data['spikes']
    IX_class = (cluster_class[:, 0] == class_num)
    return spikes[IX_class, :]


def event2matrix(spike_events, t_start, t_end):
    '''
    Transform a list of spike trains specified in time into a matrix of ones and zeros
    :param spike_events: list of spikes
    :param t_start: (int) in msec
    :param t_end: (int) in msec
    :return:
    '''
    num_trials = len(spike_events)
    mat = np.zeros((num_trials, (int(t_end)-int(t_start))))
    for i_trial, spike_train in enumerate(spike_events):
        for j_spike in spike_train[0]:
            mat[i_trial, int(j_spike-t_start)] = 1
    return mat


