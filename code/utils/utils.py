import scipy.io as sio
import os
import numpy as np

def get_target_words(TrialInfo, patient, session, block_type):
    target_words = list(set(TrialInfo['target_words']))
    # for target_word in sorted(target_words):



    if patient == 83 and session == 2:
        if block_type == 'pragmatic':
            target_words = {}
            target_words[1] = {}
            target_words[1]['singular'] = 'Wolkenkratzer'
            target_words[1]['plural'] = 'Wolkenkratzer'
            target_words[2] = {}
            target_words[2]['singular'] = 'Balkon'
            target_words[2]['plural'] = 'Balkons'
            target_words[3] = {}
            target_words[3]['singular'] = 'Strand'
            target_words[3]['plural'] = 'Strände'
            target_words[4] = {}
            target_words[4]['singular'] = 'Klavier'
            target_words[4]['plural'] = 'Klaviere'
            target_words[5] = {}
            target_words[5]['singular'] = 'Prüfung'
            target_words[5]['plural'] = 'Prüfungen'
        elif block_type == 'syntactic':
            target_words = {}
            target_words[1] = {}
            target_words[1]['singular'] = 'Wolkenkratzer'
            target_words[1]['plural'] = ''
            target_words[2] = {}
            target_words[2]['singular'] = 'Balkon'
            target_words[2]['plural'] = ''
            target_words[3] = {}
            target_words[3]['singular'] = 'Strand'
            target_words[3]['plural'] = ''
            target_words[4] = {}
            target_words[4]['singular'] = 'Klavier'
            target_words[4]['plural'] = ''
            target_words[5] = {}
            target_words[5]['singular'] = 'Prüfung'
            target_words[5]['plural'] = ''
            target_words[6] = {}
            target_words[6]['singular'] = 'Fahrrad'
            target_words[6]['plural'] = ''
            target_words[7] = {}
            target_words[7]['singular'] = 'Küste'
            target_words[7]['plural'] = ''
            target_words[8] = {}
            target_words[8]['singular'] = 'Politik'
            target_words[8]['plural'] = ''
            target_words[9] = {}
            target_words[9]['singular'] = 'Strand'
            target_words[9]['plural'] = ''

    elif patient == 83 and session == 3:
        if block_type == 'pragmatic':
            target_words = {}
            target_words[1] = {}
            target_words[1]['singular'] = 'Wolkenkratzer'
            target_words[1]['plural'] = 'Wolkenkratzer'
            target_words[2] = {}
            target_words[2]['singular'] = 'Linie'
            target_words[2]['plural'] = 'Linien'
            target_words[3] = {}
            target_words[3]['singular'] = 'Küste'
            target_words[3]['plural'] = 'Küsten'
            target_words[4] = {}
            target_words[4]['singular'] = 'Abend'
            target_words[4]['plural'] = 'Abende'
        elif block_type == 'syntactic':
            target_words = {}
            target_words[1] = {}
            target_words[1]['singular'] = 'Wolkenkratzer'
            target_words[1]['plural'] = ''
            target_words[2] = {}
            target_words[2]['singular'] = 'Linie'
            target_words[2]['plural'] = ''
            target_words[3] = {}
            target_words[3]['singular'] = 'Küste'
            target_words[3]['plural'] = ''
            target_words[4] = {}
            target_words[4]['singular'] = 'Abend'
            target_words[4]['plural'] = ''


    elif patient == 87 and session == 8:
        if block_type == 'pragmatic':
            target_words = {}
        elif block_type == 'syntactic':
            target_words = {}
            target_words[1] = {}
            target_words[1]['singular'] = 'Ablagerung'
            target_words[1]['plural'] = ''
            target_words[2] = {}
            target_words[2]['singular'] = 'Melodie'
            target_words[2]['plural'] = ''
    return target_words


def get_data(path2data, block_type, ending=''):

    TrialInfo = {}

    trial_info = sio.loadmat(os.path.join(path2data, 'trialInfos', 'TrialInfos'))
    trial_order = sio.loadmat(os.path.join(path2data, 'trialInfos', 'TrialOrder'))
    # trial_info = sio.loadmat(os.path.join(path2data, 'trialInfos', f'{block_type}.mat'))
    cherries = sio.loadmat(os.path.join(path2data, 'cherries', f'cherries_{block_type}{ending}.mat'))

    # Info about trials
    sentence_strings = trial_order['TrialOrder']['sentence'][0, 0]
    TrialInfo['sentence_strings'] = np.asarray([l[0] for s in sentence_strings for l in s if l])
    block_types = trial_order['TrialOrder']['sentence_type'][0, 0][0, :]
    TrialInfo['block_types'] = np.asarray([s[0] for s in block_types if s])
    target_word_nums = trial_order['TrialOrder']['responsive_word_Nr'][0, 0][0, :]
    TrialInfo['target_word_nums'] = np.asarray([s[0, 0] for s in target_word_nums if s])
    word_strs = trial_info['trials']['word'][0, 0][0, :]
    TrialInfo['grammatical_role'] = np.asarray([i[0] for i in trial_order['TrialOrder']['pronoun_role'][0, 0][0, :] if i])
    TrialInfo['grammatical_number'] = trial_order['TrialOrder']['grammatical_number'][0, 0][0, :]
    # TrialInfo['pronouns'] = [i[0] for i in trial_order['TrialOrder']['pronoun'][0, 0][0, :]]
    TrialInfo['target_words'] = np.asarray([l[0] for l in trial_order['TrialOrder']['responsive_word'][0, 0][0, :] if l])
    TrialInfo['target_words_number'] = np.squeeze(np.asarray([l[0] for l in trial_order['TrialOrder']['responsive_word_Nr'][0, 0][0, :] if l]))
    if ending == '_sentence':
        TrialInfo['word_strings'] = np.asarray([s[0] if s else '' for s in word_strs])
    else:
        TrialInfo['word_strings'] = np.asarray([s[0] for s in word_strs if s])

        # TrialInfo['grammatical_number'] = [i[0] for i in trial_info['TrialOrder']['grammatical_Nr'][0, 0][0, :]]

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
    # max_t = np.ceil(max([max(sublist) for sublist in spike_events]))
    # min_t = np.floor(min([min(sublist) for sublist in spike_events]))
    # if not max_t:
    #     return np.zeros((num_trials, (int(t_end)-int(t_start))))
    # print(min_t, max_t, t_start, t_end)
    # assert t_end > max_t
    # assert t_start < min_t
    mat = np.zeros((num_trials, (int(t_end)-int(t_start))))
    for i_trial, spike_train in enumerate(spike_events):
        for j_spike in spike_train[0]:
            mat[i_trial, int(j_spike-t_start)] = 1
    return mat

def smooth_with_gaussian(time_series, sfreq=1000, gaussian_width = 50, N=1000):
    # gaussian_width in samples
    # ---------------------
    import math
    from scipy import signal

    norm_factor = np.sqrt(2 * math.pi * gaussian_width ** 2)/sfreq # sanity check: norm_factor = gaussian_window.sum()
    gaussian_window = signal.general_gaussian(M=N, p=1, sig=gaussian_width) # generate gaussian filter
    norm_factor = (gaussian_window/sfreq).sum()
    smoothed_time_series = np.convolve(time_series, gaussian_window/norm_factor, mode="full") # smooth
    smoothed_time_series = smoothed_time_series[int(round(N/2)):-(int(round(N/2))-1)] # trim ends
    return smoothed_time_series


