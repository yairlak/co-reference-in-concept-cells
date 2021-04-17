#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 13:29:52 2021

@author: yl254115

"""
import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1 import make_axes_locatable
import utils
from scipy.ndimage import gaussian_filter1d

def generate_rasters(unit, cherries, TrialInfo, target_word_dict, sentence, args):
    
    # GET INDICIES
    IXs_to_target_trials = get_indices_to_target_trials(TrialInfo, target_word_dict, sentence, args.block)
    
    # GET SPIKE DATA
    spike_trains = []
    for IXs in IXs_to_target_trials:
        spike_trains.append(cherries[unit]['trial_data'][IXs])
    
    # INIT FIGURE
    fig, axes = init_fig(args.block) 
    
    # PLOT RASTER
    axes = plot_rasters(axes, spike_trains, args)
    
    
    # grammatical_numbers = {'pragmatic': ['singular', 'plural'], 'syntactic': ['singular']}[args.block]
    
    
    
    # sentence_example = {}
    # num_words_to_plot = len(sentence.split())
    # t_min, t_max = 0, (1+num_words_to_plot)*args.soa
    
    
        
            
    
    
    # print(f'unit {unit}, sentence {sentence}')
    
    # num_trials = len(trial_spike_trains)
            
    # cnt = 0
    # color = 'k'
    # for i_trial, spike_train in enumerate(trial_spike_trains):
    #     ax.eventplot([0], orientation='horizontal', lineoffsets=i_trial, linelengths=1,
    #                                 linewidths=0, colors=color, linestyles='solid') # To make sure another line is generated, add a dummy event (zero width)
    #     ax.eventplot(spike_train, orientation='horizontal', lineoffsets=cnt, linelengths=args.line_length,
    #                                 linewidths=3, colors=color, linestyles='solid') # Plot events
    #     # target_word = TrialInfo['target_word_nums'][IX_sentences[0]]
    #     # target_word = {1:'Wolkenkratzer', 2:'Abend', 3:'Küste', 4:'Linie', 5:'?'}[target_word]
    #     # axs[i_sentence].set_title(target_word, fontsize=24)
    #     cnt += 1

    # COSMETICS

    # if i_sentence % num_cols == 0:
    # ax.set_ylabel('Trial Number', fontsize=24)

    
    # event_times = [0, args.fixation_time] + list(range(args.fixation_time+args.soa, args.fixation_time+args.soa*(num_words_to_plot+1), args.soa))
    # for i, x in enumerate(event_times):
    #     if i==0:
    #         lw=2
    #         color='k'
    #     else:
    #         lw=1
    #         color='b'
    #     ax.axvline(x=x, ls='--', color=color, lw=lw)
    # ax.set(yticks = range(cnt), yticklabels = range(1, cnt+1))
    # ax.tick_params(axis='both', which='major', labelsize=12)
    # ax.tick_params(axis='both', which='minor', labelsize=12)

    # sentence_example = sentence
    # words_to_ticks = [' ', '+'] + sentence_example.split()
    # ax.set(xticks=event_times, xlim=(t_min, t_max))
    # ax.set_xticklabels(words_to_ticks, rotation=90, fontsize=22, fontweight='bold')
    
    # # FIRING RATE
    # raster_matrix = spiketrain2raster(trial_spike_trains)
    # raster_matrix_smoothed = raster_matrix.copy()
    # for t in range(num_trials):
    #     raster_matrix_smoothed[t, :] = gaussian_filter1d(raster_matrix[t, :], float(args.smooth_raster)*1000) # 1000Hz is assumed as sfreq
    # axPsth.plot(np.mean(raster_matrix_smoothed, axis=0))
    
    ##########################
    # COSMETICS AND SAVE FIG #
    ##########################
    # path2figures = f'../figures/rasters/patient_{patient}/session_{session}/{block_type}'
    # fn_fig = f"raster_{patient}_{session}_{block_type}_{cherries[unit]['channel_name']}_{cherries[unit]['channel_num']}_{cherries[unit]['class_num']}_sentence.png"
    # if not os.path.exists(path2figures):
    #     os.makedirs(path2figures)
    # fig.suptitle(
    #     f"Channel #{cherries[unit]['channel_num']} ({cherries[unit]['channel_name']}), class {cherries[unit]['class_num']}; {cherries[unit]['kind']}",
    #     fontsize=26)
    # fig.text(0.5, 0.04, 'Time (ms)', ha='center', fontsize=24)

    # plt.tight_layout()
    # plt.subplots_adjust(top=0.9, left=0.05, right=0.95, hspace=0.5)
    # # plt.savefig(os.path.join(path2figures, fn_fig))
    # plt.close(fig)
    
    
    
    # fig.suptitle(
        # f"Channel #{cherries[unit]['channel_num']} ({cherries[unit]['channel_name']}), class {cherries[unit]['class_num']}; {cherries[unit]['kind']}",
        # fontsize=26)
    # fig.text(0.5, 0.04, 'Time (ms)', ha='center', fontsize=24)
    # plt.subplots_adjust(hspace=0.8)
    
    # elif args.block == 'pragmatic':
        # PREPARE FIGURE
        
        # fig, _ = plt.subplots(figsize=(10, 10))
        # nrows = 20; 
        # ncols = 10 # number of rows in subplot grid.
        # axSubj = plt.subplot2grid((nrows, ncols), (0, 0), rowspan=8, colspan=10)
        # axObj = plt.subplot2grid((nrows, ncols), (8, 0), rowspan=8, colspan=10)
        # axPsth = plt.subplot2grid((nrows, ncols), (16, 0), rowspan=4, colspan=10)

        ######################
        # PLOT SPIKE PROFILE #
        ######################
        # if args.plot_spike_profile:
        #     axs[-1] = utils.get_spike_profile(args.path2data, cherries[unit]['channel_num'], cherries[unit]['class_num'])
        
        
        ################
        # PLOT RASTERS #
        ################
        # for i_word, word_num in enumerate(target_words.keys()):
        #     # plt.subplots(2, num_cols, figsize=(30, 15))
        #     for role in ['Subj', 'Obj']:
        #         # pronouns = []
        #         cnt=0
        #         for i_gn, grammatical_number in enumerate(grammatical_numbers):
        #             word_str = target_words[word_num][grammatical_number]
        #             print(word_str, word_num, grammatical_number)
        #             # FIND INDICES OF CURRENT WORD
        #             IX_word_str = [i for i, w in enumerate(TrialInfo['word_strings']) if w == word_str]
        #             if word_str == 'Wolkenkratzer':  # find grammatical number based on previous article (Der/Die).
        #                 article = {'singular': 'Der', 'plural': 'Die'}[grammatical_number]
        #                 IX_word_str = [i for i in IX_word_str if TrialInfo['word_strings'][i - 1] == article]

                    
        #             IX_sentence_end = [i for (i, w) in enumerate(TrialInfo['word_strings']) if not w]
        #             # For each appearance of target work in 'word_string',
        #             # gives an intex to sentence number that contains it:
        #             IX_sentences = [] 
        #             for ix_word in IX_word_str:
        #                 x = [1 if ix_word > ix_end else 0 for ix_end in IX_sentence_end]
        #                 IX_sentences.append(sum(x))

        #             # filter sentences from the other grammatical role
        #             IX_sentences = [IX for i, IX in enumerate(IX_sentences) if TrialInfo['grammatical_role'][i] == role]
        #             # pronouns.extend([TrialInfo['pronouns'][i] for i in IX_sentences])
        #             trial_spike_trains = cherries[unit]['trial_data'][IX_sentences]
        #             color = {'singular': 'b', 'plural': 'r'}[grammatical_number]
        #             for i_trial, spike_train in enumerate(trial_spike_trains):
        #                 if role == 'Subj':
        #                     axSubj.eventplot([0], orientation='horizontal', lineoffsets=i_trial, linelengths=1,
        #                                           linewidths=0, colors=color,
        #                                           linestyles='solid')  # To make sure another line is generated, add a dummy event (zero width)
        #                     axSubj.eventplot(spike_train, orientation='horizontal', lineoffsets=cnt,
        #                                           linelengths=linelength,
        #                                           linewidths=3, colors=color, linestyles='solid',
        #                                           label=grammatical_number)  # Plot events
        #                     if i_gn == 0: axSubj.set_title(f'{word_str} ({role})', fontsize=24)
        #                     cnt += 1


        #                 else:
        #                     axObj.eventplot([0], orientation='horizontal', lineoffsets=i_trial,
        #                                           linelengths=1,
        #                                           linewidths=0, colors=color,
        #                                           linestyles='solid')  # To make sure another line is generated, add a dummy event (zero width)
        #                     axObj.eventplot(spike_train, orientation='horizontal', lineoffsets=cnt,
        #                                           linelengths=linelength,
        #                                           linewidths=3, colors=color, linestyles='solid',
        #                                           label=grammatical_number)  # Plot events
        #                     if i_gn == 0: axObj.set_title(f'{word_str} ({role})', fontsize=24)
        #                     cnt += 1

        #             # spike_mat = utils.event2matrix(cherries[unit]['trial_data'][IX_sentences], 0, 1e4)
        #             # if spike_mat is not None:
        #             #     mean_spikes = utils.smooth_with_gaussian(spike_mat, gaussian_width=100)

        #             sentence_example[role] = TrialInfo['sentence_strings'][IX_sentences[0]].split('/')
        #         ax = {'Subj': axSubj, 'Obj': axObj}[role]
        #         ax.set(yticks=range(len(pronouns)), yticklabels=pronouns)

        #     # axPSTH = divider.append_axes(position='bottom', size="100%", pad=1) #, sharex=axObj)
        #     # axPSTH.plot(mean_spikes)
        #     # COSMETICS
        #     max_ed_sentence2 = 0
        #     for role in ['Obj', 'Subj']:
        #         num_words_sentence1 = len(sentence_example[role][0].split())
        #         num_words_sentence2 = len(sentence_example[role][1].split())
        #         num_words_to_plot = 1 + num_words_sentence1 + num_words_sentence2
        #         st_sentence1 = fixation_time
        #         ed_sentence1 = fixation_time + SOA * num_words_sentence1
        #         st_sentence2 = fixation_time + num_words_sentence1 * SOA + ISI
        #         ed_sentence2 = fixation_time + num_words_sentence1 * SOA + ISI + SOA * num_words_sentence2
        #         if ed_sentence2 >  max_ed_sentence2:
        #             max_ed_sentence2 = ed_sentence2


        #     for role in ['Obj', 'Subj']:
        #         ax = {'Subj':axSubj, 'Obj':axObj}[role]
        #         num_words_sentence1 = len(sentence_example[role][0].split())
        #         num_words_sentence2 = len(sentence_example[role][1].split())
        #         num_words_to_plot = 1 + num_words_sentence1 + num_words_sentence2
        #         st_sentence1 = fixation_time
        #         ed_sentence1 = fixation_time + SOA * num_words_sentence1
        #         st_sentence2 = fixation_time + num_words_sentence1 * SOA + ISI
        #         ed_sentence2 = fixation_time + num_words_sentence1 * SOA + ISI + SOA * num_words_sentence2
        #         event_times = [0] + list(range(st_sentence1, ed_sentence1, SOA)) + list(
        #             range(st_sentence2, ed_sentence2, SOA))

        #         event_labels = ['+'] + sentence_example[role][0].split() + sentence_example[role][1].split()
        #         # if i_word % num_cols == 0:
        #         #     ax.set_ylabel('Trial Number', fontsize=24)

        #         for i, x in enumerate(event_times):
        #             if i == 1:
        #                 lw = 2
        #                 color = 'k'
        #             elif i == 1+num_words_sentence1:
        #                 lw = 2
        #                 color = 'k'
        #             else:
        #                 lw = 1
        #                 color = 'b'
        #             ax.axvline(x=x, ls='--', color=color, lw=lw)
        #         ax.tick_params(axis='both', which='major', labelsize=12)
        #         ax.tick_params(axis='both', which='minor', labelsize=12)

        #         ax.set_xlim((0, max_ed_sentence2))
        #         ax.set_xticks(event_times)
        #         ax.set_xticklabels(event_labels, rotation=45, ha="right")
                # print(event_times, event_labels)
    return


def get_indices_to_target_trials(TrialInfo, target_word_dict, sentence, block):
    
    IXs_target_sentences = []
    if block == 'syntactic':
        IXs_target_sentences.append([ix for ix, (s, bt) in enumerate(zip(TrialInfo['sentence_strings'], TrialInfo['block_types'])) if bt == block and s == sentence])
    elif block == 'pragmatic':
        IXs_target_sentences = []
        
        for role in ['Subj', 'Obj']:
            for grammatical_number in ['singular', 'plural']:
                word_str = target_word_dict[grammatical_number]
                # print(word_str, word_num, grammatical_number)
                # FIND INDICES OF CURRENT WORD
                IX_word_str = [i for i, w in enumerate(TrialInfo['word_strings']) if w == word_str]
                
                IX_sentence_end = [i for (i, w) in enumerate(TrialInfo['word_strings']) if not w]
                # For each appearance of target work in 'word_string',
                # gives an intex to sentence number that contains it:
                IXs_target_sentences = [] 
                for ix_word in IX_word_str:
                    x = [1 if ix_word > ix_end else 0 for ix_end in IX_sentence_end]
                    IXs_target_sentences.append(sum(x))

                # Append sentences from the current grammatical role
                IXs_target_sentences.append([IX for i, IX in enumerate(IXs_target_sentences) if TrialInfo['grammatical_role'][i] == role])
            
                # if word_str == 'Wolkenkratzer':  # find grammatical number based on previous article (Der/Die).
                #     article = {'singular': 'Der', 'plural': 'Die'}[grammatical_number]
                #     IX_word_str = [i for i in IX_word_str if TrialInfo['word_strings'][i - 1] == article]

                
        
    elif block == 'miniscreening':
        IXs_target_sentences = []
    
    return IXs_target_sentences


def init_fig(block):
    '''
    

    Parameters
    ----------
    block : string
        syntactic/pragmatic/miniscreening.

    Returns
    -------
    fig : pyplot figure handle
        DESCRIPTION.
    axs : list of pyplot axis handles
        corresponding axes (last element corresponds to the PSTH plot)

    '''
    axs = []
    fig, _ = plt.subplots(figsize=(10, 10))
    nrows = 12 if block == 'syntactic' else 20
    ncols = 10 # number of rows in subplot grid.
    axs.append(plt.subplot2grid((nrows, ncols), (0, 0), rowspan=8, colspan=10))
    if block == 'pragamatic': # add another subplot for syntactic role
        axs.append(plt.subplot2grid((nrows, ncols), (8, 0), rowspan=8, colspan=10))
    axs.append(plt.subplot2grid((nrows, ncols), (nrows-4, 0), rowspan=4, colspan=10))
    return fig, axs


def spiketrain2raster(spike_trains):
    
    max_t = 0
    for spike_train in spike_trains:
        if max_t < np.max(spike_train):
            max_t = np.max(spike_train)
    raster_mat = np.zeros((len(spike_trains), 1+int(max_t)))
    for i_trial, spike_train in enumerate(spike_trains):
        raster_mat[i_trial, list(map(int, spike_train[0]))] = 1
    
    return raster_mat


def plot_rasters(axs, spike_trains, block, args):
    
    color = 'b'
    label = 'X'
    for ax, spikes in zip(axs, spike_trains):
        cnt = 0
        for i_trial, spike_train in enumerate(trial_spike_trains):
            ax.eventplot([0], orientation='horizontal', lineoffsets=i_trial, linelengths=1,
                                                       linewidths=0, colors=color,
                                                      linestyles='solid')  # To make sure another line is generated, add a dummy event (zero width)
            ax.eventplot(spike_train, orientation='horizontal', lineoffsets=cnt,
                                                       linelengths=args.linelength,
                                                       linewidths=3, colors=color, linestyles='solid',
                                                       label=label)  # Plot events
            # ax.set_title(f'{word_str} ({role})', fontsize=24)
            cnt += 1
    
    return

def plot_spike_profile():
    ######################
    # PLOT SPIKE PROFILE #
    ######################
    spikes = utils.get_spike_profile(path2data, cherries[unit]['channel_num'], cherries[unit]['class_num'])
    spikes_mean = np.mean(spikes, axis=0)
    spikes_std = np.std(spikes, axis=0)
    axs[-1].plot(spikes_mean, color='k', lw=3)
    axs[-1].fill_between(range(len(spikes_mean)), spikes_mean - spikes_std, spikes_mean + spikes_std,
                          color='m', alpha=0.2)
    axs[-1].set_ylim([-50, 100])
    axs[-1].set_xlabel('Time Samples', fontsize=24)
    axs[-1].set_ylabel('uV', fontsize=24)
    axs[-1].set_title('Spike Profile', fontsize=24)
    if len(axs) > len(sentences) + 1:
        extra_axs = (len(axs) - len(sentences) - 1)
        for i in range(extra_axs):
            axs[-2 - i].set_axis_off()
    return ax


# if args.block == 'syntactic':
    #     num_words_to_plot = len(sentences[0].split())
    # elif args.block == 'pragmatic':
    #     num_words_to_plot = 10
    
    # event_times = [0] + list(range(args.fixation_time, args.fixation_time+args.soa*(num_words_to_plot), args.soa))
    
    
    # ADD VERTICAL LINES
    # for ax in axes[:-1]:
    #     for i, x in enumerate(event_times):
    #         if i==0:
    #             lw=2
    #             color='k'
    #         else:
    #             lw=1
    #             color='b'
    #         ax.axvline(x=x, ls='--', color=color, lw=lw)
    #     ax.tick_params(axis='both', which='major', labelsize=12)
    #     ax.tick_params(axis='both', which='minor', labelsize=12)

    # X-LABELS & LIMS
    # words_to_ticks = ['+'] + sentences[0].split()
    # print(words_to_ticks, event_times)
    
    
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
    
    elif patient == 84 and session == 1:
        if block_type == 'pragmatic':
            target_words = {}
            target_words[1] = {}
            target_words[1]['singular'] = 'Kaffee'
            target_words[1]['plural'] = 'Kaffee'
        elif block_type == 'syntactic':
            target_words = {}
            target_words[1] = {}
            target_words[1]['singular'] = 'Wolkenkratzer'
            target_words[1]['plural'] = ''
            


    elif patient == 87 and session == 1:
        if block_type == 'pragmatic':
            target_words = {}
            target_words[1] = {}
            target_words[1]['singular'] = 'Kaffee'
            target_words[1]['plural'] = 'Kaffeen'
        elif block_type == 'syntactic':
            target_words = {}
            target_words[1] = {}
            target_words[1]['singular'] = 'Kaffee'
            target_words[1]['plural'] = ''
            target_words[2] = {}
            target_words[2]['singular'] = 'Spaghetti'
            target_words[2]['plural'] = ''

    elif patient == 87 and session == 8:
        if block_type == 'pragmatic':
            target_words = {}
            target_words[1] = {}
            target_words[1]['singular'] = 'Ablagerung'
            target_words[1]['plural'] = 'Ablagerungen'
        elif block_type == 'syntactic':
            target_words = {}
            target_words[1] = {}
            target_words[1]['singular'] = 'Ablagerung'
            target_words[1]['plural'] = ''
            target_words[2] = {}
            target_words[2]['singular'] = 'Melodie'
            target_words[2]['plural'] = ''
    return target_words
