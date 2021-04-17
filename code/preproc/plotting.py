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

def generate_rasters(unit, cherries, TrialInfo, sentence, sentence_number, args):
    
    # GET INDICES
    IXs_to_target_trials, sentences, labels = get_indices_to_target_trials(TrialInfo, sentence, sentence_number, args)
    
    # GET SPIKE DATA
    spike_trains = []
    for IXs in IXs_to_target_trials:
        spike_trains.append(cherries[unit]['trial_data'][IXs])
    
    # INIT FIGURE
    num_axes = len(IXs_to_target_trials)
    fig, axes = init_fig(args.block, num_axes) 
    
    # PLOT RASTER
    axes = plot_rasters(axes, spike_trains, labels, args)
    
    # COSMETICS
    axes = cosmetics(axes, sentences, args)

    return fig, axes


def get_indices_to_target_trials(TrialInfo, sentence, sentence_number_target, args):
    '''
    
    Parameters
    ----------
    TrialInfo : TYPE
        DESCRIPTION.
    sentence : TYPE
        DESCRIPTION.
    block : TYPE
        DESCRIPTION.

    Returns
    -------
    IXs_sentences : TYPE
        DESCRIPTION.
    sentences : TYPE
        DESCRIPTION.

    '''
    IXs_sentences = []
    labels = []
    if args.block == 'syntactic':
        # Target sentences
        IXs_sentences.append([ix for ix, (s, bt) in enumerate(zip(TrialInfo['sentence_strings'], TrialInfo['block_types'])) if bt == args.block and s == sentence])
        target_sentence = [sentence]
        labels.append('target')
        # Control
        if 'ref2target' in list(TrialInfo):
            IXs_control = [IX for IX, sent_num in enumerate(TrialInfo['ref2target']) if sent_num == sentence_number_target]
            if IXs_control:
                IXs_sentences.append(IXs_control)
                labels.append('control')
                control_sentence = list(set(TrialInfo['sentence_strings'][IXs_control]))
                assert len(control_sentence) == 1
            else:
                control_sentence = []
        else:
            control_sentence = []
        sentences = target_sentence + control_sentence
    elif args.block == 'pragmatic':
        sentences = []
        if args.patient == 84: # HACK FOR PREVIOUS DESIGN (before pt87)
            roles = ['_']
            roles = ['Subj', 'Obj']
        else:
            roles = ['Subj', 'Obj']
        for role in roles: # '--' is the control trials
            for number in ['Singular', 'Plural']:
                # Target
                IXs_target = [IX for IX, (r, n) in enumerate(zip(TrialInfo['grammatical_role'], TrialInfo['grammatical_number'])) if r == role and n == number[0]]
                if IXs_target:
                    IXs_sentences.append(IXs_target)
                    labels.append(f'{role} {number} (target)')
                    target_sentence = list(set(TrialInfo['sentence_strings'][IXs_target]))
                    assert len(target_sentence) == 1
                    if args.patient not in [83, 87]:
                        # control
                        target_sentence_number = [TrialInfo['sentence_numbers'][IX] for IX in IXs_sentences[-1]]
                        target_sentence_number = list(set(target_sentence_number))
                        assert len(target_sentence_number) == 1
                        IXs_control = [IX for IX, n in enumerate(TrialInfo['ref2target']) if n == target_sentence_number[0]]
                        IXs_sentences.append(IXs_control)
                        labels.append(f'{role} {number} (control)')
                        control_sentence = list(set(TrialInfo['sentence_strings'][IXs_control]))
                        assert len(control_sentence) == 1
                    else:
                        control_sentence = []
                    sentences.extend(target_sentence+control_sentence)
                    
        #IXs_sentences = [l for l in IXs_sentences if l] # remove empty sublists
        # Add control trials
        #IXs_control_sentences = [IX for IX, r in enumerate(TrialInfo['grammatical_role']) if r == '--']
        #control_sentences = list(set(TrialInfo['sentence_strings'][IXs_control_sentences]))
        #for sentence in control_sentences:
        #    IXs_sentences.append([IX for IX, s in enumerate(TrialInfo['sentence_strings']) if s==sentence])
                
        
    elif args.block == 'miniscreening':
        IXs_sentences = []
    
    
    elif args.block == 'all_trials':
        sentences = TrialInfo['sentence_strings']
        IXs_sentences = [list(range(len(sentences)))]
        labels = [''] * len(sentences)
        
    return IXs_sentences, sentences, labels


def init_fig(block, num_axes):
    '''
    

    Parameters
    ----------
    block : TYPE
        DESCRIPTION.
    num_axes : TYPE
        DESCRIPTION.

    Returns
    -------
    fig : TYPE
        DESCRIPTION.
    axes : TYPE
        DESCRIPTION.

    '''
    
    height = 2
    vspace = 2
    nrows = (num_axes + 1) * (height+vspace) - vspace # The +1 is for the firing-rate panel
    ncols = 12 # number of rows in subplot grid.
    
    axes = []
    fig, _ = plt.subplots(figsize=(0.7*ncols, 0.7*nrows))
    
    
    for rw in range(0, nrows, (height+vspace)):
        axes.append(plt.subplot2grid((nrows, ncols), (rw, 0), rowspan=height, colspan=10))
    return fig, axes


def plot_rasters(axes, list_spike_trains, labels, args):
    '''
    

    Parameters
    ----------
    axes : TYPE
        DESCRIPTION.
    list_spike_trains : TYPE
        DESCRIPTION.
    args : TYPE
        DESCRIPTION.

    Returns
    -------
    axes : TYPE
        DESCRIPTION.

    '''
    
    # Target/control is distinguished based on line style (ls)
    # Singluar/plural based on color: bluish (blue/cyan) vs redish (red/magenta)
    # Subject/object role based secondary (cyan/magenta for obj) vs primary (blue/red for subj) color
    
    colors, lss = get_color_ls(labels)
                
    for i_ax, (ax, spike_trains) in enumerate(zip(axes[:-1], list_spike_trains)):
        num_trials = spike_trains.shape[0]
        for i_trial, spike_train in enumerate(spike_trains):
            ax.eventplot([0], orientation='horizontal', lineoffsets=i_trial, linelengths=1,
                                                       linewidths=0, colors=colors[i_ax],
                                                      linestyles='solid')  # To make sure another line is generated, add a dummy event (zero width)
            ax.eventplot(spike_train, orientation='horizontal', lineoffsets=i_trial,
                                                       linelengths=args.line_length,
                                                       linewidths=3, colors=colors[i_ax], linestyles='solid',
                                                       label=labels[i_ax])  # Plot events
        
        ax.set_yticks = [num_trials-1]
        ax.set_yticklabels = str(num_trials)
        ax.set_ylabel(labels[i_ax].replace(' ', '\n'), color=colors[i_ax], fontsize=14, rotation=0, horizontalalignment='center', verticalalignment='center', labelpad=20)
    
        # FIRING RATE
        raster_matrix, min_t, max_t = spiketrain2raster(spike_trains)
        firing_rate = gaussian_filter1d(np.mean(raster_matrix, axis=0), float(args.smooth_raster)) # sr = 1000Hz is assumed as sfreq
        axes[-1].plot(firing_rate*1e3, color=colors[i_ax], ls=lss[i_ax], label=labels[i_ax], lw=2)
    # axes[-1].legend()
    axes[-1].set_ylabel('Firing\nrate\n(Hz)', fontsize=14, color='k', rotation=0, horizontalalignment='center', verticalalignment='center', labelpad=20)
    axes[-1].set_ylim([0, 20])
    
    return axes

def get_color_ls(labels):
    colors, lss = [], []
    for i_label, label in enumerate(labels):
        if 'control' in label:
            lss.append('--')
        else:
            lss.append('-')
    
        if 'Subj' in label:
            if 'Singular' in label:
                colors.append('b')
            else:
                colors.append('r')
        else:
            if 'Singular' in label:
                colors.append('c')
            else:
                colors.append('m')
    return colors, lss


def spiketrain2raster(spike_trains):
    '''
    

    Parameters
    ----------
    spike_trains : TYPE
        DESCRIPTION.

    Returns
    -------
    raster_mat : TYPE
        DESCRIPTION.
    min_t : TYPE
        DESCRIPTION.
    max_t : TYPE
        DESCRIPTION.

    '''
    min_t, max_t = 1e6, 1e-6
    for spike_train in spike_trains:
        if spike_train.any():
            if max_t < np.max(spike_train): max_t = np.max(spike_train)
            if min_t < np.min(spike_train): min_t = np.min(spike_train)
    raster_mat = np.zeros((len(spike_trains), 1+int(max_t)))
    for i_trial, spike_train in enumerate(spike_trains):
        IX_spikes = [int(t) for t in spike_train if t>=0] # remove negative spike times (before fixation), since raster_matrix starts at t=0
        raster_mat[i_trial, IX_spikes] = 1
    
    return raster_mat, min_t, max_t


def cosmetics(axes, sentences, args):
    '''
    

    Parameters
    ----------
    axes : TYPE
        DESCRIPTION.
    sentences : TYPE
        DESCRIPTION.
    args : TYPE
        DESCRIPTION.

    Returns
    -------
    axes : TYPE
        DESCRIPTION.

    '''
    
    sentences = [s.replace('/', ' / ') for s in sentences]
    for i_ax, ax in enumerate(axes[:-1]):
        words_to_ticks = ['+'] + sentences[i_ax].split()
        num_words_to_plot = len(words_to_ticks) - 1 # without fixation
        event_times = [0] + list(range(args.fixation_time, args.fixation_time+args.soa*(num_words_to_plot), args.soa))
        event_times = [0, args.fixation_time]
        for w in words_to_ticks[1:-1]:
            if w == '/':
                event_times[-1] += args.isi_pragmatic
            else:
                event_times.append(event_times[-1] + args.soa)
        if '/' in words_to_ticks: words_to_ticks.remove('/')
        
        for i, x in enumerate(event_times):
            if i==0:
                lw=2
                color='k'
            else:
                lw=1
                color='b'
            ax.axvline(x=x, ls='--', color=color, lw=lw)
            axes[-1].axvline(x=x, ls='--', color=color, lw=lw)
        
        ax.set_xticks(event_times)
        ax.set_xlim((-100, event_times[-1]+args.isi_question + 1000))
        ax.set_xticklabels(words_to_ticks, rotation=90, fontsize=10, fontweight='bold')    
    
    # Firing rate plot at the bottom
    axes[-1].set_xticks(event_times)
    axes[-1].set_xticklabels(event_times)
    axes[-1].set_xlim((-100, event_times[-1]+args.isi_question + 1000))
    # axes[-1].set_xticklabels(words_to_ticks, rotation=90, fontsize=12, fontweight='bold')    

    if args.block != 'all_trials':
        title = f'Pt {args.patient:03d} s{args.session}; {args.ch_name} ch-{args.ch_num} unit-{args.cluster}; {args.target_word}'
        plt.suptitle(title, fontsize=16)
    return axes