import os
import numpy as np
import matplotlib.pyplot as plt
import utils

patient = 83
# session = 2
# block_type = 'syntactic' # syntactic/pragmatic
SOA = 600 # in msec

for session in [3]:
    # for block_type in ['pragmatic', 'syntactic']:
    for block_type in ['pragmatic']:
        #############
        # LOAD DATA #
        #############
        target_words = utils.get_target_words(patient, session, block_type)
        path2data = f'../data/patient_{patient}_s{session}'
        TrialInfo, cherries = utils.get_data(path2data, block_type)
        assert len(TrialInfo['word_strings']) == cherries[1]['trial_data'].shape[0] # check that there's unit activity for each word

        ####################
        # GENERATE RASTERS #
        ####################
        num_target_words = len(target_words)
        times = np.arange(-1000, 2000)
        linelength = 0.3

        for unit in cherries.keys():
            # PREPARE FIGURE
            num_cols = int(np.ceil((num_target_words+1)/2))
            fig, axs = plt.subplots(2, num_cols, figsize=(30, 10))
            axs = axs.flat

            ######################
            # PLOT SPIKE PROFILE #
            ######################
            spikes = utils.get_spike_profile(path2data, cherries[unit]['channel_num'], cherries[unit]['class_num'])
            spikes_mean = np.mean(spikes, axis=0)
            spikes_std = np.std(spikes, axis=0)
            axs[-1].plot(spikes_mean, color='k', lw=3)
            axs[-1].fill_between(range(len(spikes_mean)), spikes_mean - spikes_std, spikes_mean + spikes_std, color='m', alpha=0.2)
            axs[-1].set_ylim([-50, 100])
            axs[-1].set_xlabel('Time Samples', fontsize=24)
            axs[-1].set_ylabel('uV', fontsize=24)
            axs[-1].set_title('Spike Profile', fontsize=24)
            if len(axs)> num_target_words+1:
                extra_axs = (len(axs)-num_target_words-1)
                for i in range(extra_axs):
                    axs[-2-i].set_axis_off()

            ###############
            # PLOT RASTER #
            ###############
            for i_word, word_num in enumerate(target_words.keys()):
                cnt = 0
                grammatical_numbers = {'pragmatic':['singular', 'plural'], 'syntactic':['singular']}[block_type]
                for i_gn, grammatical_number in enumerate(grammatical_numbers):
                    word_str = target_words[word_num][grammatical_number]
                    print(word_str, word_num, grammatical_number)
                    # FIND INDICES OF CURRENT WORD
                    IX_word_str = [i for (i, w) in enumerate(TrialInfo['word_strings']) if w == word_str]
                    if word_str == 'Wolkenkratzer': # find grammatical number based on previous article (Der/Die).
                        article = {'singular':'Der', 'plural':'Die'}[grammatical_number]
                        IX_word_str = [i for i in IX_word_str if TrialInfo['word_strings'][i-1] == article]

                    # DRAW SPIKE TRAINS
                    trial_spike_trains = cherries[unit]['trial_data'][IX_word_str]
                    color = {'singular':'b', 'plural':'r'}[grammatical_number]
                    for i_trial, spike_train in enumerate(trial_spike_trains):
                        axs[i_word].eventplot([0], orientation='horizontal', lineoffsets=i_trial, linelengths=1,
                                                    linewidths=0, colors=color, linestyles='solid') # To make sure another line is generated, add a dummy event (zero width)
                        axs[i_word].eventplot(spike_train, orientation='horizontal', lineoffsets=cnt, linelengths=linelength,
                                                    linewidths=3, colors=color, linestyles='solid', label=grammatical_number) # Plot events
                        if i_gn==0: axs[i_word].set_title(word_str, fontsize=24)
                        cnt += 1

                # COSMETICS
                if i_word % num_cols == 0:
                    axs[i_word].set_ylabel('Trial Number', fontsize=24)
                for x in range(0, SOA*3, SOA):
                    lw=2 if x==0 else 1
                    axs[i_word].axvline(x=x, ls='--', color='k', lw=lw)
                axs[i_word].set(yticks = range(cnt), yticklabels = range(1, cnt+1), xlim=(-1000, 2000))
                axs[i_word].tick_params(axis='both', which='major', labelsize=12)
                axs[i_word].tick_params(axis='both', which='minor', labelsize=12)

            ##########################
            # COSMETICS AND SAVE FIG #
            ##########################
            path2figures = f'../figures/rasters/patient_{patient}/session_{session}/{block_type}'
            fn_fig = f"raster_{patient}_{session}_{block_type}_{cherries[unit]['channel_name']}_{cherries[unit]['channel_num']}_{cherries[unit]['class_num']}.png"
            if not os.path.exists(path2figures):
                os.makedirs(path2figures)
            fig.suptitle(
                f"Channel #{cherries[unit]['channel_num']} ({cherries[unit]['channel_name']}), class {cherries[unit]['class_num']}; {cherries[unit]['kind']}",
                fontsize=26)
            fig.text(0.5, 0.04, 'Time (ms)', ha='center', fontsize=24)
            plt.savefig(os.path.join(path2figures, fn_fig))
            plt.close(fig)
