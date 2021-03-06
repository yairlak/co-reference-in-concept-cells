import os, mne
import numpy as np
import utils, utils_mne
from tqdm import tqdm


gaussian_width = 25 # in samples
sfreq = 1000 # [Hz]
patient = 83
session = 3
block_type = 'pragmatic'

#############
# LOAD DATA #
#############
path2data = f'../data/patient_{patient}_s{session}'
target_words = utils.get_target_words(patient, session, block_type)
TrialInfo, cherries = utils.get_data(path2data, block_type, '_sentence')
metadata = utils_mne.create_metadata(TrialInfo)
all_channels_data = []
num_units = len(cherries)


###############
# SMOOTH DATA #
###############
print(f'smoothing spike data')
for unit in tqdm(range(1, num_units+1)):
    spike_mat = utils.event2matrix(cherries[unit]['trial_data'], 0, 1e4)
    smoothed_spike_mat = []
    for spike_train in spike_mat:
        smoothed_spike_train = utils.smooth_with_gaussian(spike_train, sfreq=sfreq, gaussian_width=gaussian_width)
        smoothed_spike_mat.append(smoothed_spike_train)
    all_channels_data.append(smoothed_spike_mat)
all_channels_data = np.asarray(all_channels_data).swapaxes(0, 1)

##################
# CONVERT TO MNE #
##################
print(all_channels_data.shape)
info = mne.create_info(ch_names=['unit '+str(u+1) for u in range(num_units)], sfreq=1000, ch_types=['misc' for _ in range(num_units)])
epochs = mne.EpochsArray(all_channels_data, info, metadata=metadata)

########
# SAVE #
########
fname = f'patient_{patient}_s{session}_smoothed_{1000*gaussian_width/sfreq}_msec-epo.fif'
epochs.save(os.path.join(path2data, fname))