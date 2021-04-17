import os, mne
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn import metrics
import comparisons

print(mne.__version__)

gaussian_width = 25 # in samples
sfreq = 1000 # [Hz]
patient = 83
session = 3
block_type = 'pragmatic'
comparison = 'word_identity'


#############
# LOAD DATA #
#############
path2data = f'../data/patient_{patient}_s{session}'
fname = f'patient_{patient}_s{session}_smoothed_{1000*gaussian_width/sfreq}_msec-epo.fif'
epochs = mne.read_epochs(os.path.join(path2data, fname))

###############
# GET QUERIES #
###############
if isinstance(comparison, str):
    for c, d in comparisons.comparison_list().items():
        if d['name'] == comparison:
            comparison = d
            break
elif isinstance(comparison, int):
    comparison = comparisons.comparison_list()[comparison]
else:
    raise('Wrong type of comparison (only string or integer are supported)')
print(comparison)

if isinstance(comparison['train_queries'], str):
    metadata_header = comparison['train_queries']
    query_vals = set(epochs.metadata[metadata_header])
    queries_train = []
    for query_val in query_vals:
        query = f'{metadata_header}=={query_val}'
        queries_train.append(query)

##################
# classification #
##################

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

if len(queries_train)>2:
    clf = make_pipeline(OneVsRestClassifier(LogisticRegression(solver='lbfgs')))
    scoring = 'roc_auc_ovo'
else:
    clf = make_pipeline(LogisticRegression(solver='lbfgs'))
    scoring = 'roc_auc'
time_gen = mne.decoding.GeneralizingEstimator(clf, n_jobs=1, scoring=scoring, verbose=True)

X, y = [], []
for i_query, query_train in enumerate(queries_train):
    curr_X = epochs[query_train].copy().decimate(250).get_data()
    X.append(curr_X)
    y.append(np.ones(curr_X.shape[0]) * i_query)
X = np.concatenate(X, axis=0)
y = np.concatenate(y, axis=0)
print(X.shape, y.shape)
# csp = CSP(n_components=3, norm_trace=False)

scores = mne.decoding.cross_val_multiscore(time_gen, X, y, cv=2, n_jobs=-1)

# Mean scores across cross-validation splits
scores = np.mean(scores, axis=0)
fig, ax = plt.subplots(1, 1)
print(scores.shape)
im = ax.imshow(scores, interpolation='lanczos', origin='lower', cmap='RdBu_r', vmin=0., vmax=1.)
# extent = epochs.times[[0, -1, 0, -1]]
ax.set_xlabel('Testing Time (s)')
ax.set_ylabel('Training Time (s)')
ax.set_title(f'Word_number_{target_word_num}.png')
ax.axvline(0, color='k')
ax.axhline(0, color='k')
plt.colorbar(im, ax=ax)
plt.savefig(f'../figures/Word_number_{target_word_num}.png')
print(f'Saved to: ../figures/gat/Word_number_{target_word_num}.png')