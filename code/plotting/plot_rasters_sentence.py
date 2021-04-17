import os, argparse
import utils
from plotting import generate_rasters
import numpy as np
import matplotlib.pyplot as plt
import copy


parser = argparse.ArgumentParser('Plot sentence rasters')
# PATIENT AND BLOCK
parser.add_argument('--patient', default = 89, type=int)
parser.add_argument('--session', default = 1, type=int)
parser.add_argument('--block', choices= ['miniscreening', 'syntactic', 'pragmatic'], default = 'syntactic', type=str)
parser.add_argument('--channels', default = [77], nargs='*', type=int, help='Channels to run. If empty executes for all found units')
# PARADIGM
parser.add_argument('--soa', default = 700, type=int, help='(msec)')
parser.add_argument('--isi-pragmatic', default = 1000, type=int, help='(msec) break duration between sentence1 and sentence2 in the pragmatic block')
parser.add_argument('--isi-question', default = 2000, type=int, help='(msec) break duration between sentence1 and sentence2 in the pragmatic block')
parser.add_argument('--fixation-time', default = 1300, type=int, help='(msec)')
# PLOTTING
parser.add_argument('--plot-spike-profile', default = False, action='store_true', help='Adds spike profile to plot.')
parser.add_argument('--times', default = [-1000, 2000], help='Plotting window')
parser.add_argument('--line-length', default = 0.3, type=int, help='Line length in rasters')
parser.add_argument('--smooth-raster', default = 100, type=int, help='Gaussian window for firing rate (msec)')
# PATHS
parser.add_argument('--path2data', default = '../../', type=str)
parser.add_argument('--path2figures', default = '../../figures/rasters', type=str)
args = parser.parse_args()
print(args)

#############
# LOAD DATA #
#############
session_folders = [os.path.basename(x[0]) for x in os.walk(os.path.join(args.path2data, 'Results', f'{args.patient:03d}'))]
session_folder = [x for x in session_folders if x.startswith(f'{args.patient:03d}') and x.endswith(f'pa{args.session}')][0]
path2data = os.path.join(args.path2data, 'Results', f'{args.patient:03d}', session_folder)
ending = '' if args.block=='miniscreening' else '_sentence'
TrialInfo, cherries = utils.get_data(path2data, args, ending)
# assert len(TrialInfo['word_strings']) == cherries[1]['trial_data'].shape[0] # check that there's unit activity for each word

# KEEP ONLY INFO RELEVANT FOR CURRENT BLOCK
IX_block = np.asarray([bt==args.block for bt in TrialInfo['block_types']])
for k in TrialInfo.keys(): 
      if k not in ['word_strings']: TrialInfo[k] = TrialInfo[k][IX_block] # Filter trials relevant for current block type
#     assert TrialInfo[k].size == cherries[1]['trial_data'].size # sanity check: cherries and trial_info are of the same size



print(list(set(TrialInfo['target_words'])))
print(path2data)
[print(k, len(v)) for k,v in TrialInfo.items()]
print('-'*100)
print('Num trials in cherries:', cherries[1]['trial_data'].shape[0])
for u in cherries.keys():
    assert cherries[u]['trial_data'].shape[0] == len(TrialInfo['sentence_strings']) # match spike data with trial info
print('-'*100)

####################
# GENERATE RASTERS #
####################

if not args.channels: # If empty then run for all found units
    units = cherries.keys()
else:
    units = [u for u in cherries.keys() if int(cherries[u]['channel_num']) in args.channels]

for unit in units:
    for target_word in list(set(TrialInfo['target_words'])):
        args.unit = unit
        args.ch_name = cherries[unit]['channel_name'] 
        args.ch_num = int(cherries[unit]['channel_num']) 
        args.cluster = int(cherries[unit]['class_num']) 
        args.target_word = target_word 
        if args.block == 'syntactic':
            sentences = set(TrialInfo['sentence_strings'])
            sentences = [s for s in sentences if '/' not in s] # '/' is separator of pragamtic-block sentences
            sentences = [s for s in sentences if target_word in s]
            
            # LOOP OVER SENTENCES
            for i_sent, sentence in enumerate(sorted(sentences)):
                fig, axes = generate_rasters(unit, cherries, TrialInfo, sentence, args)
                IXs_sentence = [ix for ix, (s, bt) in enumerate(zip(TrialInfo['sentence_strings'], TrialInfo['block_types'])) if bt == args.block and s == sentence]
                
                # Get sentence number
                sentence_number = TrialInfo['sentence_numbers'][IXs_sentence]
                if args.patient in [89]:
                    sentence_number = [n[0,0] for n in sentence_number]
                sentence_number = list(set(sentence_number))
                assert len(sentence_number) == 1
                sentence_number = sentence_number[0]
                ########
                # SAVE #
                ########
                path2figures = os.path.join(args.path2figures, f'patient_{args.patient:03d}/session_{args.session}/{args.block}')
                if not os.path.exists(path2figures):
                    os.makedirs(path2figures)
                fn_fig = f"raster_{args.patient:03d}_{args.session}_{args.block}_{cherries[unit]['channel_name']}_ch_{int(cherries[unit]['channel_num'])}_cl_{int(cherries[unit]['class_num'])}_{target_word}_sentence_num_{sentence_number}.png"
                plt.savefig(os.path.join(path2figures, fn_fig))
                plt.close(fig)
                print('Figure saved to: ', os.path.join(path2figures, fn_fig))
        
        elif args.block == 'pragmatic':
            IX_target_word = [i for i, tw in enumerate(TrialInfo['target_words']) if tw == target_word]
            TrialInfo_target_word = copy.deepcopy(TrialInfo)
            for k in TrialInfo_target_word.keys(): 
                if k not in ['word_strings']: TrialInfo_target_word[k] = TrialInfo_target_word[k][IX_target_word] # Filter trials relevant for current block type
            cherries_target_word = copy.deepcopy(cherries)
            for u in cherries_target_word.keys():
                cherries_target_word[u]['trial_data'] = cherries_target_word[u]['trial_data'][IX_target_word]
                
            fig, axes = generate_rasters(unit, cherries_target_word, TrialInfo_target_word, [], args)
            # fig, axes = generate_rasters(unit, cherries, TrialInfo, [], args)
                        
            
            ########
            # SAVE #
            ########
            path2figures = os.path.join(args.path2figures, f'patient_{args.patient:03d}/session_{args.session}/{args.block}')
            if not os.path.exists(path2figures):
                os.makedirs(path2figures)
            fn_fig = f"raster_pt{args.patient:03d}_s{args.session}_{args.block}_{cherries[unit]['channel_name']}_ch_{int(cherries[unit]['channel_num'])}_cl_{int(cherries[unit]['class_num'])}_{target_word}.png"
            plt.subplots_adjust(top=0.95, bottom=0.05)
            plt.savefig(os.path.join(path2figures, fn_fig))
            plt.close(fig)
            print('Figure saved to: ', os.path.join(path2figures, fn_fig))