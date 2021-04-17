import sys, os, argparse
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
sys.path.append('..')
from plotting import utils
import HTML
import numpy as np

parser = argparse.ArgumentParser('Plot sentence rasters')
parser.add_argument('--path2data', default = '../../', type=str)
args = parser.parse_args()
print(args)

path2output = '../../HTMLs/overview_plots/'

patients_sessions = [(83, 3), (89, 1)]

#######################
# HTML ALL PATIENTS   #
#######################
# WRITE TO HTML FILE
fn_html = f'overview_all_patients.html'
fn_html = os.path.join(path2output, fn_html)
os.makedirs(os.path.dirname(fn_html), exist_ok=True)

text_list = HTML.HTML_all_patients(patients_sessions)
with open(fn_html, 'w') as f:
    for line in text_list:
        f.write("%s\n" % line)
print('HTML saved to: ', fn_html)

####################
# HTML PER PATIENT #
####################
for patient, session in patients_sessions:
    
    args.patient = patient
    args.session = session
    for block in ['syntactic', 'pragmatic']:
        args.block = block
        # Get TrialInfo
        session_folders = [os.path.basename(x[0]) for x in os.walk(os.path.join(args.path2data, 'Results', f'{args.patient:03d}'))]
        session_folder = [x for x in session_folders if x.startswith(f'{args.patient:03d}') and x.endswith(f'pa{args.session}')][0]
        path2data = os.path.join(args.path2data, 'Results', f'{args.patient:03d}', session_folder)
        TrialInfo, _ = utils.get_data(path2data, args, '')
        # Filter to curren block
        IX_block = np.asarray([bt==args.block for bt in TrialInfo['block_types']])
        for k in TrialInfo.keys(): 
            if k not in ['word_strings']: TrialInfo[k] = TrialInfo[k][IX_block] # Filter trials relevant for current block type
        # Get sentence strings, numbers and target-words
        sentences = sorted(list(set(TrialInfo['sentence_strings'])))
        sentence_numbers, target_words = [], []
        for sentence in sentences:
            IXs_sentence = [ix for ix, (s, bt) in enumerate(zip(TrialInfo['sentence_strings'], TrialInfo['block_types'])) if bt == block and s == sentence]
            sentence_number = TrialInfo['sentence_numbers'][IXs_sentence]
            if args.patient in [89]:
                sentence_number = [n[0,0] for n in sentence_number]
            sentence_number = list(set(sentence_number))
            assert len(sentence_number) == 1
            sentence_numbers.append(sentence_number[0])
            target_word = TrialInfo['target_words'][IXs_sentence]
            target_word = list(set(target_word))
            assert len(sentence_number) == 1
            target_words.append(target_word[0])
        # Build an HTML with all sentences
        text_list = HTML.HTML_sentences(sentences, sentence_numbers, target_words, args)
        # WRITE TO HTML FILE
        fn_html = f'{args.patient}_{args.session}_{args.block}_all_sentences.html'
        fn_html = os.path.join(path2output, fn_html)
        with open(fn_html, 'w') as f:
            for line in text_list:
                f.write("%s\n" % line)
        print('HTML saved to: ', fn_html)
        
        #####################
        # HTML PER SENTENCE #
        #####################       
            
        for sentence_string, sentence_number, target_word in zip(sentences, sentence_numbers, target_words):
            args.sentence_number = sentence_number
            args.sentence_string = sentence_string
            args.target_word = target_word
            text_list = HTML.HTML_units(args)
            fn_html = f'{args.patient}_{args.session}_{args.block}_sentence_{sentence_number}.html'
            fn_html = os.path.join(path2output, fn_html)
            with open(fn_html, 'w') as f:
                for line in text_list:
                    f.write("%s\n" % line)
            print('HTML saved to: ', fn_html)
