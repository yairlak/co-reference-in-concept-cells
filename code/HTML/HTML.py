import sys, os, argparse
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
sys.path.append('..')

# from utils import load_settings_params
from scipy import io
import os, glob, argparse
import numpy as np


def HTML_all_patients(patients_sessions):
    '''
    '''

    text_list = []
    # HEADERS
    text_list.append('<head>\n')
    text_list.append(f'<title> Overview plots </title>\n')
    text_list.append('</head>\n')
    text_list.append('<body>\n')
    
    # MAIN
    for patient, session in patients_sessions:
        for block in ['syntactic', 'pragmatic']:
            fn_html = f'{patient}_{session}_{block}_all_sentences.html'
            text_list.append(f'<a href={fn_html}>patient {patient} - session {session} - {block}</a>')
            text_list.append(f'<br>\n')

    return text_list


def HTML_sentences(sentences, sentence_numbers, target_words, args):
    '''
    per_probe_htmls - list of sublists; each sublist contains [patient, probe_names, fn_htmls]
    '''

    text_list = []
    # HEADERS
    text_list.append('<head>\n')
    text_list.append(f'<title> {args.patient} {args.session} {args.block} </title>\n')
    text_list.append('</head>\n')
    text_list.append('<body>\n')
    
    # MAIN
    text_list.append(f'<h2 style="font-family:tempus sans itc;"><p>{args.patient}, {args.session}, {args.block}</p>')
    for sentence_string, sentence_num, target_word in zip(sentences, sentence_numbers, target_words):
        fn_html = f'{args.patient}_{args.session}_{args.block}_sentence_{sentence_num}.html'
        text_list.append(f'<a href={fn_html}>{sentence_string} ({sentence_num};{target_word})</a>')
        text_list.append(f'<br>\n')
        
    return text_list


def HTML_units(args):
    '''
    per_probe_htmls - list of sublists; each sublist contains [patient, probe_names, fn_htmls]
    '''

    text_list = []
    
    path2figures = f'../../figures/rasters/patient_{args.patient:03d}/session_{args.session}/{args.block}'
    pattern = f'raster_{args.patient:03d}_{args.session}_{args.block}_*_ch_*_cl_*_{args.target_word}_sentence_num_{args.sentence_number}.png'
    fns_figs = glob.glob(os.path.join(path2figures, pattern))
    # HEADERS
    text_list.append('<head>\n')
    text_list.append(f'<title> {args.patient} {args.session} {args.block} {args.sentence_number} </title>\n')
    text_list.append('</head>\n')
    text_list.append('<body>\n')
    
    # MAIN
    print(f'{args.patient} {args.session} {args.block} {args.sentence_number}', len(fns_figs))
    for fn_fig in fns_figs:
        text_list.append(f'<img class="right" src="{fn_fig}" stye="width:1024px;height:512px;">\n')
        # text_list.append('<br>\n')
        
    return text_list