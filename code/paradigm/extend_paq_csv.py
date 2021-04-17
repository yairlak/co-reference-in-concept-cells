#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 20:43:51 2021

@author: yl254115
"""

import os, glob, csv

patients = ['084', '086', '087']
path2data = os.path.join('../../Results')
for patient in patients:
    for d in os.scandir(os.path.join(path2data, patient)):
        if d.is_dir() and d.name.startswith(patient):
            files = glob.glob(os.path.join(path2data, patient, d.name, 'paq_sentences*.csv'))
            assert len(files) == 1
            print('Duplicating paq csv file of patient/session: ', patient, d.name)
            fn_paq = files[0]
            fn_new_paq = os.path.join(os.path.dirname(fn_paq), 'extended_' + os.path.basename(fn_paq))
            with open(fn_paq, 'r') as infile, open(fn_new_paq, 'w') as outfile:
                reader = csv.reader(infile, delimiter=';')
                previous_row = None
                for i_row, row in enumerate(reader):
                    if i_row == 0: # header
                        extended_row = row + ['%ref2target']
                    else:
                        #print(row[3], row[7])
                        if row[7] in ['-', '--', '_', '-0'] or len(row[7])==0: # control sentence
                            stim_num_of_target = previous_row[3]
                        else:
                            stim_num_of_target = ' '
                        extended_row = row + [stim_num_of_target]
                    outfile.write(';'.join(extended_row)+'\n')
                    previous_row = row