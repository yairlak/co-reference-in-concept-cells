#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 00:15:19 2021

@author: yl254115
"""
import os
import pandas as pd


path2channel_file = '../../responsive_channels.xlsx'
print(f'Loading responsive-channels file: {path2channel_file}')
responsive_channels = pd.read_excel(path2channel_file)

for i_row, row in responsive_channels.iterrows():
    if pd.isna(row.patient): raise()
    channels = list(set(str(row.Channel).split(';')))
    # syntactic
    cmd = f"python plot_rasters_sentence.py --patient {int(row.patient)} --session {int(row.session)} --channels {' '.join(channels)} --block syntactic"
    print(cmd)
    os.system(cmd)
    
    # pragmatic
    cmd = f"python plot_rasters_sentence.py --patient {int(row.patient)} --session {int(row.session)} --channels {' '.join(channels)} --block pragmatic"
    print(cmd)
    os.system(cmd)