#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 14:16:59 2021

@author: yl254115
"""

path2nev = '/home/yl254115/projects/coreference/Results/087/087e47pa8/events'
import os, sys
sys.path.insert(1, '/home/yl254115/projects/brPY')
from brpylib import NevFile
import neo

from neo import io
import matplotlib.pyplot as plt
import numpy as np
import sys

recording_system = 'Neuralynx'


if recording_system == 'Neuralynx':
    NIO = io.NeuralynxIO(path2nev)
    print(NIO)
    #print('Sampling rate of signal:', NIO._sigs_sampling_rate)
    time0, timeend = NIO._timestamp_limits[0]
    print('time0, timeend = ', time0, timeend)
elif recording_system == 'BlackRock':
    NIO = io.BlackrockIO(os.path.join(path2nev, 'Events.nev'))
    events = NIO.nev_data['NonNeural']
    time_stamps = [e[0] for e in events]
    event_num = [e[4] for e in events]
    plt.plot(np.asarray(time_stamps)/40000, event_num, '.')
    plt.show()