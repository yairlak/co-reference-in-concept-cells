#!/bin/bash -x
for patient in 87; do
	for session in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
		for block in 'pragmatic' 'syntactic'; do
			python raw2mne.py --patient $patient --session $session --block $block 
		done
	done
done
