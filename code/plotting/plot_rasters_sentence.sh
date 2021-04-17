
for patient in 86 87; do
	for block in syntactic pragmatic; do
		for session in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
			python plot_rasters_sentence.py --session $session --block $block --channels --patient $patient
		done
	done
done
