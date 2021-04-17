

for WORD in Hund Strand Kaffee; do

echo python get_related_words.py -w $WORD
python get_related_words.py -w $WORD --download-images --max-num-images 3 > ../output/$WORD.txt

done
