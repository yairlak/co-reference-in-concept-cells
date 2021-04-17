# Related links:
# --------------
# https://github.com/Blubberli/germanetpy/blob/master/germanetpy_tutorial.ipynb
# https://github.com/Germanet-sfs/germanetpy

import argparse
import sys, os, io
import numpy as np
from germanetpy.germanet import Germanet
from germanetpy.synset import WordCategory, WordClass
from germanetpy.path_based_relatedness_measures import PathBasedRelatedness
import googletrans
import urllib  # the lib that handles the url stuff

parser = argparse.ArgumentParser(description='Plot word embeddings')
parser.add_argument('--filename-wordvecs', type=str, default='/volatile/Projects/SU_Florian/Stimuli/word_emebedding_German/model.txt')
parser.add_argument('-w', '--target-word', type=str, default='Spaghetti')
parser.add_argument('-n', '--num-neighbors', type=int, default=30)
parser.add_argument('--path2output', type=str, default='../output')
parser.add_argument('--download-images', action='store_true', default=False)
parser.add_argument('--max-num-images', default=5, type=int, help='Limit the number of images to download per synset (set to -1 to download all).')
parser.add_argument('--lexical-info', action='store_true', default=False)
args=parser.parse_args()


########
# INIT #
########
args.target_word = args.target_word.capitalize()
output_dir = os.path.join(args.path2output, args.target_word)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

translator = googletrans.Translator()
target_word_en = translator.translate(args.target_word, src='de', dest='en').text

##################
# Word Embedding #
##################

def load_vectors(fname, vocab_size = 50000):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    words = []; data = []; i = 0
    if vocab_size == 0:
        vocab_size = n
    print('Vocabulary size (#words): %i, vector dimension %i' % (vocab_size, d))
    for line in fin:
        if i < vocab_size - 1:
            tokens = line.rstrip().split(' ')
            words.append(tokens[0])
            data.append(list(map(float, tokens[1:])))
            i += 1
    return words, np.vstack(data), n, d


# LOAD DATA
vocab, wordvecs, n, d = load_vectors(args.filename_wordvecs, vocab_size = 25000)

with open('vocab_German_25k.txt', 'w') as f:
    for w in vocab:
        f.write('%s\n'%w)

# FIND NEAREST NEIGHBORS
wordvecs = wordvecs/(np.sqrt((wordvecs*wordvecs).sum(axis=1))[:, np.newaxis]) # L2-normalize word vectors
try:
    IX_word = vocab.index(args.target_word.lower())
except:
    raise(f'Word {args.target_word} was not found!')


wordvec = wordvecs[IX_word, :]
D = wordvecs @ wordvec
IX_nearest_neighbors = np.argsort(-D)[1:args.num_neighbors]


############
# GermaNet #
############
germanet = Germanet("data")

target_synsets = germanet.get_synsets_by_orthform(args.target_word)
# the lengths of the retrieved list is equal to the number of possible senses for a word
s = "%s(%s) has %d senses:" % (args.target_word.upper(), target_word_en, len(target_synsets))
print('\n', '-'*len(s), '\n', s, '\n', '-'*len(s), '\n')
for i_sense, target_synset in enumerate(target_synsets):

    ###############
    # Synset Info #
    ###############
    s = 'SENSE %i:' % (i_sense+1)
    print('\n', s)
    print('-'*len(s))


    for lexunit in target_synset.lexunits:
        lexunit = germanet.get_lexunit_by_id(lexunit.id)
        wiktionary_paraphrases = lexunit.wiktionary_paraphrases
        if wiktionary_paraphrases:
            for wiki_phrase in wiktionary_paraphrases:
                if hasattr(wiki_phrase, 'wiktionary_sense'):
                    if wiki_phrase.wiktionary_sense:
                        print('%s (%s)' % (wiki_phrase.wiktionary_sense, translator.translate(wiki_phrase.wiktionary_sense, src='de', dest='en').text))
        if lexunit.ili_records:
            print(lexunit.ili_records)
            wnid = lexunit.ili_records[0].pwn30id.split('-')[1]
            url = f'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n{wnid}'
            if args.download_images:
                URL2images = urllib.request.urlopen(url)  # it's a file like object and works just like a file
                for i_url, url in enumerate(URL2images):  # files are iterable
                    if i_url < args.max_num_images:
                        fn = os.path.join(output_dir, f'{args.target_word}_{i_url}.jpg')
                        try:
                            urllib.request.urlretrieve(url.decode("utf-8").strip(), fn)
                        except:
                            print(f'Image failed to be downloaded from: {url.decode("utf-8").strip()}', file=sys.stderr)
                        # print(url.decode("utf-8").strip())
            else:
                print(f'http://www.image-net.org/synset?wnid=n{wnid}')

    relation_types = [relation.name for relation, related_synsets in target_synset.relations.items() if related_synsets]
    relation_types = list(set(relation_types))
    for relation_type in sorted(relation_types):
        print('\n%s:'%relation_type.split('.')[-1].upper())
        for relation, related_synsets in target_synset.relations.items():
            if related_synsets and relation.name == relation_type:
                for related_synset in related_synsets:
                    for lexunit in related_synset.lexunits:
                        orthform = lexunit.orthform
                        orthform_en =  translator.translate(orthform, src='de', dest='en').text
                        print("%s (%s)" % (orthform, orthform_en))

                        # Download images
                        if args.download_images:
                            if not os.path.exists(os.path.join(output_dir, 'SENSE %i' % (i_sense+1), relation_type, orthform)):
                                os.makedirs(os.path.join(output_dir, 'SENSE %i' % (i_sense+1), relation_type, orthform))

                            if lexunit.ili_records:
                                wnid = lexunit.ili_records[0].pwn30id.split('-')[1]
                                url = f'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n{wnid}'
                                URL2images = urllib.request.urlopen(url)  # it's a file like object and works just like a file
                                for i_url, url in enumerate(URL2images):  # files are iterable
                                    if i_url < args.max_num_images:
                                        fn = os.path.join(output_dir, 'SENSE %i' % (i_sense+1), relation_type, orthform, f'{orthform}_{i_url}.jpg')
                                        try:
                                            urllib.request.urlretrieve(url.decode("utf-8").strip(), fn)
                                        except:
                                            print(f'Image failed to be downloaded from: {url.decode("utf-8").strip()}', file=sys.stderr)




    ###################
    # Path Simliarity #
    ###################
    print('\nSIMILAR WORDS:')

    johannis_wurm = germanet.get_synset_by_id("s49774")
    leber_trans = germanet.get_synset_by_id("s83979")
    relatedness_calculator = PathBasedRelatedness(germanet=germanet, category=WordCategory.nomen, max_len=35,
                                                  max_depth=20, synset_pair=(johannis_wurm, leber_trans))
    for ix_neighbor in IX_nearest_neighbors:
        neighbor = vocab[ix_neighbor]
        neighbor_en = translator.translate(neighbor, src='de', dest='en').text
        sim_cosine = D[ix_neighbor]
        neighbor_synsets = germanet.get_synsets_by_orthform(neighbor.capitalize())

        s = f"{neighbor} ({neighbor_en}), {sim_cosine};"
        if neighbor_synsets:
            max_sim = 0 # Find lexical item of neighbor that is most similar to current sense of the target word
            for neighbor_synset in neighbor_synsets:
                if neighbor_synset.word_category == WordCategory.nomen:
                    sim_simple = relatedness_calculator.simple_path(neighbor_synset, target_synset)
                    if sim_simple > max_sim:
                        sim_simple_best = sim_simple
                        neighbor_synset_best = neighbor_synset
                        max_sim = sim_simple_best
                    # sim_leacock_chodorow = relatedness_calculator.leacock_chodorow(neighbor_synset, target_synset)

            web_links = []
            if max_sim > 0:
                str_lexical_units = 'Lexical units:'
                for l in neighbor_synset.lexunits:
                    str_lexical_units += '%s|' % (l.orthform)
                    str_lexical_units = str_lexical_units[:-1] + ';'
                    if l.ili_records:
                        str_lexical_units += '(%s;%s)|' % (l.ili_records[0].english_equivalent, l.ili_records[0].lexunit_id)
                        wnid = l.ili_records[0].pwn30id.split('-')[1]

                        url = f'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n{wnid}'
                        if args.download_images:
                            if not os.path.exists(
                                    os.path.join(output_dir, 'SENSE %i' % (i_sense + 1), 'Similar words', l.orthform)):
                                os.makedirs(
                                    os.path.join(output_dir, 'SENSE %i' % (i_sense + 1), 'Similar words', l.orthform))

                            URL2images = urllib.request.urlopen(
                                url)  # it's a file like object and works just like a file
                            for i_url, url in enumerate(URL2images):  # files are iterable
                                if i_url < args.max_num_images:
                                    fn = os.path.join(output_dir, 'SENSE %i' % (i_sense + 1), 'Similar words', l.orthform, f'{l.orthform}_{i_url}.jpg')
                                    try:
                                        urllib.request.urlretrieve(url.decode("utf-8").strip(), fn)
                                    except:
                                        print(f'Image failed to be downloaded from: {url.decode("utf-8").strip()}', file=sys.stderr)
                                    # print(url.decode("utf-8").strip())
                        else:
                            print(f'http://www.image-net.org/synset?wnid=n{wnid}')

                s += f" {sim_simple}, {str_lexical_units}"
                for lexunit in neighbor_synset_best.lexunits:
                    lexunit = germanet.get_lexunit_by_id(lexunit.id)
                    wiktionary_paraphrases = lexunit.wiktionary_paraphrases
                    for wiki_phrase in wiktionary_paraphrases:
                        if hasattr(wiki_phrase, 'wiktionary_sense'):
                            if wiki_phrase.wiktionary_sense:
                                s+= 'Wiki: %s (%s)' % (wiki_phrase.wiktionary_sense, translator.translate(wiki_phrase.wiktionary_sense, src='de', dest='en').text)
            print(s)
            [print(w) for w in web_links]

    ################
    # Lexical Info #
    ################
    if args.lexical_info:
        for lexunit in target_synset.lexunits:
            orth_forms = lexunit.get_all_orthforms()
            print(orth_forms)
            print(lexunit.compound_info)
            print(lexunit.relations)
            print(lexunit.incoming_relations)
            print(lexunit.wiktionary_paraphrases)
            print(lexunit.ili_records)

print('Results saved to: %s.txt' % os.path.join(output_dir, args.target_word))