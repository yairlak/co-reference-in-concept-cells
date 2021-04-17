import argparse
import io
import numpy as np

parser = argparse.ArgumentParser(description='Plot word embeddings')
parser.add_argument('--filename-wordvecs', type=str, default='/volatile/Projects/SU_Florian/Stimuli/word_emebedding_German/model.txt')
parser.add_argument('-w', '--target-word', type=str, default='Mann')
parser.add_argument('-n', '--num-neighbors', type=int, default=30)
args=parser.parse_args()

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

# FIND NEAREST NEIGHBORS
wordvecs = wordvecs/(np.sqrt((wordvecs*wordvecs).sum(axis=1))[:, np.newaxis]) # L2-normalize word vectors
try:
    IX_word = vocab.index(args.target_word.lower())
except:
    raise(f'Word {args.target_word} was not found!')


wordvec = wordvecs[IX_word, :]
D = wordvecs @ wordvec
IX_nearest = np.argsort(-D)

ttl = f'\nNearset Neighbors to {args.target_word.upper()}:'
print(ttl)
print('-'*len(ttl))
print("\n{0:50} {1}".format('Word', 'Cosine distance'))
print("{0:50} {1}".format('----', '---------------'))
[print("{0:50} {1}".format(vocab[ix], D[ix])) for ix in IX_nearest[:args.num_neighbors] if ix!=IX_word]
