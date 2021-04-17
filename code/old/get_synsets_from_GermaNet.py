from germanetpy.germanet import Germanet
from germanetpy.frames import Frames
from germanetpy.filterconfig import Filterconfig
from germanetpy.synset import WordCategory, WordClass
germanet = Germanet("data")

target_word = "Mann"
synsets = germanet.get_synsets_by_orthform(target_word)
# the lengths of the retrieved list is equal to the number of possible senses for a word, in this case 2
print("\n%s has %d senses " % (target_word.upper(), len(synsets)))
for synset in synsets:
    print()
    print("Synset id %s:\n----------------\nLexical units: %s\nWord category: %s\nsemantic field: %s.\n" %
          (synset.id, ', '.join([l.orthform for l in synset.lexunits]), synset.word_category, synset.word_class))

    for relation, related_synsets in synset.relations.items():
        print("\nrelation : %s; %s" % (relation, related_synsets))

    print("The synset has a depth of %d \n is it the root node? %s  \n is it a leaf node? %s"
          % (synset.min_depth(), str(synset.is_root()), str(synset.is_leaf())))

    lexical_info = False
    if lexical_info:
        for lexunit in synset.lexunits:
            orth_forms = lexunit.get_all_orthforms()
            print(orth_forms)
            print(lexunit.compound_info)
            print(lexunit.relations)
            print(lexunit.incoming_relations)
            print(lexunit.wiktionary_paraphrases)
            print(lexunit.ili_records)