import nltk
from nltk import load_parser

nltk.data.show_cfg('NLTK\simple-sem.fcfg')
parser = load_parser('NLTK\simple-sem.fcfg', trace=0)
sentence = 'Angus gives a bone to every dog'
tokens = sentence.split()
for tree in parser.parse(tokens):
    print(tree.label()['SEM'])