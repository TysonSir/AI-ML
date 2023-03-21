'''
Stemming 词⼲提取：⼀般来说，就是把不影响词性的inflection的⼩尾巴砍掉
walking 砍ing = walk
walked 砍ed = walk
当然 nlkt 中还有几种stemmer：PorterStemmer，SnowballStemmer，LancasterStemmer
'''

from nltk.stem.porter import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.snowball import SnowballStemmer

stemmer = PorterStemmer()
print(stemmer.stem('maximum'))
print(stemmer.stem('walking'))
print(stemmer.stem('walked'))

stemmer = LancasterStemmer()
print(stemmer.stem('maximum'))
print(stemmer.stem('walking'))
print(stemmer.stem('walked'))

stemmer = SnowballStemmer('english')
print(stemmer.stem('maximum'))
print(stemmer.stem('walking'))
print(stemmer.stem('walked'))