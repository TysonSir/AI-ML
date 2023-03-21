'''
Stopwords 停用词
对于注重理解⽂本『意思』的应⽤场景来说歧义太多
'''

import nltk
from nltk.corpus import stopwords
sents = 'what does the fox say' # ['fox', 'say']
sents = 'there is a book' # book
sents = 'Jack like beautiful Rose is fine' # ['Jack', 'like', 'beautiful', 'Rose', 'fine']

text = nltk.word_tokenize(sents)
print(text)

filtered_words = [word for word in text if word not in stopwords.words('english')]
print(filtered_words)