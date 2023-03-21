'''
Lemmatization 词形归⼀：把各种类型的词的变形，都归为⼀个形式
went 归⼀ = go
are 归⼀ = be
'''

from nltk.stem import WordNetLemmatizer as WL
wl = WL()
print(wl.lemmatize('dogs')) # dog
print(wl.lemmatize('went'))
print(wl.lemmatize('are')) # are
print(wl.lemmatize('are', pos='v')) # 理解为动词，be
