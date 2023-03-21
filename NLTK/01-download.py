'''
下载语料库
nltk包里面包含了很多的语料库corpus，有古藤堡，莎士比亚作品啊之类。
'''
import nltk
# nltk.download()

from nltk.corpus import brown #加载布朗大学语料库
print(brown.categories())

#该语料库的句子数，单词数
print(len(brown.sents()),len(brown.words()))