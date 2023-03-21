'''
就是把长句子拆成有"意义"的小部件
'''
import re
import nltk

# punkt.zip需要解压
# nltk.download('punkt')

def hello():
    sentence = "hello, world"
    # sentence = "你好，世界" # 不支持中文
    tokens = nltk.word_tokenize(sentence)
    print(tokens) # ['hello', ',', 'world']

def net():
    # 特殊的组合单词，网络用语
    tweet = 'RT @angelababy: love you baby! :D http://ah.love #168cm'
    print(nltk.word_tokenize(tweet)) 
    # ['RT', '@', 'angelababy', ':', 'love', 'you', 'baby', '!', ':', 'D', 'http', ':', '//ah.love', '#', '168cm']

def net_pro():
    emotions_str = r"""
        (?: # (?:)表示非捕获分组
            [:=;] # 眼睛
            [oO\-]? # ⿐鼻⼦子
            [D\)\]\(\]/\\OpP] # 嘴
        )"""
    regex_str = [
        emotions_str,
        r'<[^>]+>', # HTML tags
        r'(?:@[\w_]+)', # @某⼈人
        r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", # 话题标签
        r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', # URLs
        r'(?:(?:\d+,?)+(?:\.?\d+)?)', # 数字
        r"(?:[a-z][a-z'\-_]+[a-z])", # 含有 - 和 ‘ 的单词
        r'(?:[\w_]+)', # 其他
        r'(?:\S)' # 其他
        ]

    tokens_re = re.compile(r'('+'|'.join(regex_str)+')', re.VERBOSE | re.IGNORECASE) #所有特殊字符的匹配式
    emotion_re = re.compile(r'^'+emotions_str+'$',re.VERBOSE | re.IGNORECASE) #表情符号的匹配式

    def tokenize(s):
        return tokens_re.findall(s) #匹配出所有特殊字符

    def preprocess(s,lowercase=False):
        tokens = tokenize(s)
        if lowercase:
            tokens = [token if emotion_re.search(token) else token.lower() for token in tokens]#对非表情符号进行小写处理
        return tokens #返回所有特殊字符

    tweet = 'RT @angelababy: love you baby! :D http://ah.love #168cm'
    print(preprocess(tweet))
    # ['RT', '@angelababy', ':', 'love', 'you', 'baby', '!', ':D', 'http://ah.love', '#168cm']

net_pro()