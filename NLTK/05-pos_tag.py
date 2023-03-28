'''
词性标注
 POS TAG （Part-Of-Speech)
'''
import nltk, pprint
# nltk.download('tagsets')
'''
>>> nltk.help.upenn_tagset('RB')
RB: adverb
    occasionally unabatingly maddeningly adventurously professedly
    stirringly prominently technologically magisterially predominately
    swiftly fiscally pitilessly ...
'''

def test1():
    # text = nltk.word_tokenize('what does the fox say')
    # 分词
    text = nltk.word_tokenize("They refuse to permit us to obtain the refuse permit")
    print(text)

    # 词性标注
    tagged = nltk.pos_tag(text)
    print(tagged)

    # 显示每个标注的含义
    tags = set(tag[1] for tag in tagged)
    print(tags)
    for tag in tags:
        nltk.help.upenn_tagset(tag)

def test2():
    from nltk.corpus import brown
    # brown.readme()
    brown_news_tagged = brown.tagged_words(categories='news', tagset='universal') # universal-通用标记集
    tag_fd = nltk.FreqDist(tag for (word, tag) in brown_news_tagged)
    tag_fd_num = tag_fd.most_common() # 词性数量统计
    print(tag_fd_num)
    # for tag, num in tag_fd_num:
    #     nltk.help.upenn_tagset(tag)
    tag_fd.plot()

test2()
