'''
实体识别：分块技术
比如：We saw the yellow dog ，
按照分块的思想，会将后三个词语分到NP中，而里面的三个词又分别对应 DT/JJ/NN；
saw 分到VBD中；We 分到NP中。对于最后三个词语来说，NP就是组块（较大的集合）。为
了做到这点，可以借助NLTK自带的分块语法，类似于正则表达式，来实现句子分块。
'''
import nltk

def test1():
    sentence = [('the','DT'),('little','JJ'),('yellow','JJ'),('dog','NN'),('brak','VBD')]
    grammer = "NP: {<DT>?<JJ>*<NN>}"
    cp = nltk.RegexpParser(grammer) #生成规则
    result = cp.parse(sentence) #进行分块
    print(result)

    result.draw() #调用matplotlib库画出来

def test2():
    text = nltk.word_tokenize('We saw the yellow dog')
    print(text)

    sentence = nltk.pos_tag(text)
    print(sentence)

    grammer = "NP: {<DT>?<JJ>*<NN>}"
    cp = nltk.RegexpParser(grammer) #生成规则

    result = cp.parse(sentence) #进行分块
    print(type(result)) # <class 'nltk.tree.tree.Tree'>
    print(result)
    result.draw() #调用matplotlib库画出来

def test2_2():
    sentence = [('the','DT'),('little','JJ'),('yellow','JJ'),('dog','NN'),('bark','VBD'),('at','IN'),('the','DT'),('cat','NN')]
    grammer = """NP: {<DT>?<JJ>*<NN>}
                }<VBD|NN>+{
                """  #加缝隙，必须保存换行符
    cp = nltk.RegexpParser(grammer) #生成规则
    result = cp.parse(sentence) #进行分块
    print(result)
    result.draw() #调用matplotlib库画出来


def test3():
    tree1 = nltk.Tree('NP',['Alick'])
    print(tree1)
    tree2 = nltk.Tree('N',['Alick','Rabbit'])
    print(tree2)
    tree3 = nltk.Tree('S',[tree1,tree2])
    print(tree3.label()) #查看树的结点
    tree3.draw()

def test4():
    # nltk.download('maxent_ne_chunker')
    sentence = [('the','DT'),('little','JJ'),('yellow','JJ'),('dog','NN'),('bark','VBD'),('at','IN'),('the','DT'),('cat','NN')]
    result = nltk.ne_chunk(sentence)
    print(result)
    result.draw() #调用matplotlib库画出来

def test4_2():
    sentence = """We saw the yellow dog"""
    tokens = nltk.word_tokenize(sentence)
    tagged = nltk.pos_tag(tokens)
    entities = nltk.ne_chunk(tagged)
    print(entities)
    entities.draw() #调用matplotlib库画出来


test4_2()
