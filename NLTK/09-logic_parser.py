import nltk
from nltk import load_parser
from nltk.sem.logic import LogicParser


def test1():
    # 输入自然语言文本
    text = "all dogs bark"

    # 定义文法规则
    grammar = nltk.CFG.fromstring("""
        S -> NP VP
        NP -> Det N
        VP -> V | V NP
        Det -> 'all'
        N -> 'dogs' | 'cats'
        V -> 'bark'
    """)

    # 创建解析器
    parser = nltk.ChartParser(grammar)

    # 解析自然语言文本
    for tree in parser.parse(text.split()):
        print(tree)
        tree.draw()
        # 构造逻辑表示式
        expression = LogicParser().parse(str(tree))
        # 打印逻辑表示式
        print(expression)

def test2():
    # nltk.data.show_cfg('grammars/book_grammars/sql0.fcfg')
    cp = load_parser('NLTK/sql0.fcfg')
    query = 'What cities are located in China'

    trees = list(cp.parse(query.split()))
    trees[0].draw()

    answer = trees[0].label()['SEM']
    answer = [s for s in answer if s]
    q = ' '.join(answer)
    print(q)

test2()