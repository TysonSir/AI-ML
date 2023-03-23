import nltk
from nltk.sem.logic import LogicParser

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