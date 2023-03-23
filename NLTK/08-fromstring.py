'''
文法
自定义文法
写法上与上一篇博文的分类规则思路基本一致，并且更简单、更直观，可以和之前的对比着看。
'''

import nltk
from nltk.corpus import treebank

def test1():
    grammar = nltk.CFG.fromstring("""
    S -> NP VP
    VP -> V NP | V NP PP
    PP -> P NP
    V -> "saw" | "ate" | "walked"
    NP -> "John" | "Mary" | "Bob" | Det N | Det N PP
    Det -> "a" | "an" | "the" | "my"
    N -> "man" | "dog" | "cat" | "telescope" | "park"
    P -> "in" | "on" | "by" | "with"
    """)
    sent = 'Mary saw my dog'.split()
    rd_parser = nltk.RecursiveDescentParser(grammar)
    tree_generator = rd_parser.parse(sent)

    for tree in tree_generator:
        print('tree: ',tree)
        tree.draw()

def test2():
    # 下面程序展示了利用简单的过滤器，找出带句子补语的动词
    tree = treebank.parsed_sents('wsj_0001.mrg')[0]
    print(tree) #查看封装好的文法
    # tree.draw()

    def filter(tree):
        child_nodes = [child.label() for child in tree if isinstance(child,nltk.Tree)]
        return (tree.label() == 'VP') and ('S' in child_nodes)#找出带句子补语的动词

    print([subtree for tree in treebank.parsed_sents() \
            for subtree in tree.subtrees(filter)])

test2()