import nltk

def test1():
    read_expr = nltk.sem.Expression.fromstring
    SnF = read_expr('SnF')
    NotFnS = read_expr('-FnS')
    R = read_expr('SnF -> -FnS')
    prover = nltk.Prover9() # 需要安装：https://www.cs.unm.edu/~mccune/prover9/gui/v05.html
    prover.prove(NotFnS, [SnF, R]) # True

def test2():
    val = nltk.Valuation([('P', True), ('Q', True), ('R', False)])
    val['P']
    dom = set()
    g = nltk.Assignment(dom)
    m = nltk.Model(dom, val)	
    print(m.evaluate('(P & Q)', g))# True
    print(m.evaluate('-(P & Q)', g))# False
    print(m.evaluate('(P & R)', g))# False
    print(m.evaluate('(P | R)', g))# True

test2()