

def RMM(dict, sentence): # 逆向最大匹配算法RMM函数，参数dict: 词典 ，参数sentence: 句子 
    rmmresult = [] 
    max_len = max([len(item) for item in dict])# max_len定义为词典中最长词长度 
    start = len(sentence) 
    while start != 0: # RMM 为逆向，start 从末尾位置开始，指向开头位置即为结束 
        index = start - max_len # 逆向时 index 的初始值为 start 的索引 - 词典中元素的最大长度或句子开头 
        if index < 0: 
           index = 0
        for i in range(max_len): 
            # 当分词在字典中时或分到最后一个字时，将其加入到结果列表中 
            if (sentence[index:start] in dict) or (len(sentence[index:start]) == 1): 
                # print(sentence[index:start], end='/') 
                rmmresult.insert(0, sentence[index:start])   
                start = index# 分出一个词，start 设置到 index 处 
                break                                    
            index += 1 # 如果匹配失败，则去掉最前面一个字符
    return rmmresult


word_dict = ['小车', '的', '最大运行速度', '1', '米/秒']
sent = '小车的最大运行速度1米/秒'
words = RMM(word_dict, sent)
print('--'.join(words))