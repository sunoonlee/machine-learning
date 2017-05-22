#!usr/bin/env python3
# coding: utf-8
"""读取语料 生成 n-gram 模型"""

from collections import Counter, defaultdict
#from pprint import pprint
from random import random
import jieba


# N = 2  # N元模型
START = '$$' # 句首的 token
BREAK = '。！？'  # 作为句子结束的符号
IGNORE = '\n “”"《》〈〉()*'  # 忽略不计的符号


def process_segs(segments):
    """对 segments (iterator) 进行处理，返回一个 list. 处理规则： 
    - 忽略 \n、空格、引号、书名号等
    - 在断句符号后添加 START token
    """
    results = [START for i in range(N-1)]
    for seg in segments:
        if seg in IGNORE:
            continue
        else:
            results.append(seg)
            if seg in BREAK:
                results.extend([START for i in range(N-1)])
    # 小瑕疵：segments 会以 start token 结束，但对语言模型没有影响，暂且忽略
    return results


def count_ngram(segments):
    """统计 N-gram 出现次数"""
    dct = defaultdict(Counter)
    for i in range(N-1, len(segments)):
        context = tuple(segments[i-N+1:i])
        word = segments[i]
        dct[context][word] += 1
    return dct


def to_prob(dct):
    """将次数字典转换为概率字典"""
    prob_dct = dct.copy()
    for context, count in prob_dct.items():
        total = sum(count.values())
        for word in count:
            count[word] /= total  # works in Python 3
    return prob_dct


def generate_word(context):
    """根据 context 及条件概率，随机生成 word"""
    r = random()
    psum = 0
    for word, prob in prob_dct[context].items():
        psum += prob
        if psum > r:
            return word
    #return START


def generate_sentences(m):
    """生成 m 个句子"""
    sentences = []
    text = ''
    context = tuple(START for i in range(N-1))
    i = 0
    while (i < m):
        word = generate_word(context)
        text = text + word
        context = tuple((list(context) + [word])[1:])
        if word in BREAK:
            sentences.append(text)
            text = ''
            context = tuple(START for i in range(N-1))
            i += 1
    return sentences


for N in range(2, 6):
    print('\n*** reading corpus ***')
    with open('ZhangAiLing.txt') as f:
        corpus = f.read()
    print('*** cutting corpus ***')
    raw_segments = jieba.cut(corpus)
    print('*** processing segments ***')
    segments = process_segs(raw_segments)
    print('*** generating {}-gram count dict ***'.format(N))
    dct = count_ngram(segments)
    print('*** generating {}-gram probability dict ***'.format(N))
    prob_dct = to_prob(dct)
    #pprint(prob_dct)
    print('*** generating sentences ***')
    with open('generated_{}gram.txt'.format(N), 'w') as f:
        f.write('\n'.join(generate_sentences(20)))
