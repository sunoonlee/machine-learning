#!/usr/bin/env python3
# coding: utf-8

""" data reading helper functions for conv_nlp.py """

from collections import Counter
import jieba
import numpy as np


IGNORE = ' \n'  # 忽略的字符
DOC_LENGTH = 20  # 预设的固定句子长度
PADDING = '<PD>'  # 句子长度不足时的占位符


def read_docs_and_labels(filename):
    """从文件读取样本, 去除忽略字符, 得到句子和标签列表"""    
    with open(filename, encoding='utf-8') as f:
        lines = f.readlines()
    
    docs, labels = [], []
    for line in lines:
        text, label = line.split('\t')

        words_in_doc = []
        for word in jieba.cut(text):
            if (word not in IGNORE) and (not word.isdigit()):
                words_in_doc.append(word)
        
        docs.append(words_in_doc)
        labels.append(int(label.strip()))
    return docs, labels


def fix_doc_length(docs):
    """将所有样本的词列表调整为固定长度"""
    for i in range(len(docs)):
        if len(docs[i]) < DOC_LENGTH:
            docs[i].extend([PADDING] * (DOC_LENGTH - len(docs[i])))
        else:
            docs[i] = docs[i][:DOC_LENGTH]
    return docs


def build_vocab(train_file, count_limit=3):
    """由训练样本集建立词表, 仅计入出现次数超过设定值(默认为3)的词"""
    train_docs, _ = read_docs_and_labels(train_file)
    train_docs = fix_doc_length(train_docs)

    flat_words = [w for doc in train_docs for w in doc]
    word_cnt = Counter(flat_words)

    vocab = ['UNK']
    for word, count in word_cnt.most_common():
        if count > count_limit:
            vocab.append(word)
        else:
            break
    return vocab


def generate_inputs_and_labels(filename, vocab):
    """生成 numpy array 类型的输入和标签数据"""
    docs, labels = read_docs_and_labels(filename)
    docs = fix_doc_length(docs)
    idx_dict = dict(zip(vocab, range(len(vocab))))  # 由词映射到词序号的字典

    # convert word lists to index lists
    idxes = []
    for doc in docs:
        idxes_of_one_doc = []
        for word in doc:
            idx = idx_dict[word] if (word in vocab) else 0
            idxes_of_one_doc.append(idx)
        idxes.append(idxes_of_one_doc)

    inputs = np.asarray(idxes)
    labels = np.asarray(labels)
    return inputs, labels


def test():
    """仅用于简单测试此模块是否正常工作"""
    train_file = 'train_shuffle.txt'
    test_file = 'test_shuffle.txt'
    vocab = build_vocab(train_file)
    train_inputs, train_labels = generate_inputs_and_labels(train_file, vocab)
    test_inputs, test_labels = generate_inputs_and_labels(test_file, vocab)

    print('vocab size = ', len(vocab))
    print('train inputs shape: ', train_inputs.shape)
    print('test inputs shape: ', test_inputs.shape)


if __name__ == "__main__":
    test()
