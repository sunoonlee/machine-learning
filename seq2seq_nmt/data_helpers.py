from collections import Counter
from datetime import datetime
import re

import jieba
from nltk.stem.porter import PorterStemmer


buckets = [(11, 11), (15, 15)]
_PAD, _GO, _EOS, _UNK = '<PAD>', '<GO>', '<EOS>', '<UNK>' 
_PAD_ID, _GO_ID, _EOS_ID, _UNK_ID = 0, 1, 2, 3
_START_VOCAB = (_PAD, _GO, _EOS, _UNK)
STEMMER = PorterStemmer()


def en_tokenizer(sentence):
    """参: https://github.com/tensorflow/models/blob/master/tutorials/rnn/translate/data_utils.py"""
    _word_split = re.compile("([.,!?\"':;)(])")
    words = []
    for space_separated_fragment in sentence.strip().split():
        words.extend(_word_split.split(space_separated_fragment))
    return [STEMMER.stem(w) for w in words if w]


def zh_tokenizer(sentence):
    return [w for w in jieba.cut(sentence.strip()) 
            if w not in (' ', '\t', '\n',)]


def is_zh(word):
    """判断是否为中文"""
    return '\u4e00' <= word[0] <= '\u9fff'


def count_zh(sentence):
    """中文词数 in tokenized sentence"""
    return sum([int(is_zh(word)) for word in sentence])


def is_a_good_pair(enc_sentence, dec_sentence):
    """判断平行语料是否满足筛选条件"""
    return all((
        5 < len(enc_sentence) <= 15,
        5 < len(dec_sentence) <= 15,
        count_zh(dec_sentence) >= 4,
        not ({'年', '月', '日'} < set(dec_sentence)),
    ))


def full2half(text):
    """全角转换为半角"""
    result = ''
    for char in text:
        if '\uff01' <= char <= '\uff5e':
            char = chr(ord(char) - 0xfee0)
        result += char
    return result


def read_lines(filename, start_from, lines_to_read):
    lines = []
    with open(filename, encoding='utf-8') as f:
        count = 0
        for line in f:
            if count >= start_from:
                lines.append(line)
            count += 1
            if lines_to_read and (count >= start_from + lines_to_read - 1):
                return lines


def read_sentences(encoder_file, decoder_file, start_from=0, lines_to_read=None):
    """读取文件, 返回满足筛选条件的 tokenized sentences. 默认从英文到中文"""
    enc_lines = read_lines(encoder_file, start_from, lines_to_read)
    dec_lines = read_lines(decoder_file, start_from, lines_to_read)
    dec_lines = [full2half(line) for line in dec_lines]
    assert len(enc_lines) == len(dec_lines)

    encoder_sentences = []
    decoder_sentences = []
    
    count = 1
    for enc_line, dec_line in zip(enc_lines, dec_lines):
        try:
            enc_sentence = en_tokenizer(enc_line)
            dec_sentence = zh_tokenizer(dec_line)
            if is_a_good_pair(enc_sentence, dec_sentence):  # 筛选条件
                encoder_sentences.append(enc_sentence)
                decoder_sentences.append(dec_sentence)
        except IndexError:
            print('e', end='')
        
        if count % 10000 == 0:
            print(count, end='')
        elif count % 1000 == 0:
            print('=', end='')
        count += 1
        
    return encoder_sentences, decoder_sentences


def build_vocab(sentences, vocab_min_freq):
    """生成词表. 前几个位置留给 _START_VOCAB 的特殊 token """
    vocab = list(_START_VOCAB)
    words_flat = [w for s in sentences for w in s]
    word_cnt = Counter(words_flat)
    for word, count in word_cnt.most_common():
        if count >= vocab_min_freq:
            vocab.append(word)
    return vocab


def add_go_and_eos_tokens(sentences):
    """为 decoder 的输入语句增加首尾 token"""
    for i in range(len(sentences)):
        sentences[i] = [_GO] + sentences[i] + [_EOS]
    return sentences


def sentence2ids(sentence, length, word2id, pad_from_start=True):
    """tokens to indexes"""
    ids = [_PAD_ID] * length
    l = len(sentence)
    if l < length:
        if pad_from_start:
            ids[(length - l):] = [word2id.get(w, _UNK_ID) for w in sentence]
        else:
            ids[:l] = [word2id.get(w, _UNK_ID) for w in sentence]
    else:
        ids = [word2id.get(w, _UNK_ID) for w in sentence[:length]]
    return ids


def bucket_and_pad(enc_sentences, dec_sentences, buckets, word2id_enc, word2id_dec):
    """ to do """
    num_sentences = len(enc_sentences)
    
    encoder_data = [[] for _ in range(len(buckets))]
    decoder_data = [[] for _ in range(len(buckets))]
    
    # bucketing. 此时 decoder_sentences 已加首尾 token.
    for i in range(num_sentences):
        for bucket_id, (encoder_size, decoder_size) in enumerate(buckets):
            if len(enc_sentences[i]) <= encoder_size and len(dec_sentences[i]) <= decoder_size:
                encoder_data[bucket_id].append(
                    sentence2ids(enc_sentences[i], encoder_size, word2id_enc, True))
                decoder_data[bucket_id].append(
                    sentence2ids(dec_sentences[i], decoder_size, word2id_dec, False))
                break
    
    return encoder_data, decoder_data


def unk_proportion(sentence):
    """计算句子中 unk 的比例"""
    cnt = Counter(sentence)
    return cnt[_UNK_ID] / (sum(cnt.values()) - cnt[_PAD_ID])


def unk_proportions(sentences):
    """句子列表的 unk 比例"""
    return [unk_proportion(s) for s in sentences]


def main():
    print('nothing in main function yet...')


if __name__ == '__main__':
    main()