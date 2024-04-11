import math
from nltk.util import ngrams
from collections import Counter
import re
import jieba
import os
import numpy as np

def preprocess_text(text):
    """读取所有语料库信息，只保留中文字符，去除隐藏符号、标点符号等无用信息"""
    punctuation_pattern = r'[。，、；：？！（）《》【】“”‘’…—\-,.:;?!\[\](){}\'"<>]'
    text0 = re.sub(punctuation_pattern, '', text)
    text1 = re.sub(r'[\n\r\t]', '', text0)
    text2 = re.sub(r'[^\u4e00-\u9fa5]', '', text1)
    return text2

def load_stopwords(filepath):
    """读取停用词"""
    with open(filepath, 'r', encoding='utf-8') as file:
        stopwords = set([line.strip() for line in file.readlines()])
    return stopwords

'''加载文本数据'''
directory = 'H:\Desktop\DeepNlp\jyxstxtqj_downcc.com'
all_text = ""
for filename in os.listdir(directory):
    if filename.endswith('.txt'):  # 确保只读取.txt文件
        file_path = os.path.join(directory, filename)
        with open(file_path, 'r', encoding='ansi') as file:
            text = file.read()
            all_text += text

'''加载停用词列表.分字/词并过滤停用词'''
cleaned_text = preprocess_text(all_text)
stopwords = load_stopwords('H:\Desktop\DeepNlp\stopwords.txt')
words = [word for word in jieba.cut(cleaned_text) if word not in stopwords]
characters = [character for character in cleaned_text if character not in stopwords]

'''计算词unigrams, bigrams和trigrams的频率'''
unigrams = Counter(words)
bigrams = Counter(ngrams(words, 2))
trigrams = Counter(ngrams(words, 3))

'''计算字unigrams, bigrams和trigrams的频率'''
c_unigrams = Counter(characters)
c_bigrams = Counter(ngrams(characters, 2))
c_trigrams = Counter(ngrams(characters, 3))

'''计算一元词信息熵'''
word_entropy_unigram = 0
total_unigrams = sum(unigrams.values())
for count in unigrams.values():
    prob = count / total_unigrams
    word_entropy_unigram -= prob * math.log2(prob)

'''计算二元词信息熵'''
word_entropy_bigram = 0
total_bigrams = sum(bigrams.values())
for bi, bigram_count in bigrams.items():
    uni_count = unigrams[bi[0]]
    cond_prob = bigram_count / uni_count
    bigram_prob = bigram_count / total_bigrams
    word_entropy_bigram -= bigram_prob * math.log2(cond_prob)

'''计算一元字信息熵'''
char_entropy_unigram = 0
total_c_unigrams = sum(c_unigrams.values())
for count in c_unigrams.values():
    prob = count / total_c_unigrams
    char_entropy_unigram -= prob * math.log2(prob)

'''计算二元字信息熵'''
char_entropy_bigram = 0
total_c_bigrams = sum(c_bigrams.values())
for bi, bigram_count in c_bigrams.items():
    uni_count = c_unigrams[bi[0]]
    cond_prob = bigram_count / uni_count
    bigram_prob = bigram_count / total_c_bigrams
    char_entropy_bigram -= bigram_prob * math.log2(cond_prob)

'''计算三元词信息熵'''
entropy = 0
total_trigrams = sum(trigrams.values())
for tri, trigram_count in trigrams.items():
    bi = tri[:2]
    bigram_count = bigrams[bi]
    cond_prob = trigram_count / bigram_count
    trigram_prob = trigram_count / total_trigrams
    entropy -= trigram_prob * math.log2(cond_prob)

'''计算三元字信息熵'''
c_entropy = 0
c_total_trigrams = sum(c_trigrams.values())
for c_tri, c_trigram_count in c_trigrams.items():
    c_bi = c_tri[:2]
    c_bigram_count = c_bigrams[c_bi]
    c_cond_prob = c_trigram_count / c_bigram_count
    c_trigram_prob = c_trigram_count / c_total_trigrams
    c_entropy -= c_trigram_prob * math.log2(c_cond_prob)


print(f"一元字信息熵为: {char_entropy_unigram}")
print(f"一元词信息熵为: {word_entropy_unigram}")
print(f"二元字信息熵为: {char_entropy_bigram}")
print(f"二元词信息熵为: {word_entropy_bigram}")
print(f"三元字信息熵为: {c_entropy}")
print(f"三元词信息熵为: {entropy}")