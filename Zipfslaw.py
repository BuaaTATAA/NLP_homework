import jieba
import re
import matplotlib.pyplot as plt
from collections import Counter
import os
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用于正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用于正常显示负号

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

def plot_zipf(word_counts):
    """绘制齐夫定律图"""
    ranks = range(1, len(word_counts) + 1)
    frequencies = [freq for _, freq in word_counts]
    #绘制原始数据的对数图
    plt.figure(figsize=(10, 6))
    plt.loglog(ranks, frequencies, marker=".", label = "实际数据")
    #拟合对数排名和对数频率之间的线性关系
    log_ranks = np.log(ranks)
    log_freqs = np.log(frequencies)
    coefficients = np.polyfit(log_ranks, log_freqs, 1)  # 1 表示线性拟合
    polynomial = np.poly1d(coefficients)
    fit_freqs = np.exp(polynomial(log_ranks))
    #绘制拟合曲线
    plt.loglog(ranks, fit_freqs, label="拟合曲线", linestyle='--', color='red')
    print(coefficients)
    plt.xlabel('Log of Rank')
    plt.ylabel('Log of Frequency')
    plt.title('Zipf\'s Law in CN-Corpus')
    plt.legend()
    plt.show()

# 加载文本数据
directory = 'H:\Desktop\DeepNlp\jyxstxtqj_downcc.com'
all_text = ""
for filename in os.listdir(directory):
    if filename.endswith('.txt'):  # 确保只读取.txt文件
        file_path = os.path.join(directory, filename)
        with open(file_path, 'r', encoding='ansi') as file:
            text = file.read()
            all_text += text

# 清理文本
cleaned_text = preprocess_text(all_text)
# 加载停用词列表.分词并过滤停用词
stopwords = load_stopwords('H:\Desktop\DeepNlp\stopwords.txt')
words = [word for word in jieba.cut(cleaned_text) if word not in stopwords]
# 统计词频
word_counts = Counter(words).most_common()
# 绘制齐夫定律图
plot_zipf(word_counts)