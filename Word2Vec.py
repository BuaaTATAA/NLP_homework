# -*- coding: utf-8 -*-
import os
import jieba
import re
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

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

def cut_text(text, stopwords):
    words = jieba.cut(text)
    return [word for word in words if word not in stopwords and word.strip()]

def get_paragraph_vector(paragraph, model):
    # 分词并去除停用词
    words = [word for word in jieba.cut(paragraph) if word not in stopwords]
    # 计算词向量
    word_vectors = [model.wv[word] for word in words if word in model.wv]
    # 取平均值作为段落向量
    if len(word_vectors) > 0:
        return np.mean(word_vectors, axis=0)
    else:
        return np.zeros(model.vector_size)

# 计算段落间相似度
def calculate_similarity(paragraph1, paragraph2, model):
    vector1 = get_paragraph_vector(paragraph1, model)
    vector2 = get_paragraph_vector(paragraph2, model)
    if np.all(vector1 == 0) or np.all(vector2 == 0):
        return 0  # 若有一个段落无有效向量，则相似度为0
    else:
        return cosine_similarity([vector1], [vector2])[0][0]

'''加载文本数据'''
corpus = []
directory = 'H:\Desktop\DeepNlp\data'
for filename in os.listdir(directory):
    if filename.endswith('.txt'):  # 确保只读取.txt文件
        file_path = os.path.join(directory, filename)
        with open(file_path, 'r', encoding='ansi') as file:
            text = file.read()
            cleaned_text = preprocess_text(text)
            corpus.append(cleaned_text)

'''加载停用词列表.分字/词并过滤停用词'''
stopwords = load_stopwords('H:\Desktop\DeepNlp\stopwords.txt')
processed_corpus = [cut_text(text, stopwords) for text in corpus]

'''训练Word2Vec模型 '''
model = Word2Vec(sentences=processed_corpus, vector_size=100, window=5, min_count=5, workers=16, epochs=50)

'''保存模型'''
model.save("jin_yong_word2vec.model")


'''加载模型'''
model = Word2Vec.load("jin_yong_word2vec.model")

'''查询词之间的相似度'''
word1 = "杨过"
word2 = "小龙女"
similarity_score = model.wv.similarity(word1, word2)
print(f"词语 '{word1}' 和 '{word2}' 的相似度得分为：{similarity_score}")

'''KMeans聚类'''
# 获取词汇表中的所有词语及其向量
words = list(model.wv.index_to_key)
word_vectors = np.array([model.wv[word] for word in words])

# 使用KMeans算法进行聚类
num_clusters = 10  # 设定要分的簇数
kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(word_vectors)

# 获取每个词语对应的簇标签
labels = kmeans.labels_

# 获取 Cluster 1 的词向量和索引
cluster_index = 1
cluster_indices = [index for index, label in enumerate(labels) if label == cluster_index]
cluster_words = [words[index] for index in cluster_indices]
cluster_vectors = word_vectors[cluster_indices]

# 使用t-SNE进行降维
tsne = TSNE(n_components=2, random_state=0)
cluster_vectors_2d = tsne.fit_transform(cluster_vectors)

# 绘制散点图
plt.figure(figsize=(8, 8))
plt.scatter(cluster_vectors_2d[:, 0], cluster_vectors_2d[:, 1], label=f'Cluster {cluster_index}')


plt.legend()
plt.title(f'Word Clustering Visualization for Cluster {cluster_index}')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.show()

# 打印每个簇中的词语
clusters = {}
for i in range(num_clusters):
    clusters[i] = []

for word, label in zip(words, labels):
    clusters[label].append(word)

for cluster_id, cluster_words in clusters.items():
    print(f"Cluster {cluster_id}: {', '.join(cluster_words)}")

# 示例段落
paragraph1 = "颜烈跨出房门，只见过道中一个　　颜烈跨出房门，只见过道中一个中年士人拖着鞋皮，踢踏踢踏的直响，一路打着哈欠迎面过来。那士人似笑非笑，挤眉弄眼，一副惫懒神气，全身油腻，衣冠不整，满脸污垢，看来少说也有十多天没洗脸了，拿着一柄破烂的油纸黑扇，边摇边行。颜烈见这人衣着明明是个斯文士子，却如此肮脏，不禁皱了眉头，加快脚步，只怕沾到了那人身上的污秽。突听那人干笑数声，声音甚是刺耳，经过他身旁时，顺手伸出折扇，在他肩头一拍。颜烈身有武功，这一下竟没避开，不禁大怒，喝道：“干什么？”那人又是一阵干笑，踢踏踢踏的向前去了，只听他走到过道尽头，对店小二道：“喂，伙计啊，你别瞧大爷身上破破烂烂的，大爷可有的是银子。有些小子可邪着哪，他就是仗着身上光鲜吓人。招摇撞骗，勾引妇女，吃白食，住白店，全是这种小子，你得多留点儿神。稳稳当当的，让他先交了房饭钱再说。”也不等那店小二答腔，又是踢踏踢踏的走了。颜烈更是心头火起，心想好小子，这话不是冲着我来么？店小二听那人一说，斜眼向他看了一眼，不禁起疑，走到他跟前，哈了哈腰，陪笑道：“您老别见怪，不是小的无礼……”颜烈知他意思，哼了一声道：“把这银子给存在柜上！”伸手往怀里一摸，不禁呆了。他囊里本来放着四五十两银子，一探手，竟已空空如也。店小二见他脸色尴尬，只道穷酸的话不错，神色登时不如适才恭谨，挺腰凸肚的道：“怎么？没带钱么？”颜烈道：“你等一下，我回房去拿。”他只道匆匆出房，忘拿银两，那知回入房中打开包裹一看，包里几十两金银竟然尽皆不翼而飞。这批金银如何失去，自己竟是茫然不觉，那倒奇了，寻思：“适才包氏娘子出去解手，我也去了茅房一阵，前后不到一柱香时分，怎地便有人进房来做了手脚？嘉兴府的飞贼倒是厉害。”店小二在房门口探头探脑的张望，见他银子拿不出来，发作道：“这女娘是你原配妻子吗？要是拐带人口，可要连累我们呢！”包惜弱又羞又急，满脸通红。颜烈一个箭步纵到门口，反手一掌，只打得店小二满脸是血，还打落了几枚牙齿。店小二捧住脸大嚷大叫：“好哇！住店不给钱，还打人哪！”颜烈在他屁股上加了一脚，店小二一个筋斗翻了出去。包惜弱惊道：“咱们快走吧，不住这店了。”颜烈笑道：“别怕，没了银子问他们拿。”端了一张椅子坐在房门口头。过不多时，店小二领了十多名泼皮，抡棒使棍，冲进院子来。颜烈哈哈大笑，喝道：“你们想打架？”忽地跃出，顺手抢过一根杆棒，指东打西，转眼间打倒了四五个，那些泼皮平素只靠逞凶使狠，欺压良善，这时见势头不对，都抛下棍棒，一窝蜂的挤出院门，躺在地下的连爬带滚，唯恐落后。包惜弱早已吓的脸上全无血色，颤声道：“事情闹大了，只怕惊动了官府。”颜烈笑道：“我正要官府来。”包惜弱不知他的用意，只得不言语了。过不半个时辰，外面人声喧哗，十多名衙役手持铁尺单刀，闯进院子，把铁链抖的当啷当啷乱响，乱嘈嘈的叫道：“拐卖人口，还要行凶，这还了得？凶犯在那里？”颜烈端坐椅上不动。众衙役见他衣饰华贵，神态俨然，倒也不敢贸然上前。带头的捕快喝道：“喂，你叫什么名字？到嘉兴府来干什么？”颜烈道：“你去叫盖运聪来！”盖运聪是嘉兴府的知府，众衙役听他直斥上司的名字，都是又惊又恐。那捕快道：“你失心疯了么？乱呼乱叫盖大爷。”颜烈从怀里取出一封信来，往桌上一掷，抬头瞧着屋顶，说道：“你拿去给盖运聪瞧瞧，看他来是不来？”那捕快取信件，见了封皮上的，吃了一惊，但不知真伪，低声对众衙役道：“看着他，别让他跑了。”随即飞奔而出。包惜弱坐在房中，心里怦怦乱跳，不知吉凶。"
paragraph2 = "完颜洪烈眼前一花，只见一个道人手中托了一口极大的铜缸，迈步走上楼来，定睛看时，只吓得心中突突乱跳，原来这道人正是长春子丘处机。完颜洪烈这次奉父皇之命出使宋廷，要乘机阴结宋朝大官，以备日后入侵时作为内应。陪他从燕京南来的宋朝使臣王道乾趋炎附势，贪图重贿，已暗中投靠金国，到临安后替他拉拢奔走。那知王道乾突然被一个道人杀死，连心肝首级都不知去向。完颜洪烈大惊之余，生怕自己阴谋已被这道人查觉，当即带同亲随，由临安府的捕快衙役领路，亲自追拿刺客。追到牛家村时与丘处机遭遇，不料这道人武功高极，完颜洪烈尚未出手，就被他一甩手箭打中肩头，所带来的衙役随从被他杀的干干净净。完颜洪烈如不是在混战中先行逃开，又得包惜弱相救，堂堂金国王子就此不明不白的葬身在这小村之中了。完颜洪烈定了定神，见他目光只在自己脸上掠过，便全神贯注的瞧着焦木和那七人，显然并未认出自己，料想那日自己刚探身出来，便给他羽箭掷中摔倒，并未看清楚自己面目，当即宽心，再看他手中托的那口大铜缸时，一惊之下，不由得欠身离椅。"
# 计算段落相似度
similarity_score = calculate_similarity(paragraph1, paragraph2, model)
print(f"段落1与段落2的语义相似度：{similarity_score}")





