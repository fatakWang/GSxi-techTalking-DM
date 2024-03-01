import jieba
import re
import numpy as np
from sklearn.decomposition import PCA
import gensim
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import os 


f=open("Word2Vecres/res.txt",mode="w+")
dir_name="./data/csv"
file_name=os.listdir(dir_name)

csvs={f:pd.read_csv(dir_name+"/"+f,encoding="gbk") for f in file_name}

# 创建停用词列表
stopword_file="cn_stopwords.txt"
stopwords=[line.strip() for line in open(stopword_file, 'r',encoding='UTF-8').readlines()]


# 定义函数实现分词
def cutsentences(sentences,stopwords):
    temp = jieba.lcut(sentences)  #结巴分词 精确模式
    words = []
    for i in temp:
        #过滤掉所有的标点符号
        i = re.sub("[\s+\.\!\/_,$%^*(+\"\'””《》]+|[+——！，。？、~@#￥%……&*（）：；‘]+", "", i)
        if len(i) > 1 and i not in stopwords:
            words.append(i)
    if len(words) > 0:
        return words

for csv_name in file_name:
    f.write("====================%s=====================\n\n\n"%(csv_name))
    csv=csvs[csv_name]
    # 提取原句,并且分词
    sentence_list=[]

    for content in csv["原句"]:
        if pd.isnull(content):
            pass
        else:
            sentence_list.append(content)

    words_list=[]
    for sentence in sentence_list:
        words_list.append(cutsentences(sentence,stopwords))

    try:
        # 构造词向量
        # 调用Word2Vec训练
        # 参数：size: 词向量维度；window: 上下文的宽度，min_count为考虑计算的单词的最低词频阈值
        model=Word2Vec(words_list,epochs=50,min_count=0)
        try:
            # print("创新的词向量：\n",model.wv.get_vector('创新'))
            print("\n和创新相关性最高的前5个词语：")
            print(model.wv.most_similar('创新', topn = 5))# 与创新最相关的前20个词语
            f.write("\n和创新相关性最高的前5个词语：\n")
            f.write(str(model.wv.most_similar('创新', topn = 5)))
            f.write("\n")
        except KeyError:
            pass
        try:
            print("\n和科技相关性最高的前5个词语：")
            print(model.wv.most_similar('科技', topn = 5))# 与创新最相关的前20个词语
            f.write("\n和科技相关性最高的前5个词语：\n")
            f.write(str(model.wv.most_similar('科技', topn = 5)))
            f.write("\n")
        except KeyError:
            pass
        # 将词向量投影到二维空间
        rawWordVec = []
        word2ind = {}
        for i, w in enumerate(model.wv.index_to_key): #index_to_key 序号,词语
            rawWordVec.append(model.wv[w]) #词向量
            word2ind[w] = i #{词语:序号}
        rawWordVec = np.array(rawWordVec)
        X_reduced = PCA(n_components=2).fit_transform(rawWordVec)

        # 绘制几个特殊单词的向量
        try:
            innovate_knnwv=[l[0] for l in model.wv.most_similar('科技', topn = 5)]
        except KeyError:
            innovate_knnwv=[]
        try:
            tech_knnwv=[l[0] for l in model.wv.most_similar('创新', topn = 5)]
        except KeyError:
            tech_knnwv=[]
        words = set(innovate_knnwv+tech_knnwv)

        # 绘制星空图
        # 绘制所有单词向量的二维空间投影
        fig = plt.figure(figsize = (15,15))
        ax = fig.gca()
        ax.set_facecolor('white')
        ax.plot(X_reduced[:, 0], X_reduced[:, 1], '.', markersize = 10, alpha = 1, color = 'black')


        # 设置中文字体 否则乱码
        zhfont1 = matplotlib.font_manager.FontProperties(fname='word2evc/华文仿宋.ttf', size=16)
        for w in words:
            if w in word2ind:
                ind = word2ind[w]
                xy = X_reduced[ind]
                plt.plot(xy[0], xy[1], '.', alpha =1, color = 'orange',markersize=10)
                plt.text(xy[0], xy[1], w, fontproperties = zhfont1, alpha = 1, color = 'red')

        plt.title("5 nearnet word of tech and inno")
        plt.savefig("Word2Vecres/img/%s.jpg"%(csv_name[:-4] + "_" +"word2vec"))
        f.flush()
    except RuntimeError:
        pass
