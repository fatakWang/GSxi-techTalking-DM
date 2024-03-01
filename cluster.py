import pandas as pd
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import matplotlib.pyplot as plt 

def kw2lmh(i,keyword):
    return str_num_list[i][keyword]

f=open("cluster/res.txt",mode="w+")
dir_name="./data/csv"
file_name=os.listdir(dir_name)

csvs={f:pd.read_csv(dir_name+"/"+f,encoding="gbk") for f in file_name}

for csv_name in file_name:
    f.write("====================%s=====================\n\n\n"%(csv_name))
    csv=csvs[csv_name]
    # 提取分词结果、及其词频,构建一个分词到词频的词典
    str_list=[]
    num_list=[]
    strs=[]
    nums=[]
    for content,num in zip(csv["分词"],csv["词频"]):
        if pd.isnull(content):
            str_list.append(strs)
            num_list.append(nums)
            strs,nums=[],[]
        else:
            strs.append(content)
            nums.append(num)
    str_list.append(strs)
    num_list.append(nums)

    # print(str_list)
    # print(len(num_list))

    str_num_list=[]
    for strs,nums in zip(str_list,num_list):
        str_num={}
        for str_,num in zip(strs,nums):
            str_num[str_]=num
        str_num_list.append(str_num)

    # 提取关键词
    keyword_list=[]
    keywords=[]
    for content in csv["关键词(TF-IDF)"]:
        if pd.isnull(content):
            if len(keywords)!=0:
                # assert len(keywords)==15
                keyword_list.append(keywords)
                keywords=[]
        else:
            keywords.append(content)

    keyword_dict=[]
    for i,keywords in enumerate(keyword_list):
        keyword_dict.append({keyword:kw2lmh(i,keyword) for keyword in keywords})

    keyword_df=pd.DataFrame(keyword_dict)
    keyword_df.fillna(0,inplace=True)
    
    keyword_df.to_csv("cluster/%s"%(csv_name[:-3]+"csv"),index=None,mode="w+")
    cluster_raw=keyword_df


    try:
       # 聚类数从2到8，来产生聚类结果
       for n in range(2,min(9,cluster_raw.shape[0]+1)):
           f.write("====聚类数目为%d====\n\n"%(n))
           cluster=KMeans(n_clusters=n,n_init=100)
           cluster.fit(cluster_raw)
           # print(cluster.labels_)
           f.write("聚类结果："+str(cluster.labels_)+"\n")
           # print(cluster.inertia_)
           f.write("聚类损失："+str(cluster.inertia_) + "\n")
           # print(cluster.n_iter_)
           f.write("迭代轮数：" + str(cluster.n_iter_) + "\n")
           for i,cluster_center in  enumerate(cluster.cluster_centers_):
               # print("第%d个类:"%(i),cluster.feature_names_in_[cluster_center>0.5])
               f.write("第%d个类:"%(i)+str(cluster.feature_names_in_[cluster_center>1])+"\n")

           try:
               tsne = TSNE()
               tsne.fit_transform(cluster_raw)  # 进行数据降维,降成两维
               cluster_new=tsne.embedding_

               plt.clf()
               for cluster_num in range(cluster.cluster_centers_.shape[0]):
                   d = cluster_new[cluster.labels_ == cluster_num]
                   # print(d)
                   plt.scatter(d[:, 0], d[:, 1], label=cluster_num)

               plt.title("tsne")

               plt.legend(loc="best")
               plt.savefig("cluster/img/%s.jpg" % (csv_name[:-4] + "_" + str(n) + "_cluster"))
           except ValueError:
               pass

           plt.clf()
           f.flush()
    except ValueError:
        pass

