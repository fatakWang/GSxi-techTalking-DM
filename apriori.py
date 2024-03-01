import pandas as pd
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt 
from matplotlib.font_manager import FontProperties
fonts = FontProperties(fname="word2evc/华文仿宋.ttf",size=14)
import networkx as nx
from pylab import mpl
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.frequent_patterns import association_rules
import random
f=open("association_relu/res.txt",mode="w+")

mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus']=False

def kw2lmh(i,keyword):
    if str_num_list[i][keyword]>0:
        return 1

def draw_graph(rules):
  plt.clf()
  plt.figure(figsize=(15, 6),dpi=200)
  G1 = nx.DiGraph()
   
  color_map=[]
  N = rules.shape[0]
  rules_to_show=N
  colors = np.random.rand(N)    
  strs=[ "R%d"%(i) for i in range(N)]   
   
   
  for i in range (rules_to_show):      
    G1.add_nodes_from(["R"+str(i)])
    
    for a in rules.iloc[i]['antecedents']:
                
        G1.add_nodes_from([a],fontproperties = fonts)
        
        G1.add_edge(a, "R"+str(i), color=colors[i] , weight = 2)
       
    for c in rules.iloc[i]['consequents']:
             
            G1.add_nodes_from([c],fontproperties = fonts)
            
            G1.add_edge("R"+str(i), c, color=colors[i],  weight=2)
 
  for node in G1:
       found_a_string = False
       for item in strs: 
           if node==item:
                found_a_string = True
       if found_a_string:
            color_map.append('yellow')
       else:
            color_map.append('green')       
 
 
  edges = G1.edges()
  colors = [G1[u][v]['color'] for u,v in edges]
  weights = [G1[u][v]['weight'] for u,v in edges]
 
  pos = nx.spring_layout(G1,k=N*10)
  nx.draw(G1, pos,  node_color = color_map, edge_color=colors, width=weights, font_size=16, with_labels=False)            
   
  for p in pos:  # raise text positions
           pos[p][1] += 0.07
  nx.draw_networkx_labels(G1, pos)


  plt.savefig("association_relu/graphimg/%s.jpg" % (csv_name[:-4] + "association_relu"))

  nx.draw_networkx


dir_name="./data/csv"
file_name=os.listdir(dir_name)

csvs={f:pd.read_csv(dir_name+"/"+f,encoding="gbk") for f in file_name}

for csv_name in file_name:
    f.write("====================%s=====================\n\n\n" % (csv_name))
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

    keyword_df=pd.DataFrame(keyword_dict,dtype=int)
    keyword_df.fillna(0,inplace=True)

    dropcols=[]
    for col in keyword_df.columns:
        # col="专业化"
        drop=True
        for row in keyword_df[col]:
            if row!=0:
                # print(row)
                drop=False
        if drop:
            dropcols.append(col)

    keyword_df.drop(columns=dropcols,axis=1,inplace=True)

    # 样本总数
    try:
        itemsetNum=keyword_df.shape[0]
        # 认为只要出现次数超过2次就是频繁出现的
        frequent_itemsets = fpgrowth(keyword_df, min_support=2/itemsetNum, use_colnames=True)
        # 关联规则挖掘
        rules=association_rules(frequent_itemsets, metric="confidence", min_threshold=0.8)
        # 按照lift排序
        rules.sort_values(by='lift',ascending=False, inplace=True)
        print(rules)

        rules=rules[:min(20,rules.shape[0])]
        f.write(rules.to_string())
        f.flush()
    except (ValueError,ZeroDivisionError):
        pass



