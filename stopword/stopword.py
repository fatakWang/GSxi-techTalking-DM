# -*- coding:utf-8 -*-
# 作者     ：Administrator
# 创建时间 ：2020/8/24 16:33
# 文件     ：stopword.py
# IDE      :PyCharm
# -*-coding:utf-8-*-


import os
"""
    合并文本文件
"""
mergefiledir = os.getcwd()+'\\stopwords-master'
filenames = os.listdir(mergefiledir)
file = open('stopwords.txt', 'w')

for filename in filenames:
    filepath = mergefiledir + '\\' + filename
    print(filepath)
    for line in open(filepath,"r",encoding="utf-8"):
        file.writelines(line)
    file.write('\n')

"""
    去重
"""
lines = open('stopwords.txt', 'r')
newfile = open('stopword.txt', 'w')
new = []
for line in lines.readlines():
    if line not in new:
        new.append(line)
        newfile.writelines(line)

file.close()
newfile.close()
