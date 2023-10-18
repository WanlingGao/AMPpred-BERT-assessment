import re
import random
import pandas as pd
import os.path as osp
tmp = ''
# 统计正样本数量
pp = 0
# 统计负样本数量
n = 0
# p使用正则表达式将fasta序列进行分词，其中数字决定分词方式
p = re.compile(r'(\w{1})(?!$)')

with open('./input.fasta', 'r') as f1:
        with open('./train_positive.tsv', 'w') as f2:
            for line in f1.readlines():
                line = line.strip()
                if len(line) == 0:
                    continue
                if line[0] == '>' and len(tmp) == 0:
                    continue
                if line[0] == '>' and len(tmp) > 0:
                    tmp += '\n'
                    res = "train\t1\t\t" + p.sub(r'\1 ', tmp)
                    f2.writelines(res)
                    tmp = ''
                    continue
                tmp += line

df_1=pd.read_csv('train_positive.tsv',header=None)
df_2=pd.read_csv('train_negative.tsv',header=None)
df=pd.concat([df_1,df_2],axis=0)
df_3=df.sample(frac=1,random_state=1)
df_3.to_csv('train.tsv',index=False)