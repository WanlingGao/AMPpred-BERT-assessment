from bert_sklearn import BertClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
x_train=[]
with open(r'C:\x_train.txt')  as l:
    lines=l.readlines()
    for line in lines:
        line=line.rsplit()
        x_train.append(line)
y_train=[]
with open(r'C:\y_train.txt')  as l:
    lines=l.readlines()
    for line in lines:
        line=line.rsplit()
        y_train.append(line)
x_test=[]
with open(r'C:\x_test.txt')  as l:
    lines=l.readlines()
    for line in lines:
        line=line.rsplit()
        x_test.append(line)
y_test=[]
with open(r'C:\y_test.txt')  as l:
    lines=l.readlines()
    for line in lines:
        line=line.rsplit()
        y_test.append(line)
model = BertClassifier()
model.train_batch_size=16
model.eval_batch_size=16
model.learning_rate=2e-5
model.epochs=10
model.fit(x_train, y_train)
model.score(x_test,y_test)
pre=model.predict(x_test)
prod=model.predict_proba(x_test)[:,1]
# print(pre)
pre=pd.DataFrame(pre)
prod=pd.DataFrame(prod)
prod.to_csv('result.txt')
pre.to_csv('result.txt')
accuracy = accuracy_score(y_test, pre)
f1_score= f1_score(y_test,pre,pos_label='1')
precision=precision_score(y_test,pre,pos_label='1')
MCC=matthews_corrcoef(y_test,pre)
auc=roc_auc_score(y_test,prod)
confusion_matrix=confusion_matrix(y_test,pre)
TP=confusion_matrix[0][0]
FN=confusion_matrix[0][1]
TN=confusion_matrix[1][1]
FP=confusion_matrix[1][0]
Sn=TP/(TP+FN)
Sp=TN/(TN+FP)
print(f'Sensitivity: {Sn}')
print(f'Specificity: {Sp}')
print(f'Test Accuracy: {accuracy}')
print(f'Test f1_score: {f1_score}')
print(f'Test precision_score: {precision}')
print(f'Test matthews_corrcoef: {MCC}')
print(f'Test roc_auc_score: {auc}')
model.save('bert(9.11fold5).bin')


