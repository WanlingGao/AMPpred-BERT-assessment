# integrating model

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import confusion_matrix
import sklearn.svm as svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
df_1=pd.read_csv('AMP-BERT/fold 1/prob.txt',header=None)
df_2=pd.read_csv('Bert-Protein/fold 1/prob.txt',header=None)
df_3=pd.read_csv('cAMPs_pred/fold 1/prob.txt',header=None)
df_3=df_3.iloc[:,1]
df_4=pd.read_csv('LMPred/fold 1/prob.txt',header=None)
label=pd.read_csv('train/1fold/y_test.csv',header=None)
df=pd.concat([df_1,df_2,df_3,df_4,label],axis=1,ignore_index=False)
df.columns=['0','1','2','3','4']
X=df.iloc[:,:4]
y=df.iloc[:,4]
# x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
x_train=df.iloc[:537,[0,3]]
x_test=df.iloc[537:,[0,3]]
y_train=df.iloc[:537,4]
y_test=df.iloc[537:,4]
model = svm.SVC(C=1,kernel='rbf',gamma='auto')
model.fit(x_train,y_train)
pred = model.predict(x_test)
accuracy = accuracy_score(y_test, pred)
f1_score= f1_score(y_test,pred)
precision=precision_score(y_test,pred)
MCC=matthews_corrcoef(y_test,pred)
confusion_matrix=confusion_matrix(y_test,pred)
TP=confusion_matrix[0][0]
FN=confusion_matrix[0][1]
TN=confusion_matrix[1][1]
FP=confusion_matrix[1][0]
Sn=TP/(TP+FN)
Sp=TN/(TN+FP)
print(f'Sensitivity: {Sn:.4f}')
print(f'Specificity: {Sp:.4f}')
print(f'Test precision_score: {precision:.4f}')
print(f'Test Accuracy: {accuracy:.4f}')
print(f'Test f1_score: {f1_score:.4f}')
print(f'Test matthews_corrcoef: {MCC:.4f}')
