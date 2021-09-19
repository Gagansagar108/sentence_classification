#!/usr/bin/env python
# coding: utf-8

# In[366]:


import os
import re
import fitz
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten,Embedding 
from tensorflow.keras import Sequential
import seaborn as sns
from nltk.tokenize import sent_tokenize,word_tokenize 


# In[530]:


path_h='E:\GAGAN_ASSIGNMENT\ASSIGNMENT\HISTORY'
history=''

files=os.listdir(path_h)
for file in files:
    f_path=os.path.join(path_h,file)
    pdf=fitz.open(f_path)
    pages=pdf.page_count
    for i in range(pages):
        page_text=pdf.load_page(i)
        text=page_text.get_text()
        history=history+text
        
        
    


# In[ ]:





# In[433]:





# In[531]:


science=' ' 
path_h='E:\GAGAN_ASSIGNMENT\ASSIGNMENT\SCIENCE'

files=os.listdir(path_h)
for file in files:
    f_path=os.path.join(path_h,file)
    pdf=fitz.open(f_path)
    pages=pdf.page_count
    for i in range(pages):
        page_text=pdf.load_page(i)
        text=page_text.get_text()
        science=science+text
        
        


# In[532]:


def cleaned(text):
    cleaned_text=re.sub('/(\r\n)+|\r+|\n+|\t+/','',text)       #removes lines 
    cleaned_text=re.sub('(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w \.-]*)','',cleaned_text)  #removes links
    cleaned_text=re.sub('\d+\.','',cleaned_text)       #removes question number
    cleaned_text=re.sub('\(?[a-zA-Z0-9]+?\)','',cleaned_text)   #removes brackets 
    cleaned_text=re.sub('\[?[a-zA-Z0-9]+?\]','',cleaned_text)  #removes brackets 
    cleaned_text=re.sub('\s[a-zA-Z0-9]\s','',cleaned_text)   #removes single characters 
    return cleaned_text
science=cleaned(science.lower())
history=cleaned(history.lower())


# In[533]:



history=sent_tokenize(history)    #sentence tokenizer
science=sent_tokenize(science)   #sentence tokenizer

h_label=['h']*len(history)     #marks lables
s_label=['S']*len(science)


# In[ ]:





# In[534]:


len(science),len(s_label),len(h_label),len(history)


# In[535]:


features=np.concatenate([science,history])   #features 
labels=np.concatenate([s_label,h_label])  #labels 


# In[536]:


data=pd.DataFrame({'Question':features,'Class':labels})
data= data.sample(frac = 1)


# In[539]:


data.head()


# In[538]:


data.iloc[10:18,]


# In[540]:


y=data['Class'].astype('category').cat.codes.values            #convers classes into codes 
y[:10]


# In[541]:


maxlen=max(list(map(lambda x : len(x.split(' ')),features)))    #max word in a sentence it will 
maxlen


# Method1 :word embedding with keras embedded layer

# In[542]:


encoded= [ one_hot(x,600) for x in data['Question']]
vectors_features=pad_sequences(encoded,maxlen=maxlen,padding='post')
print(vectors_features.shape)
vectors_features


# In[543]:


X_train ,X_test,y_train,y_test=train_test_split(vectors_features,y,test_size=0.2,random_state=200)     
X_train.shape,y_train.shape,X_test.shape,y_test.shape


# implementing the models  

# In[544]:


vocab_size=600
embeded_vector_size=100
model_cnn=Sequential()                
model_cnn.add(Embedding(vocab_size,embeded_vector_size,input_length=maxlen))
model_cnn.add(Flatten())
model_cnn.add(Dense(128,activation='relu'))
model_cnn.add(Dense(32,activation='relu'))
model_cnn.add(Dense(1,activation='sigmoid'))


# In[545]:


model_cnn.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model_cnn.fit(X_train,y_train,epochs=50)
loss,acc_cnn = model_cnn.evaluate(X_test, y_test,batch_size=32, verbose=0)
loss,acc_cnn


# In[ ]:





# Support Vector Classifier
# 

# In[546]:


svc=SVC()              
model_svc=svc.fit(X_train,y_train)
pred_svc=model_svc.predict(X_test)
acc_svc=accuracy_score(y_test,pred_svc)
print('SVC:',acc_svc)


# Logistic Regression 

# In[547]:


Lgr=LogisticRegression()
model_lgr=Lgr.fit(X_train,y_train)
pred_lgr=model_lgr.predict(X_test)
acc_lgr=accuracy_score(y_test,pred_lgr)
print('LogisticRegression:',acc_lgr)


# KNeighborsClassifier

# In[553]:


knn=KNeighborsClassifier()
model_knn=knn.fit(X_train,y_train)
pred_knn=model_knn.predict(X_test)
acc_knn=accuracy_score(y_test,pred_knn)
print('KNN:',acc_knn)


# In[552]:


DTclf = DecisionTreeClassifier()
model_DTclf = DTclf.fit(X_train,y_train)
pred_DTclf = model_DTclf.predict(X_test)
acc_DTclf=accuracy_score(y_test,pred_DTclf)
print('DecisionTreeClassifier:',acc_DTclf)


# In[550]:


gnb = GaussianNB()
model_gnb=gnb.fit(X_train, y_train)
pred_gnb=model_gnb.predict(X_test)
acc_gnb=accuracy_score(y_test,pred_gnb)
print('GaussianNB:',acc_gnb)


# In[ ]:


X_train.shape,y_train.shape,X_test.shape,y_test.shape


# In[554]:


plt_y=[acc_knn,acc_lgr,acc_svc, acc_cnn,acc_gnb,acc_DTclf]
plt_x=['KNN','LGR','SVC', 'CNN','GaussianNB','DecisionTreeClassifier']

x_axis=range(len(plt_x))
plt.figure(figsize=(15,6))
ax=sns.barplot(x=plt_x,y=plt_y,alpha=0.7)
ax.set_ylim([0,1])
for x,y in zip(x_axis,plt_y):
    ax.text(x,y+.05,round(y,2))
ax.set_box_aspect(3.5/len(ax.patches))
ax.set(xlabel='models', ylabel='accuracy ')


# In[ ]:





# METHOD 2

#  Count Vectors as features

# In[ ]:





# In[555]:


from sklearn.feature_extraction.text import CountVectorizer  
cv=CountVectorizer(max_features=160)             ##convert sentences into vector representation 
x2=cv.fit_transform(data['Question']).toarray()         #return array of vectors 
y2=data['Class'].astype('category').cat.codes.values            #convers classes into codes 


# In[ ]:


x


# In[556]:


x2.shape,y2.shape


# In[557]:



X_train2 ,X_test2,y_train2,y_test2=train_test_split(x2,y2,test_size=0.2,random_state=2)       


# SUPPOR VECTOR CLASSIFIER

# In[558]:


svc=SVC()
model_svc2=svc.fit(X_train2,y_train2)
pred_svc2=model_svc2.predict(X_test2)
acc_svc2=accuracy_score(y_test2,pred_svc2)
acc_svc2


# RandomForestClassifier

# In[559]:



rfc2=RandomForestClassifier()
model_rfc2=rfc2.fit(X_train2,y_train2)
pred_rfc2=model_rfc2.predict(X_test2)
acc_rfc2=accuracy_score(y_test2,pred_rfc2)
acc_rfc2


# KNeighborsClassifier

# In[560]:


from sklearn.neighbors import KNeighborsClassifier
knn2=KNeighborsClassifier()
model_knn2=knn2.fit(X_train2,y_train2)
pred_knn2=model_knn2.predict(X_test2)
acc_knn2=accuracy_score(y_test2,pred_knn2)
acc_knn2


# In[561]:


from sklearn.linear_model import LogisticRegression 
Lgr2=LogisticRegression()
model_lgr2=Lgr2.fit(X_train2,y_train2)
pred_lgr2=model_lgr2.predict(X_test2)
acc_lgr2=accuracy_score(y_test2,pred_lgr2)
acc_lgr2


# In[562]:


from  tensorflow.keras import layers,Sequential
from tensorflow import keras
model_cnn2=Sequential()
model_cnn2.add(Dense(128,activation='relu'))
model_cnn2.add(Dense(32,activation='relu'))
model_cnn2.add(Dense(1,activation='sigmoid'))
model_cnn2.compile(optimizer ='adam',loss ='binary_crossentropy',metrics =['accuracy'])


# In[563]:


model_cnn2.fit ( X_train2 , y_train2 , epochs =20)

loss2,acc_cnn2 = model_cnn2.evaluate(X_test2, y_test2, verbose=0)

loss2,acc_cnn2


# In[564]:


gnb2 = GaussianNB()
model_gnb2=gnb2.fit(X_train2, y_train2)
pred_gnb2=model_gnb2.predict(X_test2)
acc_gnb2=accuracy_score(y_test2,pred_gnb2)
print('gnb2:',acc_gnb2)


# In[565]:


DTclf2 = DecisionTreeClassifier()
model_DTclf2 = DTclf2.fit(X_train2,y_train2)
pred_DTclf2 = model_DTclf2.predict(X_test2)
acc_DTclf2=accuracy_score(y_test2,pred_DTclf2)
print('DTclf2:',acc_DTclf2)


# In[566]:


plt_y2=[acc_knn2,acc_lgr2,acc_svc2, acc_cnn2,acc_gnb2,acc_DTclf2]
plt_x2=['KNN','LGR','SVC', 'CNN','GaussianNB','DecisionTreeClassifier']
x_axis2=range(len(plt_x2))
plt.figure(figsize=(15,6))
ax2=sns.barplot(x=plt_x2,y=plt_y2,alpha=0.7)
ax2.set_ylim([0,1])
for x,y in zip(x_axis2,plt_y2):
    ax.text(x,y+.05,round(y,2))
ax2.set_box_aspect(3.5/len(ax.patches))
ax2.set(xlabel='models', ylabel='accuracy ')


# Comparing Model 1 and model 2

# In[567]:


model_acc=pd.DataFrame({'Model1':plt_y,'Model2':plt_y2},index=plt_x)


# In[568]:


model_acc


# In[569]:


plt.figure(figsize=(10,6))
x_axis=np.arange(len(model_acc.index))+.35
plt.bar(x_axis,model_acc.Model1,width=0.35,color='olive')
plt.bar(x_axis+.35,model_acc.Model2,width=0.35,color='darkorange')
plt.xticks(x_axis,plt_x)

plt.legend(['Model1','Model2'])
plt.show()


# Model Accuracy 

# model 2 doing comparatively  better 

# In[320]:





# In[320]:





# In[320]:





# In[320]:





# In[320]:





# In[ ]:





# APPLYING models on another data

# In[515]:


t=open(r'E:\Downloads\questions.txt')


# In[516]:


lines=t.readlines()


# In[517]:


features=[]
labels=[]
for x in lines:
    features.append(x.split(':')[1][:-3])
    labels.append(x.split(':')[0])


# In[518]:


X=cv.fit_transform((features)).toarray()       #converting sentences into vectors 


# In[519]:


df=pd.DataFrame(np.array(X))
df


# In[520]:


df['class']=labels


# In[521]:


df['class']=df['class'].astype('category').cat.codes


# In[522]:


x=df.iloc[:,:-1]


# In[523]:


y=df['class'].values
y


# In[524]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x = scaler.fit_transform(x)                   


# In[525]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
X_train ,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


# In[526]:


from sklearn.linear_model import LogisticRegression 
Lgr=LogisticRegression()
model=Lgr.fit(X_train,y_train)
pred=model.predict(X_test)
acc=accuracy_score(y_test,pred)
acc


# In[ ]:





# In[527]:


from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()
model=knn.fit(X_train,y_train)
pred=model.predict(X_test)
acc=accuracy_score(y_test,pred)
acc


# In[528]:


from sklearn import svm
from sklearn.svm import SVC
svc=SVC()
model=svc.fit(X_train,y_train)
pred=model.predict(X_test)
acc=accuracy_score(y_test,pred)
acc


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




