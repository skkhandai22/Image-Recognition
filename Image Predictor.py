#!/usr/bin/env python
# coding: utf-8

# In[50]:


import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import random
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


# In[51]:


dir='E:\\Verzeo Course\\Image Classifier\\PetImages'

categories=['Cat','Dog']

data=[]

for category in categories:
    path=os.path.join(dir,category)
    label=categories.index(category)
    
    for img in os.listdir(path):
        imgpath=os.path.join(path,img)
        pet_img=cv2.imread(imgpath,0)
        try:
            pet_img=cv2.resize(pet_img,(50,50))
            image=np.array(pet_img).flatten()
            
            data.append([image,label])
            
        except Exception as e:
            pass


pick_in=open('data.pickle','wb')
pickle.dump(data,pick_in)
pick_in.close()


# In[52]:


pick_in=open('data.pickle','rb')
data=pickle.load(pick_in)
pick_in.close()


# In[53]:


random.shuffle(data)
features=[]
labels=[]

for feature ,label in data:
    features.append(feature)
    labels.append(label)


# In[54]:


x_train,x_test,y_train,y_test=train_test_split(features,labels,test_size=0.01)
    
model=SVC(C=1,kernel='poly',gamma='auto')
model.fit(x_train,y_train)


# In[55]:


pick=open('medel.sav','wb')
pickle.dump(model,pick)
pick.close()


# In[56]:


prediction=model.predict(x_test)
accuracy=model.score(x_test,y_test)

categories=['Cat','Dog']

print('Accuracy: ',accuracy)
print('Prediction is : ',categories[prediction[0]])

mypet=x_test[0].reshape(50,50)
plt.imshow(mypet,cmap='gray')

plt.show()


# In[ ]:




