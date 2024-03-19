#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np 
arr=np.random.randint(10,size=9)
arr.reshape(3,3)


# <h6>Создала массив рандомных чисел 3*3<h6>

# <h6>Разделяю на массивы по три элемента<h6>

# In[7]:


arr2=np.split(arr,3)
arr2


# In[ ]:




