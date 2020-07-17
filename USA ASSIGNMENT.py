#!/usr/bin/env python
# coding: utf-8

# In[1]:


cd Desktop


# In[19]:


import matplotlib.pyplot as plt
import pandas as pd
data = pd.read_csv('USA_Housing.csv')


# In[20]:


data.describe()


# In[21]:


data.columns


# In[22]:


X = data[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
       'Avg. Area Number of Bedrooms', 'Area Population']]


# In[23]:


y = data['Price']


# In[80]:


print(y)


# In[82]:


print(X)


# In[83]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)


# In[84]:


from sklearn.linear_model import LinearRegression


# In[85]:


lm = LinearRegression()


# In[86]:


lm.fit(X_train, y_train)


# In[87]:


print(lm.intercept_)


# In[78]:


predictions = lm.predict(X_test)


# In[67]:


print(predictions)


# In[68]:


accuracy = lm.score(X_test,y_test)


# In[69]:


print(accuracy)


# In[90]:


plt.scatter(y_test, predictions, edgecolor='black')


# In[93]:


print("The accuracy of the Model:",accuracy)
AvgAreaIncome = float(input("Enter Avg. Area Income:"))
AvgAreaHouseAge = float(input("Enter Avg. Area House Age:"))
AvgAreaNumberofRooms = float(input("Enter Avg. Area Number of Rooms"))
AvgAreaNumberofBedrooms = float(input("Enter Avg. Area Number of Bedrooms:"))
AreaPopulation = float(input("Enter Area Population:"))
list1 = [[AvgAreaIncome,AvgAreaHouseAge,AvgAreaNumberofRooms,AvgAreaNumberofBedrooms,AreaPopulation]]
x_new = pd.DataFrame(list1,columns=['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
       'Avg. Area Number of Bedrooms', 'Area Population'])
y_new = lm.predict(x_new)
print(y_new)


# In[ ]:




