#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ## 1. Data Loading

# In[3]:


df = pd.read_csv("E:/My projects/Python/Customer-Churn-Records.csv")
df.head()


# In[4]:


df.info()


# In[5]:


df.describe()


# In[6]:


df.dtypes


# In[7]:


df.isnull().sum()


# ## 2. Exploratory Data Analysis
# 
# 
# 

# Exploratory data analysis is performed mainly on data to generate insights, analyze data sets and summarize the main characteristics. Here, we will develop multiple visualizations to understand the data set more throughly.

# ### 1. Geography with most credit score

# In[9]:


Credit_Score = df.groupby("Geography")["CreditScore"].sum()
x= Credit_Score.index
y= Credit_Score.values

plt.figure(figsize= (11,6))

plt.bar(x,y, color ="skyblue", width = 0.4)
plt.gcf().axes[0].yaxis.get_major_formatter().set_scientific(False)
plt.xlabel('Credit_Score', fontsize=14, labelpad=10)
for  i, v in enumerate(Credit_Score):
    plt.text(i, v*1.02, str(v), horizontalalignment ='center',fontweight='bold', fontsize=10)


# ### 2. Age groups vs Avg Credit Score

# In[10]:


bins = [0,10,20,30,40,50,60,70,80,90,100]
labels = ["0-10","10-20","20-30","30-40","40-50","50-60","60-70","70-80","80-90","90-100"]
df["binned"] = pd.cut(df["Age"],bins= bins, labels=labels )
df.head(5)


# In[11]:


avg_cre_score = df.groupby("binned")["CreditScore"].mean().round()
avg_cre_score.dropna()

plt.figure(figsize=(11,6))
a= avg_cre_score.index
b=avg_cre_score.values
plt.bar(a,b,width = 0.9, color ="lightgreen")

for i,v in enumerate(avg_cre_score):
    plt.text(i,v*1.02, str(v),horizontalalignment ='center',fontweight='bold', fontsize=10)


# ### 3. Card type vs Age group

# In[17]:


card_type = df.groupby(["Card Type", "binned"]).size().reset_index(name="count").rename(columns={"binned": "bin"})

plt.figure(figsize=(11, 6))

sns.barplot(data=card_type, x="Card Type", y="count", hue="bin")

plt.xlabel("Card Type")
plt.ylabel("Count")
plt.title("Card Type Data by Bin")
plt.legend(title="Bin")

plt.show()


# ### 4. Churn by Gender

# In[16]:


churn_gender = df.groupby(["Gender", "Exited"]).size().reset_index(name="count")

plt.figure(figsize=(11, 6))

sns.barplot(data=churn_gender, x="Exited", y="count", hue="Gender")

for i, row in churn_gender.iterrows():
    x_coord = (i % 2) - 0.15 + (i // 2) * 0.3  # Adjusted x-coordinate
    plt.text(x_coord, row["count"] * 1.02, str(row["count"]), ha="center", fontweight='bold', fontsize=10)

plt.xlabel("Exited")
plt.ylabel("Count")
plt.title("Churn Data by Gender and Exit Status")
plt.legend(title="Gender")

plt.show()




# ### 5. Churn by Geography

# In[19]:


churn_geo = df.groupby(["Geography", "Exited"]).size().reset_index(name="count")

plt.figure(figsize=(11, 6))

sns.barplot(data=churn_geo, x="Exited", y="count", hue="Geography")

for i, row in churn_geo.iterrows():
    x_coord = (i % 2) - 0.20 + (i // 2) * 0.2  # Adjusted x-coordinate
    plt.text(x_coord, row["count"] * 1.02, str(row["count"]), ha="center", fontweight='bold', fontsize=10)

plt.xlabel("Exited")
plt.ylabel("Count")
plt.title("Churn Data by Geography and Exit Status")
plt.legend(title="Geography")

plt.show()


# ### 6. Churn Distribution

# In[20]:


churn = df.groupby("Exited")["CustomerId"].count()
colors = sns.color_palette('pastel')[0:5]

plt.pie(churn, colors = colors, autopct='%1.2f%%')
plt.show()


# ### 7. Gender distribution with Credit Card

# In[22]:


churn_creditcard =  df.groupby("Gender")["CustomerId"].count()
plt.figure(figsize=(11,6))
plt.pie(churn_creditcard, autopct= "%1.2f%%")
plt.title("Gender Distribution")
plt.show()


# ### 8. Correlation using HeatMap

# In[23]:


df_correlation = df.corr().round(2)
plt.figure(figsize=(11,6))
sns.heatmap(df_correlation, annot= True)
plt.title("Correlation Matrix")
plt.show()


# In[ ]:




