#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import sqlite3
import pandas as pd
import streamlit as st


# In[4]:


def getpath():
    ## get the path of prediction_data file
    cwd = os.getcwd()
    cw='/'.join(cwd.split('/'))
    if cw:
        path=cwd+'/prediction_data/games_fps_cpu_gpu.db'
    else:
        path=cwd+'\\prediction_data\\games_fps_cpu_gpu.db'
    return path


# In[5]:


path=getpath()
query='''SELECT DISTINCT Game_Name FROM games_fps'''
cnx=sqlite3.connect(path)
games=pd.read_sql(query,cnx)
cnx.close()


# In[6]:
st.title('Please select games and budget')

game1=st.selectbox('Please select the first game you want to play',games['Game_Name'],key=1)
print('You selected: ', game1)

game2=st.selectbox('Please select the second game you want to play',games['Game_Name'],key=2)
print('You selected: ', game2)

game3=st.selectbox('Please select the third game you want to play',games['Game_Name'],key=3)
print('You selected: ', game3)


# In[7]:


budget=st.text_input('Please input the budget you have in mind')


# In[36]:


path=getpath()
query='''SELECT a.CPU, a.GPU, b.PRICE+c.PRICE  as total_price, min(a.FPS) as Lowest_FPS FROM games_fps as a, cpu_price as b,      gpu_price as c  where a.CPU=b.CPU and a.GPU=c.GPU       and (a.Game_Name='{}' or a.Game_Name='{}' or a.Game_Name='{}')  and total_price<({}+0)  GROUP BY a.CPU,a.GPU ORDER BY Lowest_FPS DESC'''.format(game1,game2,game3,budget)
cnx=sqlite3.connect(path)
result=pd.read_sql(query,cnx)
cnx.close()


# In[38]:

st.title('FPS demonstrated at 1080P ultra setting')
st.dataframe(result.iloc[:5,:])

