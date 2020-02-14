#!/usr/bin/env python
# coding: utf-8

# The goal of this part is to fit the model to a sparse matrix with some games' FPS are missing. 
# The key part is to compute the MSE with some element of F representing the fps data is missing.

# ## Building a model to predict FPS
# First, we build our modle as:
# \begin{align} 
# F^{i}_{mn}=g^{i}P_{mn}+\alpha_{mn}
# \end{align}
# where $i$ is the label for games, $mn$ are the label for gpu and cpu respectively, and $\alpha$ contains other information that is game independent.
# 
# Next, I will use the current data to testify this model.

# ## Building a new model to predict FPS
# First, we build our modle as:
# \begin{align} 
# F^{i}_{mn}=g^{i}G_{m}C_{n}
# \end{align}
# where $i$ is the label for games, $mn$ are the label for gpu and cpu respectively, and $\alpha$ contains other information that is game independent.
# 
# Next, I will use the current data to testify this model.

# The number of parameters in this model is $i+m+n$.
# Because we find 'i' games fps benchmark, the number of data point will be $i*m*n$.
# In the case where $i=24,m=28,n=14$,the number of parameters are $66$ and the number of data points is $9408$.

# import pandas as pd
# import numpy as np
# import tensorflow as tf
# import matplotlib.pyplot as plt
# import random

# # The following cell is the model class. 
# Its __call__ method returns the predicted FPS according to aformentioned formula.
# Its load_variables method loads previously trained parameters which will be used by the __call__ method to make predictions.

# In[1]:


import tensorflow as tf
import random


# In[2]:


## i is the total number of games, m is the total number of GPUs considered,
## and n is th total number of CPUs considered.
class model():
    def __init__(self,shape):
        self.i=shape[0]
        self.m=shape[1]
        self.n=shape[2]
        self.g=tf.Variable(tf.random.truncated_normal(shape=(self.i,)))
        self.P=tf.Variable(tf.random.truncated_normal(shape=(self.m,self.n)))
        self.alpha=tf.Variable(tf.random.truncated_normal(shape=(self.m,self.n)))
        self.trainable_variables=[self.P,self.alpha,self.g]
        
    def __call__(self):
        F_predict=tf.concat([tf.expand_dims(self.g[j]*self.P,0) for j in range(self.i)],0)                    +tf.tile(tf.expand_dims(self.alpha,0),[self.i,1,1])
        return F_predict
    
    def load_variables(self,parameters):
        self.P=tf.constant(parameters[0])
        self.alpha=tf.constant(parameters[1])
        self.g=tf.constant(parameters[2])


# In[3]:


## model without alpha
class model_without_alpha():
    def __init__(self,shape):
        self.i=shape[0]
        self.m=shape[1]
        self.n=shape[2]
        self.g=tf.Variable(tf.random.truncated_normal(shape=(self.i,)))
        self.P=tf.Variable(tf.random.truncated_normal(shape=(self.m,self.n)))
        self.trainable_variables=[self.P,self.g]
        
    def __call__(self):
        F_predict=tf.concat([tf.expand_dims(self.g[j]*self.P,0) for j in range(self.i)],0)
        return F_predict
    
    def load_variables(self,parameters):
        self.P=tf.constant(parameters[0])
        self.g=tf.constant(parameters[2])


# In[4]:


## model that also decomposes GPU and CPU
class model_cpu_gpu():
    def __init__(self,shape):
        self.i=shape[0]
        self.m=shape[1]
        self.n=shape[2]
        self.g=tf.Variable(tf.random.truncated_normal(shape=(self.i,)))
        self.G=tf.Variable(tf.random.truncated_normal(shape=(self.m,)))
        self.C=tf.Variable(tf.random.truncated_normal(shape=(self.n,)))
        self.trainable_variables=[self.G,self.C,self.g]
        
    def __call__(self):
        P=tf.concat([tf.expand_dims(self.G[j]*self.C,0) for j in range(self.m)],0)
        F_predict=tf.concat([tf.expand_dims(self.g[j]*P,0) for j in range(self.i)],0)
        return F_predict
    
    def load_variables(self,parameters):
        self.P=tf.constant(parameters[0])
        self.g=tf.constant(parameters[2])


# # Next we defien a pipeline to train the model.

# This is the main pipeline. It takes the model, the epochs and training data F. 
# F is a np array with dimension (games,GPU,CPU).
# Non-tested FPS in the training data F should be denoted by np.nan.

# In[5]:


def train_model(model,F,savepath,epochs=200): 
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
    for epoch in range(epochs):           
        train_one_step(model,F,optimizer)  
        if epoch%10==0:
            F_predict=model()
            print('for epoch {}, MSE is {}'.format(epoch,compute_loss_sparse(F_predict,F)))
    save_model(model,savepath)


# In[6]:


## uses tensorflow to do backpropagation onece for each epoch.
def train_one_step(model,F,optimizer):
    with tf.GradientTape() as tape:
        F_predict = model()
        loss=compute_loss_sparse(F_predict, F)
        # compute gradient
        grads = tape.gradient(loss, model.trainable_variables)
        # update to weights
        optimizer.apply_gradients(zip(grads, model.trainable_variables))      


# In[7]:


## computes the mean squared error of predicted FPS with respect to the real FPS at those tested data point in F.  
def compute_loss_sparse(F_predict, F):
    mse = tf.keras.losses.MeanSquaredError()
    indices_true,indices_false=cal_indices(F)
    
    ## if there is no None data or missing data in F, return a normal mse
    ## else return mse based on the given data
    if not indices_false:
        return mse(F_predict,F) 
    else:
        F=tf.constant(F)    
        return mse(tf.gather_nd(F_predict,indices_true),tf.gather_nd(F,indices_true))


# In[8]:


## indices_true is where FPS test is given 
## indices_false is where FPS test is missing
def cal_indices(F):
    indices_true=[]
    indices_false=[]
    for i in range(F.shape[0]):
        for j in range(F.shape[1]):
            for k in range(F.shape[2]):
                if np.isnan(F[i,j,k]):
                    indices_false.append([i,j,k])
                else:
                    indices_true.append([i,j,k])
    return indices_true, indices_false


# In[9]:


def save_model(model,path):
    stored_variables=np.array([i.numpy() for i in model.trainable_variables])
    np.save(path, stored_variables,allow_pickle=True, fix_imports=True)


# # Next, we will load the data and take part of the data as validation set.
# The format of the data will be numpy.array with shape (i,m,n), with i the game label, m the GPU label, and n the CPU label.

# In[10]:


import sqlite3
import os
import pandas as pd
import numpy as np


# In[11]:


## load fps data from data base to np.array file
def sql_to_np():
    cwd = os.getcwd()
    cwd='/'.join(cwd.split('/')[:-1])
    if cwd:
        path=cwd+'/tested_data/games_fps_cpu_gpu.db'
    else:
        cwd = os.getcwd()
        cwd='\\'.join(cwd.split('\\')[:-1])
        path=cwd+'\\tested_data\\games_fps_cpu_gpu.db'
    
    cnx = sqlite3.connect(path)
    c=cnx.cursor()
    Game_Name=c.execute('''SELECT DISTINCT Game_Name FROM games_fps''').fetchall()

    Game_Name=[i[0] for i in Game_Name]


    total=[]
    GPU=[]
    CPU=[]
    for game in Game_Name:
        result=pd.read_sql('''SELECT GPU,CPU,FPS FROM games_fps where Game_Name='{}' '''.format(game),cnx)
        result=result.pivot(index='GPU', columns='CPU', values='FPS')
        result=result.sort_index()
        result=result.reindex(sorted(result.columns), axis=1)
        if len(GPU)==0:
            GPU=result.index         
        if len(CPU)==0:
            CPU=result.columns
        total.append(result.to_numpy())

    total=np.array(total)

    cnx.commit()
    c.close()
    cnx.close()
    
    return total,Game_Name,GPU,CPU


# In[12]:


## randomly set N data in F to be None and return the missing data indices
def setzero(F,N):
    indices=[]
    F_missing=np.copy(F)
    shape=F.shape
    for i in range(N):
        indices.append([random.randint(0,shape[0]-1),random.randint(0,shape[1]-1),random.randint(0,shape[2]-1)])    
    for i,j,k in indices:
        F_missing[i,j,k]=None
    
    return indices,F_missing     


# In[13]:


def validation(indices,model,F):
    mse=tf.keras.losses.MeanSquaredError()
    F_predict=model()  
    return mse(tf.gather_nd(F_predict,indices),tf.gather_nd(F,indices))


# In[14]:


def pred_to_database(F_predict,Game_Name,GPU,CPU):
    total=pd.DataFrame(columns=['CPU','GPU','FPS'])
    for i in range(len(Game_Name)):
        game_fps=pd.DataFrame(data=F_predict[i], index=GPU,  columns=CPU)
        game_fps=game_fps.unstack().reset_index().rename(columns={0:'FPS'})
        game_fps['Game_Name']=Game_Name[i]
        total=total.append(game_fps)
    total.reset_index(drop=True)

    ## get the path of prediction_data file
    cwd = os.getcwd()
    cwd='/'.join(cwd.split('/')[:-1])
    if cwd:
        path=cwd+'/prediction_data/games_fps_cpu_gpu.db'
    else:
        cwd = os.getcwd()
        cwd='\\'.join(cwd.split('\\')[:-1])
        path=cwd+'\\prediction_data\\games_fps_cpu_gpu.db'
    
    ## store data to database
    cnx = sqlite3.connect(path)
    
    try:
        c=cnx.cursor()
        c.execute('''DROP Table games_fps ''')
        c.close()
    except:
        pass
        
    total.to_sql(name='games_fps',con=cnx)

    cnx.commit()
    cnx.close()


# In[32]:


def test_to_pred():
    ## get the path of prediction_data file
    cwd = os.getcwd()
    cwd='/'.join(cwd.split('/')[:-1])
    if cwd:
        path_out=cwd+'/prediction_data/games_fps_cpu_gpu.db'
    else:
        cwd = os.getcwd()
        cwd='\\'.join(cwd.split('\\')[:-1])
        path_out=cwd+'\\prediction_data\\games_fps_cpu_gpu.db'

    if cwd:
        path_in=cwd+'/tested_data/games_fps_cpu_gpu.db'
    else:
        cwd = os.getcwd()
        cwd='\\'.join(cwd.split('\\')[:-1])
        path_in=cwd+'\\tested_data\\games_fps_cpu_gpu.db'
        
    ## store cpu gpu price information to to database
    cnx = sqlite3.connect(path_in)
    cpu_price=pd.read_sql('''SELECT * FROM cpu_price  ''',cnx).drop('index',axis=1)
    gpu_price=pd.read_sql('''SELECT * FROM gpu_price  ''',cnx).drop('index',axis=1)
    cnx.commit()
    cnx.close()

    cnx = sqlite3.connect(path_out)
    try:
        c=cnx.cursor()
        c.execute('''DROP Table cpu_price ''')
        c.close()
    except:
        pass

    try:
        c=cnx.cursor()
        c.execute('''DROP Table gpu_price ''')
        c.close()
    except:
        pass

    cpu_price.to_sql(name='cpu_price',con=cnx)
    gpu_price.to_sql(name='gpu_price',con=cnx)

    cnx.close()


# In[16]:


# This is the pipeline that train the model
# when N=0 it just train the model with existing datat
# else it uses N data as validation data and train the model with
# the rest of the data 
def train_valid_pipeline(N=0):
    F,Game_Name,GPU,CPU=sql_to_np()
    testmodel=model_cpu_gpu(F.shape)
    
    if N!=0:
        ## create some missing data
        indices,F_missing=setzero(F,N)
        i,j,k=F.shape    
        print('\n','The number of training data is {} out of {} \n'.format(np.count_nonzero(~np.isnan(F_missing)),i*j*k))
        ## use the missing data to train the model and save the model
        train_model(testmodel,F_missing,'savedmodel')
        
        ## print out the validation accuracy
        print('\n','The validation MSE is {}'.format(tf.keras.backend.get_value(validation(indices, testmodel,F))))
    
    else:
        train_model(testmodel,F,'savedmodel')
        F_model=testmodel()
        F_predict=np.copy(F)
        ##  substituet the nan value in F with the value predicted by model
        F_predict[np.isnan(F)]=F_model[np.isnan(F)]
        # save the predicted value 
        pred_to_database(F_predict,Game_Name,GPU,CPU)
        test_to_pred()
        print('Prediction Data has been written in SQL format')


# In[33]:


train_valid_pipeline()


# In[ ]:





# In[ ]:




