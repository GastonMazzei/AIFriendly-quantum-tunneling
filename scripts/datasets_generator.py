#!/usr/bin/env python
# coding: utf-8

# In[14]:


import numpy as np
import pandas as pd


from scipy.stats import bernoulli
from sklearn.preprocessing import StandardScaler

from keras.layers import Dense
from keras.models import Sequential
from keras.losses import categorical_crossentropy

from tunnel_generator import *


# $$\Delta \Psi = (V_{(x)}+E)\Psi$$
# 
# ![](../notebook-imgs/image1.png)
# ![](../notebook-imgs/image2.png)
# ![](../notebook-imgs/image3.png)

# In[15]:


def balanced_samples(d, partitions=4, **kwargs):
    """
    kwargs: verbose=True
    """

    # Shuffle and select the number of parts it 
    # will be split into
    #
    d = d.sample(frac=1)
    p = 1/partitions
    
    # Define the number of samples per partition:
    # equi-sampling will force us to select the 
    # occurrences of the less abundant
    #
    s = int(d.shape[0]/partitions)
    for a in [x*p for x in range(partitions)]:
        inner_s = sum(d.iloc[:,-1].between(a,a+p))
        if inner_s < s: s = inner_s
    s = int(s)
    if kwargs.get('verbose',True):
        print(f'\nBalancing samples:\n{s} samples per bin with {partitions} bins '\
                f'will transform the {d.shape[0]}-points dataset\ninto a '\
                f'{partitions*s}-points dataset!')
    
    # retrieve 's' items per class
    #
    data = []
    for a in [x*p for x in range(partitions)]:
        data += [d[d.iloc[:,-1].apply(lambda x: round(x,1)).between(a,a+p)].sample(frac=1).iloc[:s,:]]
    if kwargs.get('verbose',True):
        print('\nDataset balanced successfully!\n')
    return pd.concat(data,0)


# In[16]:


def main(required_length = 15E3, vainilla = True, size=300):
    df = pd.DataFrame()
    count = 0
    q = 0
    while ((len(df)<required_length) and (count<25)):
        print(f'\nIteration {count}:\nRequired Length: {required_length}\n'             f'Current Length: {len(df)}\n')
        if vainilla: df = balanced_samples(generator(
                                    50+q,
                                    50+q,
                                    50+q,
                                    verbose=True,
                                    domain = [np.linspace(1E-3,1E3,50+q),np.linspace(1E-2,0.99,50+q),np.linspace(1E-2,1E2,50+q)]),
                                           50, verbose=False,)
        else: df = generator(
                            int(1+q/5),
                            int(1+q/5),
                            int(1+q/5),
                            verbose=False,
                            vainilla=False,
                            size=size,)
        count += 1
        q += 5

    if len(df)>=required_length: print('\nSUCCESS! the required-length-condition WAS SATISFIED')
    else: print('\nFAILURE: the required-length-condition WAS NOT SATISFIED')
    print(f'\n\nREQUIRED: {required_length}\nACTUAL: {len(df)}')
    return df


# In[17]:


#df = main(10000, False)


# In[19]:


#print(df.head())
#df.to_csv('../databases/tunnel-effect-database.csv', index=False)


# In[ ]:




