#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import pandas as pd

from scipy.stats import bernoulli
from sklearn.preprocessing import StandardScaler

from functools import partial

from keras.layers import Dense
from keras.models import Sequential
from keras.losses import categorical_crossentropy


# In[9]:


def transmission_coefficient(L,V,E):
    mass_by_hbar_squared = 1
    squared_gamma_by_two = ((1-E/V)/(E/V)+(E/V)/(1-E//V)-2)/4
    beta = np.sqrt(2*mass_by_hbar_squared*(V-E))
    T_inverse = np.cosh(beta*L)**2 + squared_gamma_by_two * np.sinh(beta*L)**2
    return T_inverse**(-1)
    
def transmission_dispatcher(L,V,E,**kwargs):
    T = transmission_coefficient(L,V,E)
    if kwargs.get('categorical',True):
        return bernoulli.rvs(T,0,kwargs.get('size',1000))
    else:
        #return np.mean(bernoulli.rvs(T,0,kwargs.get('size',1000)))
        return T


# In[11]:


def generator(Ln=50, En=50, Vn=50, **kwargs):
    """
    kwargs: 
        verbose=True
        vainilla=True
            True = each iteration adds (L,V,E,T)
            False = each iteration adds 'size' number
                    of vectors like this (e.g. T=0.25, size=4):
                            L V E 0
                            L V E 0
                            L V E 1
                            L V E 0
        size = 1000
            size for vainilla=False case
    """
    def vainilla(L,V,E):
        nonlocal df
        df['L'].append(L)
        df['V'].append(V)
        df['E'].append(E)
        df['proba'].append(
            transmission_dispatcher(
                L,V,E,categorical=False))
        return df
    #
    def categorical(size,L,V,E):
        nonlocal df
        df['L'] += [L] * size
        df['V'] += [V] * size
        df['E'] += [E] * size
        df['proba'].append(
            transmission_dispatcher(
                L,V,E,categorical=True, size=size))        
        return df
    #
    if kwargs.get('vainilla', True):
        main = vainilla
        L_range = np.linspace(1E-4,1E1,Ln)
        E_range = np.linspace(1E-1,1E1,En)
        V_range = np.linspace(1E-1,5E1,Vn)
    else:
        main = partial(categorical, kwargs.get('size',1000))
        L_range = np.linspace(1E-4,1E1,min(Ln,10))
        E_range = np.linspace(1E-1,1E1,min(En,10))
        V_range = np.linspace(1E-1,5E1,min(Vn,10))
    if kwargs.get('verbose',True):
        print(f'\nGenerating {Ln*En*Vn} simulations...')
    #
    df = {'L':[],
          'V':[],
          'E':[],
          'proba':[],
         }
    for L in L_range:
        for E in E_range:
            temporal_V = [x for x in V_range if x>E]
            for V in temporal_V:
                main(L,V,E)
    #
    if kwargs.get('verbose',True):
        print(f'\nData generation ended successfully!')
        print(f'\nTotal: {len(df["proba"])} cases made sense and were saved!')
    return pd.DataFrame(df)

