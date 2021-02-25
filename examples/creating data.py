# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 08:35:35 2021

@author: 49174
"""

# data 
import numpy as np
from scipy.spatial.distance import euclidean
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt

import pandas as pd
vptxt =np.loadtxt(r'D:\thesis\data_hendrick\2D_gound_models\vp.txt')
x = np.arange(0,10,0.0125)
y = np.arange(0,10,0.0125)
xx, yy = np.meshgrid(x, y, sparse=True)
distance = np.sqrt(yy**2 + xx**2)
time = np.divide(distance,vptxt)#add the velocity file instead of vp txt

data= pd.DataFrame()
u=pd.DataFrame()
t=pd.DataFrame()
data['x']= x
data['y']= y
data.
u= distance
t= time


plt.imshow(time)
sns.heatmap(time)
