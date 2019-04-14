# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 21:44:54 2019

@author: Dell
"""


from PIL import Image
import numpy as np
import pandas as pd
'''
c=Image.open("img_00000.pgm")
print(list(c.getdata()))
'''

labels=[]
images_data=[]
for i in range(3):
    for j in range(4800):
        if i==0:
            strr="1/ped_examples/"
        elif i==1:
            strr="2/ped_examples/"
        else:
            strr="3/ped_examples/"
            
        img=np.array(Image.open(strr+"img_"+"{0:05}".format(j)+".pgm"))
        labels.append(1) 
        images_data=np.append(images_data,img) 
    for j in range(5000):
        if i==0:
            strr="1/non-ped_examples/"
        elif i==1:
            strr="2/non-ped_examples/"
        else:
            strr="3/non-ped_examples/"
        img=np.array(Image.open(strr+"img_"+"{0:05}".format(j)+".pgm"))
        labels.append(0) 
        images_data=np.append(images_data,img)     
        
df=pd.DataFrame(images_data,columns=["data"]) 
print(df.head())
     

      

