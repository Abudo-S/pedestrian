from PIL import Image
import numpy as np
import pandas as pd

#from sklearn import svm
from SVM import SVM 
'''
c=Image.open("img_00000.pgm")
print(list(c.getdata()))
'''
#4800
length1=500 
#5000
length2=500  

labels = [] 
#images_data =[]

test_labels = []
#images_test_data = []
features=[]

for i in range(648):
    features.append("feat"+str(i))      

df1 = pd.DataFrame(columns=features) 
df2 = pd.DataFrame(columns=features)
#load_training_dataset
for i in range(1,4):
    for j in range(length1): #4800
        file_dir = '/ped_examples/'
        strr = (str(i) + file_dir)
        
        img = np.array(Image.open(strr + "img_" + "{0:05}".format(j) + ".pgm"))  
        img = img.flatten()
        df1.loc[j+((i-1)*length1)]=img
        #images_data=np.append(images_data, img)
        labels.append(1) #pedestrian
        
for i in range(1,4):
    for j in range(length2): #5000
        file_dir = '/non-ped_examples/'
        strr = (str(i) + file_dir)
        
        img = np.array(Image.open(strr + "img_" + "{0:05}".format(j) + ".pgm"))
        img = img.flatten()
        df1.loc[j+((i-1)*length2+(3*length1))]=img
        #images_data=np.append(images_data, img)
        labels.append(0) #non-pedestrian
     
#load_testing_dataset
for i in range(1,3):
     for j in range(length1): #4800
         file_dir = "/ped_examples/"
         strr = ('T' + str(i) + file_dir)
         
         img = np.array(Image.open(strr + "img_" + "{0:05}".format(j) + ".pgm"))
         img = img.flatten()
         df2.loc[j+((i-1)*length1)]=img
         #images_test_data=np.append(images_test_data, img)
         test_labels.append(1) #pedestrian

     for j in range(length2): #5000
         file_dir = "/non-ped_examples/"
         strr = ('T' + str(i) + file_dir)
         
         img = np.array(Image.open(strr + "img_" + "{0:05}".format(j) + ".pgm"))
         img = img.flatten()
         df2.loc[j+((i-1)*length2+(2*length1))]=img
         #images_test_data=np.append(images_test_data, img)
         test_labels.append(1) #non-pedestrian
            
#images_data = images_data.reshape((length1*3)+(length2*3),648)
#images_test_data = images_test_data.reshape((length1*2)+(length2*2),648)
#print(df1.info)
#print(df2.info)

#print(labels)
#print(test_labels)

svc=SVM()
y_pred=svc.apply_fit_predict(df1,labels,df2)
svc.print_accuracy(test_labels,y_pred)

#print(labels)
#print(test_labels)

