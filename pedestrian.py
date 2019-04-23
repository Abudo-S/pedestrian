from PIL import Image
import numpy as np
import pandas as pd
#from sklearn.model_selection import train_test_split

#from sklearn import svm
from SVM import SVM 
from KNN import KNN
from CNN import CNN
from CNN2 import CNN2

#4800
length1=4800 
#5000
length2=5000  

labels = [] 
#images_data =[]

test_labels = []
#images_test_data = []
features=[]

for i in range(648):
    features.append("feat"+str(i))      

features.append("ped")

df1 = pd.DataFrame(columns=features) 
df2 = pd.DataFrame(columns=features)

def make_csv_from_df1(start,end,df1):
    #load_training_dataset
    for i in range(start,end):
        for j in range(length1): #4800
            file_dir = '/ped_examples/'
            strr = (str(i) + file_dir)
            
            img = np.array(Image.open(strr + "img_" + "{0:05}".format(j) + ".pgm"))  
            img = img.flatten()
            img=np.append(img,1)
            #prevent overwriting
            df1.loc[j+((i-1)*length1)]=img
            #images_data=np.append(images_data, img)
            labels.append(1) #pedestrian
            
#for i in range(start,end):
        for j in range(length2): #5000
            file_dir = '/non-ped_examples/'
            strr = (str(i) + file_dir)
            
            img = np.array(Image.open(strr + "img_" + "{0:05}".format(j) + ".pgm"))
            img = img.flatten()
            img=np.append(img,0)
            df1.loc[j+((i-1)*length2+(3*length1))]=img
            #images_data=np.append(images_data, img)
            labels.append(0) #non-pedestrian
    df1.to_csv('pedestrian.csv', index=False, mode='w')
    
def make_csv_from_df2(start,end,df2):   
    #load_testing_dataset
    for i in range(start,end):
        for j in range(length1): #4800
            file_dir = "/ped_examples/"
            strr = ('T' + str(i) + file_dir)
            
            img = np.array(Image.open(strr + "img_" + "{0:05}".format(j) + ".pgm"))
            img = img.flatten()
            img=np.append(img,1)
            print(img)
            df2.loc[j+((i-1)*length1)]=img
            #images_test_data=np.append(images_test_data, img)
            test_labels.append(1) #pedestrian

    #for i in range(start,end):
        for j in range(length2): #5000
            file_dir = "/non-ped_examples/"
            strr = ('T' + str(i) + file_dir)
            
            img = np.array(Image.open(strr + "img_" + "{0:05}".format(j) + ".pgm"))
            img = img.flatten()
            img=np.append(img,0)
            print(img)
            df2.loc[j+((i-1)*length2+(2*length1))]=img
            #images_test_data=np.append(images_test_data, img)
            test_labels.append(0) #non-pedestrian
    df2.to_csv('test_pedestrian.csv', index=False, mode='w')

train = pd.read_csv('pedestrian.csv')
X1 = train.drop('ped',axis=1).values
y1 = train['ped'].values

test=pd.read_csv('test_pedestrian.csv')
X2 = test.drop('ped',axis=1).values
y2 = test['ped'].values
         
svc=SVM()
y_pred=svc.apply_fit_predict(X1,y1,X2)
svc.print_accuracy(y2,y_pred)

knn=KNN()
y_pred=knn.apply_fit_predict(X1,y1,X2)
knn.print_accuracy(y2,y_pred)

cnn=CNN(X1,X2,29400,19600)
cnn.create_model(y1)
cnn.print_accuracy(y2)

cnn2=CNN2()
cnn2.train(X1,y1)
cnn2.print_accuracy(X2,y2)


#print(cnn.cnn_model_fn(X1.reshape(29400,36, 18, 1),y1,tf.estimator.ModeKeys.TRAIN))
#print(cnn.cnn_model_fn(X2.reshape(19600,36, 18, 1),y2,tf.estimator.ModeKeys.PREDICT))
#images_data = images_data.reshape((length1*3)+(length2*3),648)
#images_test_data = images_test_data.reshape((length1*2)+(length2*2),648)
#print(df1.info())
#print(df2.info())

#print(labels)
#print(test_labels)