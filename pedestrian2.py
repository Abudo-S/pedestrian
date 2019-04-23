from PIL import Image
import numpy as np
# import pandas as pd
import tensorflow as tf
#from sklearn.model_selection import train_test_split

#from sklearn import svm
from SVM import SVM 
'''
c=Image.open("img_00000.pgm")
print(list(c.getdata()))
'''
#4800
length1=200
#5000
length2=200

labels = []
images_data =np.array([])

test_labels =[]
images_test_data = np.array([])

#load_training_dataset
for i in range(1,4):
    for j in range(length1): #4800
        file_dir = '/ped_examples/'
        strr = (str(i) + file_dir)
        
        img = np.array(Image.open(strr + "img_" + "{0:05}".format(j) + ".pgm"))
        images_data=np.append(images_data, img)
        labels.append(1) #pedestrian
    for j in range(length2): #5000
        file_dir = '/non-ped_examples/'
        strr = (str(i) + file_dir)
        
        img = np.array(Image.open(strr + "img_" + "{0:05}".format(j) + ".pgm"))
        images_data=np.append(images_data, img)
        print(img)
        labels.append(0) #non-pedestrian
     
#load_testing_dataset
for i in range(1,3):
     for j in range(length1): #4800
         file_dir = "/ped_examples/"
         strr = ('T' + str(i) + file_dir)
         
         img = np.array(Image.open(strr + "img_" + "{0:05}".format(j) + ".pgm"))
         images_test_data=np.append(images_test_data, img)
         test_labels.append(1) #pedestrian

     for j in range(length2): #5000
         file_dir = "/non-ped_examples/"
         strr = ('T' + str(i) + file_dir)
         img = np.array(Image.open(strr + "img_" + "{0:05}".format(j) + ".pgm"))
         images_test_data=np.append(images_test_data, img)
         test_labels.append(0) #non-pedestrian

model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

print(images_data.shape)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(images_data, labels, epochs=5)
model.evaluate(images_test_data, test_labels)