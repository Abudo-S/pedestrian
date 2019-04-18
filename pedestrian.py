from PIL import Image
import numpy as np
import pandas as pd

labels = []
images_data = []

test_labels = []
images_test_data = []

#load_training_dataset
for i in range(1,4):
    for j in range(500): #4800
        file_dir = '/ped_examples/'
        strr = (str(i) + file_dir)
        
        img = np.array(Image.open(strr + "img_" + "{0:05}".format(j) + ".pgm"))
        labels = np.append(labels, 1) #pedestrian
        images_data = np.append(images_data, img)

    for j in range(500): #5000
        file_dir = '/non-ped_examples/'
        strr = (str(i) + file_dir)
        
        img = np.array(Image.open(strr + "img_" + "{0:05}".format(j) + ".pgm"))
        labels = np.append(labels, 0) #non-pedestrian
        images_data = np.append(images_data, img)
        
#load_testing_dataset
for i in range(1,3):
     for j in range(500): #4800
         file_dir = "/ped_examples/"
         strr = ('T' + str(i) + file_dir)
         
         img=np.array(Image.open(strr + "img_" + "{0:05}".format(j) + ".pgm"))
         test_labels = np.append(labels, 1) #pedestrian
         images_test_data = np.append(images_test_data, img)

     for j in range(500): #5000
         file_dir = "/non-ped_examples/"
         strr = ('T' + str(i) + file_dir)
         
         img=np.array(Image.open(strr + "img_" + "{0:05}".format(j) + ".pgm"))
         test_labels = np.append(labels, 0) #non-pedestrian
         images_test_data = np.append(images_test_data, img)
         
df1 = pd.DataFrame(images_data, columns=["data"])
df2 = pd.DataFrame(images_test_data, columns=["data"])

print(df1.head())
print(df2.head())

print(labels)
print(test_labels)
