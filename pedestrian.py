from PIL import Image
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from SVM import our_svm

# labels = np.array([])
# images_data = np.array([])

# test_labels = []
# images_test_data = []

def loadImages():
    images_data = np.array([])

    ##load_training_dataset
    for i in range(1,4):
        print(i)

        for j in range(200): #4800
            ## Pedestrian
            file_dir = '/ped_examples/'
            strr = (str(i) + file_dir)
            
            img = np.array(Image.open(strr + "img_" + "{0:05}".format(j) + ".pgm"))
            # labels = np.append(labels, 1) #pedestrian
            images_data = np.append(images_data, [img.flatten(), 1])
            
            ## Nonpedestrian
            file_dir = '/non-ped_examples/'
            strr = (str(i) + file_dir)
            
            img = np.array(Image.open(strr + "img_" + "{0:05}".format(j) + ".pgm"))
            # labels = np.append(labels, 0) #non-pedestrian
            images_data = np.append(images_data, [img.flatten(), 0])

    df = pd.DataFrame({'data':images_data[::2], 'label':images_data[1::2]})
    # df.to_csv('pedestrian.csv', index=False, mode='a', header=False)
    return df

def readFile():
    return pd.read_csv('pedestrian.csv')

## apply just for first time
## Again Abudo don't apply it again if you care about your CPU xD
# loadImages()

## applied each time you run
# df = readFile()

df = loadImages()

print(df.head())
print(df.info())

X = df['data'].values
y = df['label'].values

# print(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# svc_algo = our_svm()
# svc_algo.apply_fit_predict(X_train, y_train, X_test)


##load_testing_dataset
# for i in range(1,3):
#     for j in range(500): #4800
#         file_dir = "/ped_examples/"
#         strr = ('T' + str(i) + file_dir)
        
#         img=np.array(Image.open(strr + "img_" + "{0:05}".format(j) + ".pgm"))
#         test_labels = np.append(labels, 1) #pedestrian
#         images_test_data = np.append(images_test_data, img)

#     for j in range(500): #5000
#         file_dir = "/non-ped_examples/"
#         strr = ('T' + str(i) + file_dir)
        
#         img=np.array(Image.open(strr + "img_" + "{0:05}".format(j) + ".pgm"))
#         test_labels = np.append(labels, 0) #non-pedestrian
#         images_test_data = np.append(images_test_data, img)

# df1 = pd.DataFrame({'data':images_data[::2], 'label':images_data[1::2]})
# df2 = pd.DataFrame(images_test_data, columns=["data"])

# print(df1.head())
# print(df2.head())

# df1.to_csv('pedestrian.csv')

# print(labels)
# print(test_labels)

# X_train, X_test, y_train, y_test = train_test_split(images_data, labels, test_size=0.4)

# svc_algo = our_svm()
# svc_algo.apply_fit_predict(X_train, y_train, X_test)
