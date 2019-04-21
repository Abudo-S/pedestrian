from PIL import Image
import numpy as np
import pandas as pd
#from sklearn.model_selection import train_test_split

#from sklearn import svm
from SVM import SVM 
'''
c=Image.open("img_00000.pgm")
print(list(c.getdata()))
'''
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
         

#images_data = images_data.reshape((length1*3)+(length2*3),648)
#images_test_data = images_test_data.reshape((length1*2)+(length2*2),648)
#print(df1.info)
#print(df2.info)

#print(labels)
#print(test_labels)

svc=SVM()
y_pred=svc.apply_fit_predict(X1,y1,X2)
svc.print_accuracy(y2,y_pred)

#print(labels)
#print(test_labels)

# labels = np.array([])
# images_data = np.array([])

# test_labels = []
# images_test_data = []
'''
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

print(X_train)
'''
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
