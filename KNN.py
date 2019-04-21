from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class KNN:
    knn =KNeighborsClassifier()

    
    def __init__(self): 
        steps=[('scaler',StandardScaler()),('knn',self.KNN)]
        self.pipeline=Pipeline(steps)
    
    def apply_fit_predict(self,df,labels,test):
       self.pipeline.fit(df.astype(float),labels)
       return self.pipeline.predict(test.astype(float))
       
    def print_accuracy(self,y_test,y_pred):
        print("accuracy:"+str(accuracy_score(y_test,y_pred)))
