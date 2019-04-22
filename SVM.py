from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

class SVM:
#    c = [0.001, 0.01, 0.1, 1, 10]
#    gamma = [0.001, 0.01, 0.1, 1]
    SVC=svm.SVC(gamma='scale')
    def __init__(self):
        steps=[('scaler',StandardScaler()),('svm',self.SVC)]
        self.pipeline=Pipeline(steps)

    def apply_fit_predict(self,df,labels,test):
       self.pipeline.fit(df.astype(float),labels)
       return self.pipeline.predict(test.astype(float))
   
#    def __init__(self):
#        param_grid = {'C': self.c, 'gamma' : self.gamma}
#        self.grid_search = GridSearchCV(svm.SVC(kernel='rbf'), param_grid, cv=5)
#          
#    def apply_grid_search(self,train,labels):
#        self.grid_search.fit(train, labels)
#        return self.grid_search.best_params_    # gamma=0.001 , C=0.01
#        
#    def apply_fit_predict(self,train,labels,test,param):
#        self.SVC=svm.SVC(gamma=param['gamma'],C=param['C'])
#        print(param)
#        self.SVC.fit(train,labels)
#        return self.SVC.predict(test)
    def print_accuracy(self,y_test,y_pred):
        print("SVM_accuracy:"+str(accuracy_score(y_test,y_pred)))
        
    
    
