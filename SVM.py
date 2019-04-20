from sklearn import svm
from sklearn.metrics import accuracy_score

class SVM:
    SVC = svm.SVC(gamma='scale')
    
    
    def apply_fit_predict(self,df,labels,test):
       self.SVC.fit(df,labels)
       return self.SVC.predict(test)
       
    def print_accurcy(self,y_test,y_pred):
        print(accuracy_score(y_test,y_pred))
    
