from sklearn import svm
from sklearn.metrics import accuracy_score

<<<<<<< HEAD
class SVM:
    SVC = svm.SVC(gamma='scale')
    
    
    def apply_fit_predict(self,df,labels,test):
       self.SVC.fit(df,labels)
       return self.SVC.predict(test)
       
    def print_accuracy(self,y_test,y_pred):
        print("accuracy:"+str(accuracy_score(y_test,y_pred)))
    
        
=======
class our_svm:
    our_svc = svm.SVC()
    
    def apply_fit_predict(self, df, labels, test):
       self.our_svc.fit(df, labels)
       print(self.our_svc.predict(test))
>>>>>>> 385d046c9353ec1fa72029da658c83aef5ddb7b5
    
