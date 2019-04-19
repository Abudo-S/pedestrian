from sklearn import svm

class SVM:
    SVC = svm.SVC()
    
    def apply_fit_predict(self,df,labels,test):
       self.SVC.fit(df,labels)
       print(self.SVC.predict(test))
    
