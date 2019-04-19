from sklearn import svm

class our_svm:
    our_svc = svm.SVC()
    
    def apply_fit_predict(self, df, labels, test):
       self.our_svc.fit(df, labels)
       print(self.our_svc.predict(test))
    
