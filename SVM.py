<<<<<<< HEAD
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 00:09:58 2019

@author: Dell
"""

from sklearn import svm

class SVM:
    SVC=svm.SVC()
    
    def apply_fit_predict(self,df,labels,test):
       self.SVC.fit(df,labels)
       print(self.SVC.predict(df))
    
=======
class SVM:
    pass
>>>>>>> 1713b0889c578dbd4158c650d9af904f91e69b10
