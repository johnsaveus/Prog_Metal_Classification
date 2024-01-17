import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

def train_val_KNN(train,val):
    neighbors = range(1,15)
    for ne in neighbors:
        model = KNeighborsClassifier(ne)
        model.fit(train.drop(['Band'],axis=1),train['Band'])
        train_preds = model.predict(train.drop(['Band'],axis=1))
        val_preds = model.predict(val.drop(['Band'],axis=1))
        print('----------Train---------')
        print(accuracy_score(train['Band'],train_preds))
        print('----------Val---------)
        print(accuracy_score(train['Band'],val_preds))
class MachineLearning():
    def __init__(self,train_data,cv_split):
        self.train_data = train_data
        self.cv_split = cv_split
        self.cross_val = StratifiedShuffleSplit(n_splits=cv_split,test_size=0.25)
    def KNN(self):
        neighbors = range(1,30)
        for neigh in neighbors:
            model = KNeighborsClassifier(neigh)
            scores = cross_val_score(model,
                                    self.train_data.drop(['Band'],axis=1),
                                    self.train_data['Band'],
                                    scoring='accuracy',
                                    cv=self.cross_val)
            print(f'Accuracy for {neigh} neighbors: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))
    def SVM(self):
        C = [0.001,0.01,0.1,1,10,100,1000]
        kernels = ['rbf','linear','poly']
        for kernel in kernels:
            for c in C:
                model = SVC(C=c,kernel=kernel,probability=True)
                scores = cross_val_score(model,
                                    self.train_data.drop(['Band'],axis=1),
                                    self.train_data['Band'],
                                    scoring='accuracy',
                                    cv=self.cross_val)
                print(f'Accuracy for C = {c} and kernel = {kernel}: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))
    def RFClassifier(self):
        estimators = range(1,50)
        for est in estimators:
            model = RandomForestClassifier(n_estimators=est)
            scores = cross_val_score(model,
                                    self.train_data.drop(['Band'],axis=1),
                                    self.train_data['Band'],
                                    scoring='accuracy',
                                    cv=self.cross_val)
            print(f'Accuracy for estimators = {est}: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))