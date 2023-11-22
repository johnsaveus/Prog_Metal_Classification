from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score , confusion_matrix , f1_score



""" https://towardsdatascience.com/hyperparameter-tuning-for-support-vector-machines-c-and-gamma-parameters-6a5097416167
"""
class SVM():

    def __init__(self,X_train,y_train)
        
        self.X_train = X_train
        self.y_train = y_train

    def optimize(self,kernel = None,gamma = None,C,cv):

        if kernel is None:
            kernel = ['linear','poly','sigmoid','rbf']

        if gamma is None:
            gamma = [0.0001,0.001,0.001,0.01,0.1,1,10]

        if C is None:
            C = [0.1,1,10,100]
            
        if cv is None:
            cv = 5

        parameters = {'kernel':kernel,'gamma':gamma,'C':C}

        model = svm.SVM()

        grid_search = GridSearchCV(model,parameters,cv)

        grid_search.fit(self.X_train,self.y_train)


    def predict(self,X_test,y_test)
    
        self.y_pred
    
    def metrics(self):

        accuracy = accuracy_score(self.y_test,self.y_pred)
        conf_matrix = confusion_matrix(self.y_test,self.y_pred)
        f1 = f1_score(self.y_test,self.y_pred)

        return accuracy, conf_matrix, f1


        




