from sklearn.linear_model import LogisticRegression

from ml.modelinterface import MlInterface

import numpy as np 

class LogisticRegressionModel(MlInterface):
    
    # def __init__(self,*args,**kwargs):#,**kwargs):
    #     self.model = LogisticRegression(*args,**kwargs)#,**args)

    def __init__(self, clf = None, *args,**kwargs):
        if clf is None:
            clf = LogisticRegression(*args,**kwargs)
        super().__init__(clf)

    #def __init__(self, random_state ): #n_clusters=2, random_state=0):
    #    self.lr = LogisticRegression(random_state=random_state)
 
    def train(self, x, y):
        #print('[train()] gamma: {gamma}, kernel: {kernel}, C: {C}, impl: {_impl}'.format(gamma=self.clf.gamma, kernel=self.clf.kernel, C=self.clf.C, _impl=self.clf._impl))
        self.clf.fit(x, y)

    def predict(self, data):
        return self.clf.predict(data)

    def score(self, x, y):
        return self.clf.score(x, y)

