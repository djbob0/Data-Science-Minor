from sklearn.tree import DecisionTreeClassifier as Tree
from ml.modelinterface import MlInterface

class DecisionTreeClassifier(MlInterface):

    def __init__(self, clf=None, *args,**kwargs):
        if clf is None:
            clf = Tree(*args,**kwargs)
        super().__init__(clf)

    def train(self, x, y):
        self.clf.fit(x,y)

    def predict (self, data):
        return self.clf.predict(data)

    def score(self, x, y):
        return self.clf.score(x, y)