import numpy as np


class MlInterface:
    def __init__(self, clf):
        self.clf = clf

    def train(self, x, y):
        raise NotImplementedError()

    def test(self, x, y):
        if len(x) != len(y):
            raise ValueError('Status values need to be same lenghts')
        self.score = self.clf.score(x, y)
        print('[test()] clf.score:', self.score, 'from', len(x), 'test records')

        wrong = 0 
        array = []  
        for i in range(len(x)):
            if self.predict([x[i]]) != y[i]:
                wrong = wrong + 1
            array.append(self.predict([x[i]]))
            #print(self.predict([x[i]]), y[i])
        print(wrong, len(x))
        return np.array(array), y 

    def predict(self, data):
        return self.clf.predict(data)
        
        raise NotImplementedError()
