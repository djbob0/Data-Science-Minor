import pprint

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from controller.datacontroller import DataController

from ml.models import Models
from ml.logisticregression import LogisticRegressionModel
from ml.svc import SVCModel


class config:
    debug = False
    tables = True
    pp = pprint.PrettyPrinter(indent=4)

    exercises = 5
    workers = 20
    max_chunck_size = 100

    test_size = 0.2
    test_random_state = 42

    if debug:
        basepath = "src\\data\\cleaned-regrouped-small\\"
    else:
        basepath = "src\\data\\cleaned-regrouped\\"

print('ORTHO: Prepairing Dataset')

controller = DataController(config)
controller.run()
