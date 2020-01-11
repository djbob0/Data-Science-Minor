import os
from model.exercise import Exercise


import numpy as np
import pandas as pd
from tqdm import tqdm

class FolderController:

    def __init__(self, path, verbose=False):
        #save path to class
        self.path = path
        self.data = None
        self.files = []
        self.indicator = None
        self.verbose = verbose

        print('FolderController()', self.path)

        # Check for path
        if not os.path.exists(path):
            raise FileExistsError("Folder not found" + self.path)
        else:
            self.import_files()

    def filter_patientgroup(self, categorie):
        return [exercise for exercise in self.files
                if exercise.categorie == categorie]

    def import_files(self):
        listdir = os.listdir(self.path)

        