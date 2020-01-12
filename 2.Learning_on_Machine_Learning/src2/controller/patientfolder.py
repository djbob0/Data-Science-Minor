import os
from controller.exercises import Exercises

class Patient:

    def __init__(self, path):
        self.path = path
        self.exercises = list()

        #loops through a patient folder and creates a list of all exercises done by that patient

        for filename in os.listdir(self.path):
            
            if filename.endswith('.csv'):
                
                csvfile=(os.path.join(self.path, filename))
                exercise = Exercises(csvfile)
                self.exercises.append(exercise)
            
                
                


