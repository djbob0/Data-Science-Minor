import pandas as pd
import numpy as np
import os 

from config.config import config

class Exercises:
    

    def __init__ (self, path):
        self.path = path

        # csv file inlezen 

        self.data = pd.read_csv(self.path, names=list(range(30)))

        colnames = {0: "thorax_r_x", 1: "thorax_r_y", 2: "thorax_r_z",
            3: "clavicula_r_x", 4: "clavicula_r_y",
            5: "clavicula_r_z",
            6: "scapula_r_x", 7: "scapula_r_y", 8: "scapula_r_z",
            9: "humerus_r_x", 10: "humerus_r_y", 11: "humerus_r_z",
            12: "ellebooghoek_r",
            15: "thorax_l_x", 16: "thorax_l_y", 17: "thorax_l_z",
            18: "clavicula_l_x", 19: "clavicula_l_y",
            20: "clavicula_l_z",
            21: "scapula_l_x", 22: "scapula_l_y", 23: "scapula_l_z",
            24: "humerus_l_x", 25: "humerus_l_y", 26: "humerus_l_z",
            27: "ellebooghoek_l"}

        self.data = self.data.rename(columns= colnames)

        #making a small dataframe of 5 rows by multipling the rows with the columns
        self.dataframe = self.data[config.columns].iloc[self.get_frames()]

        self.left = self.dataframe.filter(regex=(r"._l_."))
        self.right = self.dataframe.filter(regex=(r"._r_."))
        
        #compute the numpy array of dataframe by using .to_numpy
        self.np_data = self.dataframe.to_numpy()

        # extracting path exercise type and type of exercise 
        exercisepath, self.exercisestype = os.path.split(self.path[:-4])

        #extracting patientid from path

        grouppath, self.patientid = os.path.split(exercisepath)

        #extracting patientgroup and exercisetype from patientgategory and exercise name  respectively, by indexing the position of the string.

        self.patiengroup = grouppath[-1]
        self.exercisegroup = self.exercisestype[:2]
              # Extracting metadata belonged to an exercise
        # r'C:\Users\hassa\OneDrive\Desktop\DataScience\CODE\data2.0\Catagory_1\1\AB1.csv'
        # print('path of the exercise', path)
        # print('exercisestype', exercisestype)
        # print('patientid', self.patientid)
        # print('patientcategory', self.patiengroup)
        # print('exercisegroup', self.execerisegroup)
    
    def total_rows(self):
        return int(self.data.size / len(self.data.columns))

    def get_frames(self):
        frames = []
        total_rows = self.total_rows() - 1
        for index in range(1, config.frames_counts + 1):
            frames.append(int((total_rows/ config.frames_counts) * index))
        return frames

  
    def get_dataframe_from_bodypart(dataframe, bodypart, side=None):
        bodyparts = []
        
        # Extending the bodypart with the side ("thorax" -> "thorax_l")
        if side:
            bodypart = bodypart + '_' + side

        # Looping through available DataFrame columns 
        for columname in dataframe.columns.values: 
            # Compairing column name with requested bodypart
            if bodypart in str(columname):
                # Adding it to a list for pandas to filter
                bodyparts.append(columname)
        
        # Extracting the correct DataFrame columns to df_bodypart
        df_bodypart = dataframe[bodyparts]
        return df_bodypart
        
        #exercises namen op halen

        # for filename in os.listdir(path):
        #     if filename.endswith('.csv'):
        #     print("The type of execerise is", filename)
