import tqdm
import itertools
import numpy as np 
from PIL import Image
from tabulate import tabulate

from patient.patientgroup import PatientGroup
import matplotlib.pyplot as plt 
from multiprocessing import Process, freeze_support
from pprint import pprint 


# from tools.visualize import *
from tools.visualis import Visualize 
from tools.configloader import ConfigLoader, ConfigCreator

from config import config
 
def generate(patient_groups):

    test_combinations = list()
    train_combinations = list()
    table_data = []

    for patientgroup in patient_groups:
        for patient in patientgroup:  # loop through all patients within a patientgroup
            patient_data = {}
            for exercisegroup in config.exercisegroups:
                # Loopen door AF, EL, AB, etc...
                patient_data[exercisegroup] = []

                # Looping through all exercises of patient
                for exercise in patient:  # loop through all exercises for the current patient
                    # checking each exercise's name, compaire it with
                    # the current exercise group
                    if exercise.exercisegroup == exercisegroup:
                        # Adding the exercise to the correct group
                        # If name is correct
                        patient_data[exercisegroup].append(exercise)

            # Calculating all combinations based on exercise gruops
            patient_groups = [value for key, value in patient_data.items()]

            resultaten = list(itertools.product(*patient_groups))
            
            if len(resultaten) > 0:
                patient_nr = int(patient.exercises[0].patientid)
                patientgroup = patient.exercises[0].patientgroup

                # table_data.append([
                #     patientgroup, patient_nr,
                #     len(patient_data['AF']), len(patient_data['EL']), len(patient_data['AB']),
                #     len(patient_data['RF']), len(patient_data['EH']), len(resultaten),
                #     patient_nr in config.test_patients[patientgroup]])

                if len(patient.exercises) > 0:
                    if int(patient.exercises[0].patientid) in config.test_patients[patient.exercises[0].patientgroup]:
                        test_combinations.extend(resultaten)
                    else: 
                        train_combinations.extend(resultaten)
    if config.logging:
        print(tabulate.tabulate(table_data, headers=[
            'catagorie', 'patientnummer', 'AF', 'EL', 'AB',
            'RF', 'EH', 'combinations', 'test person']))

    if config.show_visualization:
        vv = Visualize(patient_groups,catagory = [1], patients=[1, 2,3,4,5], bones=["humerus_r_y_plane", "humerus_r_z_ele", "humerus_r_y_ax"])
        vv.visualise() 

        # v = Visualise(patient_groups)
        # v.mean()
    
    print('train_combinations: ', len(train_combinations))
    print('test_combinations: ', len(test_combinations))




    def generate_data(combinations):
        #TODO modulo and normalize
        """ function recieves combination array
            1. create np:array of BONE: size 3 (x,y,z)
            
         """
        frames = 100

        np_combination_array = np.empty((0, frames, 40,3))
        np_indicator_array = np.array([]) 
            
        for exercise_combination in tqdm.tqdm(combinations):
            # data = np.append(data, exercise_flat[0])
            data = np.zeros((config.frames_counts, 0))
            np_exercise_list = np.zeros((100,3,40))
            itteration = 0

            for exercise in exercise_combination:
                np_data = exercise.np_data
                np_scalar = np.zeros((100,3))

                for index in range(8):
                    #this a bone
                    x = np_data[:,0+index*3]
                    y = np_data[:,1+index*3]
                    z = np_data[:,2+index*3]

                    np_scalar[:,0] = x
                    np_scalar[:,1] = y 
                    np_scalar[:,2] = z
                    

                    #every itteration(40 times) xyz gets appended to exerciselist
                    np_exercise_list[:,:,itteration] = np_scalar
                    itteration = itteration + 1
            
            data = np.transpose(np_exercise_list, (0, 2, 1))
            

            #creating png before combinationarray(patients) get appended
            
            # imdata = data.astype(np.uint8)
            # im = Image.fromarray(imdata, mode='RGB')
            # im.save("hassan-dev\png_data\whatever" + exercise.unique_patientnr+".png")

            data = np.reshape(data,(1,100,40,3))

            # samples timesteps feutures
            np_combination_array = np.concatenate((np_combination_array, data), axis=0)
        
            #print(np_combination_array.shape)
    
            np_indicator_array = np.append(np_indicator_array, int(int(exercise_combination[0].patientgroup) - 1) )

        np_combination_array = np_combination_array % 360
        np_combination_array = np_combination_array / 360
      
       
        return np_combination_array, np_indicator_array


    

    # Generating test data without the data enrichment 
    np_combination_test, np_indicator_test = generate_data(test_combinations)
    print(np_indicator_test.shape)
    # Generating train data WITH data enrichment 
    np_combination_train, np_indicator_train = generate_data(train_combinations)


    # print('np_combination_test, np_indicator_test: ', np_combination_test.shape, np_indicator_test.shape)
    # print('np_combination_train, np_indicator_train: ', np_combination_train.shape, np_indicator_train.shape)


    data = [np_combination_train, np_combination_test, np_indicator_train, np_indicator_test]
    return data 