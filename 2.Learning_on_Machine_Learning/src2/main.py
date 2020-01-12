import pandas as pd
import os
import sys
import pprint
import itertools
import numpy as np

from sklearn.metrics import classification_report, accuracy_score

#from LogReg import LogisticRegressionModel

from ml.LogReg import LogisticRegressionModel as LGR
from ml.DecisionTree import DecisionTreeClassifier 

from config.config import config
from controller.patientfolder import Patient
from controller.patientgroup import PatientGroup
from fd.visualize import Visualise_mean
from fd.visualize import Visualise_patient
from fd.visualize import Visualise2

print('start')

#patientgroup list
patient_group = list()

for groupid in range(1,4):
    grouppath = config.basepath.format(groupid = groupid)
    patient_group.append(PatientGroup(grouppath))



#crossjoin exercices

train_combinations = list()
test_combinations = list()

for patientgroup in patient_group:
    for patient in patientgroup.patients:
        patient_data = {}
        for exercisetype in config.exercisetypes:
            
            #empty list AF, AB, EL...
            patient_data[exercisetype] = []

            for exercise in patient.exercises:
                
                if exercise.exercisegroup == exercisetype:

                    patient_data[exercisetype].append(exercise)
        
        #print(patient_data)
        
        resultaten = list(itertools.product(patient_data['AF'],
                                            patient_data['EL'],
                                            patient_data['AB'],
                                            patient_data['RF'],
                                            patient_data['EH']))
            
        #looking if patent is in training or test group
        if len(patient.exercises) > 0:

            if int(patient.exercises[0].patientid) in config.test_selections[patient.exercises[0].patiengroup]:
                #print('patient id '  + patient.exercises[0].patientid + ' test: '+ (patient.exercises[0].patientid))
                test_combinations.extend(resultaten)
            else: 
                #print('patient id ' + patient.exercises[0].patientid + ' train: ' +(patient.exercises[0].patientid))
                train_combinations.extend(resultaten)

print('length test_combinations: ' + str(len(test_combinations)))
print('length train_combinations: ' + str(len(train_combinations)))
#print(config.pp.pprint(train_combinations))



def generate_data(combinations):
    np_combination_array = np.empty((0,len(config.columns)*config.frames_counts *len(resultaten[0]))) #WARUM *5???? weil len resultaten = 5
    np_indicator_array = np.array([])
    #print('np_combinations_array shape: ' + str(np_combination_array.shape))

    for exercise_combination in combinations:
        #for every object in the combination (alpha, bravo, charlie, delta, echo)
        data = np.array([])

        for exercise in exercise_combination:
            a1 = exercise.np_data.reshape(1,len(config.columns)*config.frames_counts)
            data = np.append(data, a1[0])
            #if config.graph:
            #    Visualize(a1)

        #print('exercise_combi data array shape: ' + str(data.shape))

        np_combination_array = np.vstack([np_combination_array, data])
        np_indicator_array = np.append(np_indicator_array, exercise_combination[0].patiengroup)
    return np_combination_array, np_indicator_array


np_combination_test, np_indicator_test = generate_data(test_combinations)
np_combination_train, np_indicator_train = generate_data(train_combinations)


Visualise_patient(patient_group, patientnr=[1],right = True, left = False, linelabel= True)
Visualise_patient.plot()
#Visualise2(patient_group)
#Visualise2.plot()
#Visualise_mean(patient_group, linelabel=True)
#Visualise_mean.plot()
print(np_combination_test.shape, np_indicator_test.shape)
print(np_combination_train.shape, np_indicator_train.shape) 

# Decisiontree = DecisionTreeClassifier()
# print("Training started")
# Decisiontree.train(np_combination_train, np_indicator_train)
# # logisticregression.test(X_test, y_test)
# print("Training done! ")

# y_pred = Decisiontree.predict(np_combination_test)
# print(classification_report(np_indicator_test, y_pred, digits=3))
# print('accuracy_score for Decision Tree is', accuracy_score(np_indicator_test, y_pred, normalize=True, sample_weight=None))


# logisticregression = LGR()
# print("Training started")
# logisticregression.train(np_combination_train, np_indicator_train)
# # logisticregression.test(X_test, y_test)
# print("Training done! ")

# y_pred = logisticregression.predict(np_combination_test)
# print(classification_report(np_indicator_test, y_pred, digits=3))
# print('accuracy_score', accuracy_score(np_indicator_test, y_pred, normalize=True, sample_weight=None))


print('done')
