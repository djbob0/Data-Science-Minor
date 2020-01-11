import os
import pandas as pd
import itertools
from patient.exercise import Exercise
ex = ['AF', 'EL', 'AB', 'RF', 'EH']

from itertools import chain, combinations

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)  # allows duplicate elements
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))
print(list(powerset(ex)))
exit()


ax = Exercise (r'C:\Users\hassa\OneDrive\Desktop\DataScience\CODE\data2.0\Catagory_2\1\EH1.csv').df.diff()

print("for EH", ax.sum(axis = 0, skipna = True))

exit()


# for AF humerus_r_y_ax 
# for EL humerus_r_z_ele
# for AB humerus_r_y_plane 
# for RF humerus_r_y_plane
# for EH humerus_r_z_ele 
 
# demo = { 
#     'ef1': 1,
#     'ef2': 2,
#     'af1':3,
# }

AB = humerus_l_y_ax 

# print(demo.items())
exit()
a = ['a', 'aa', 'aaa']
b = ['b', 'bb', 'bbb']
c = ['c', 'cc', 'ccc']
d = ['d', 'dd', 'ddd']

print(list(itertools.product(a, b, c, d)))
exit()
# # Oefeningen van patient
# # [ ............................ ]


def get_frames(count=5):
    frames = []
    size = 159 - 1
    for index in range(1, count + 1):
        print('index', index, 'num:', int((size / count) * index))


get_frames(5)
exit()
lijst['alpha']
patientexercises = ['AF',
                    'AF',
                    'EL',
                    'EL',
                    'AB',
                    'AB',
                    'RF',
                    'RF',
                    'EH',
                    'EH']

exercises = ['AF', 'EL', 'AB', 'RF', 'EH']
data = {}

for exercisename in exercises:
    data[exercisename] = []

    for exercise in patientexercises:
        if exercise == exercisename:
            data[exercisename].append(exercise)

print(data['AF'])


exit()
combinaties = list(itertools.product(alpha, bravo, charlie, delta, echo))

print(len(combinaties))
# patientgrouppath = '/Users/developer/Documents/School/DataScience/Octave/FobVisData HHS 20191009/Output/Catagory1'

# patientgroup = PatientGroup(patientgrouppath)


# csvpath = '/Users/developer/Documents/School/DataScience/Octave/FobVisData HHS 20191009/Output/Catagory1/1/EL2.csv'
# oefening = Exercise(csvpath)


# filename = 'AF1.csv'

# print(filename[:2])

# print(filename[ <start> : <einde> ])


# patientfolder = '/Users/developer/Documents/School/DataScience/Octave/FobVisData HHS 20191009/Output/Catagory1/23'

# grouppath, patientid = os.path.split(patientfolder)
# print('patientid', patientid)
# print('grouppath', grouppath)

# path, patientgroup = os.path.split(grouppath)
# print('path', path)
# print('patientgroup', patientgroup)

# print(patientid)
# print(patientgroup[-1])

# path = '/Users/developer/Documents/School/DataScience/Octave/FobVisData HHS 20191009/Output/Catagory1/1'


# filename = 'AB1.csv'

# print(path)
# print(filename)
# print(os.path.join(path, filename))


# for filename in os.listdir(path):
#     if filename.endswith('.csv'):
#         print(filename)


# colnames = list(range(30))
# print('Columnames', colnames)

# dataframe = pd.read_csv(csvpath, names=colnames, header=None)
# columns = {0: "thorax_r_x", 1: "thorax_r_y", 2: "thorax_r_z",
#            3: "clavicula_r_x", 4: "clavicula_r_y",
#            5: "clavicula_r_z",
#            6: "scapula_r_x", 7: "scapula_r_y", 8: "scapula_r_z",
#            9: "humerus_r_x", 10: "humerus_r_y", 11: "humerus_r_z",
#            12: "ellebooghoek_r",
#            15: "thorax_l_x", 16: "thorax_l_y", 17: "thorax_l_z",
#            18: "clavicula_l_x", 19: "clavicula_l_y",
#            20: "clavicula_l_z",
#            21: "scapula_l_x", 22: "scapula_l_y", 23: "scapula_l_z",
#            24: "humerus_l_x", 25: "humerus_l_y", 26: "humerus_l_z",
#            27: "ellebooghoek_l"}

# dataframe = dataframe.rename(columns=columns)
# print(type(dataframe))
# print(dataframe.head())
