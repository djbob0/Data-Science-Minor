
import math

import pandas as pd 
import numpy as np
import os

def read_dataframe_from_file(filename):
    # Checking if file exists
    if os.path.exists(filename):

        # Reading file content with Pandas to a DataFrame
        # Generating column names from 0 - 29
        dataframe = pd.read_csv(filename, names=list(range(30)))

        # Renaming column names to bodypart, thanks to previous group for the names <3
        dataframe = dataframe.rename(columns={0: "thorax_r_x_ext", 1: "thorax_r_y_ax", 2: "thorax_r_z_lat"})
        dataframe = dataframe.rename(columns={3: "clavicula_r_y_pro", 4: "clavicula_r_z_ele", 5: "clavicula_r_x_ax"})
        dataframe = dataframe.rename(columns={6: "scapula_r_y_pro", 7: "scapula_r_z_lat", 8: "scapula_r_x_tilt"})
        dataframe = dataframe.rename(columns={9: "humerus_r_y_plane", 10: "humerus_r_z_ele", 11: "humerus_r_y_ax"})
        dataframe = dataframe.rename(columns={12: "ellebooghoek_r"})
        dataframe = dataframe.rename(columns={0: "thorax_l_x_ext", 1: "thorax_l_y_ax", 2: "thorax_l_z_lat"})
        dataframe = dataframe.rename(columns={3: "clavicula_l_y_pro", 4: "clavicula_l_z_ele", 5: "clavicula_l_x_ax"})
        dataframe = dataframe.rename(columns={6: "scapula_l_y_pro", 7: "scapula_l_z_lat", 8: "scapula_l_x_tilt"})
        dataframe = dataframe.rename(columns={9: "humerus_l_y_plane", 10: "humerus_l_z_ele", 11: "humerus_l_y_ax"})
        dataframe = dataframe.rename(columns={27: "ellebooghoek_l"})
        return dataframe
    else: 
        raise FileNotFoundError("CSV Niet gevonden")

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

filename = 'Cat1_pat12_meting10_oef2'

# do transformations here:
trans_no = np.array([1, 1, 1])
trans_l = np.array([-1, -1, 1])
rotation_mode = 'XYZ'

df = read_dataframe_from_file('D:\\Hochschule\\5_Semester\\Orthoeyes\\Data\\test-data\\'+ filename + '.csv')

# Extracting some bodyparts to visualise
df_thorax_r = get_dataframe_from_bodypart(df, 'thorax', 'r')
df_thorax_l = get_dataframe_from_bodypart(df, 'thorax', 'l')
df_clavicula_r = get_dataframe_from_bodypart(df, 'clavicula', 'r')
df_clavicula_l = get_dataframe_from_bodypart(df, 'clavicula', 'l')
df_scapula_r = get_dataframe_from_bodypart(df, 'scapula', 'r')
df_scapula_l = get_dataframe_from_bodypart(df, 'scapula', 'l')
df_humerus_r = get_dataframe_from_bodypart(df, 'humerus', 'r')
df_humerus_l = get_dataframe_from_bodypart(df, 'humerus', 'l')


print(pd.DataFrame.head(df_thorax_r))
###### Blender relevant code following

