from parserr import Parser
import pandas as pd
import os

directory = 'D:\\Hochschule\\5_Semester\\Orthoeyes\\Data\\test-data\\test\\'

df_result = pd.DataFrame

def get_data(directory):
    datalist = []
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            datalist.append(Parser(directory+filename))
    return datalist

def get_mean(df, part):
    var = 0
    for value in df['thorax_r_x']:
        var += value
    mean = var / data.dataframe_size()
    return(mean)

for data in get_data(directory):
    df = data.get_bodypart('thorax', 'r')
    print(get_mean(df, 'thorax_r_x'))