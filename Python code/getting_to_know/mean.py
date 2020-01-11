from parserr import Parser
import pandas as pd
import os


directory = 'D:\\Hochschule\\5_Semester\\Orthoeyes\\Data\\test-data\\'





#df = read_dataframe_from_file(directory+ filename)
#df_thorax_r = get_bodypart(df, 'thorax', 'r')


def get_data(directory, filter):
    datalist = []
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            name = "_".join(filename[:-4].split("_",3)[3:4])
            print(name)
            if filter == name:
                datalist.append(Parser(directory+filename))
    return datalist


def get_max(df, part, data):
    var = 0
    for value in df['part']:
        var += value
    mean = var / data.dataframe_size()
    return(mean)
#print(get_mean(df, 'thorax_r_x'))


def calculate_bodypart(bodypart, filter):
    n= 1
    mean_max_r = [[],[],[]]
    mean_max_l = [[],[],[]]
    mean_min_r = [[],[],[]]
    mean_min_l = [[],[],[]]
    max_r = [[],[],[]]
    max_l = [[],[],[]]
    min_r = [[],[],[]]
    min_l = [[],[],[]]
    result = {'mean_r': [0, 0, 0], 'mean_max_r': [0,0,0],'range_r':[0,0,0],'mean_l': [0,0,0],'mean_max_l':[0,0,0], 'range_l':[0,0,0]}
    for data in get_data(directory, filter):
        df_r = data.get_bodypart(bodypart, 'r')
        df_l = data.get_bodypart(bodypart, 'l')
        for key in range(0,3,1):

            result['mean_r'][key] = result['mean_r'][key] + df_r.iloc[:, key].mean()      
            result['mean_l'][key] = result['mean_l'][key] + df_l.iloc[:, key].mean()
            
            mean_max_r[key].append(df_r.iloc[:,key].mean())
            mean_max_l[key].append(df_l.iloc[:,key].mean())

            mean_min_r[key].append(df_r.iloc[:,key].mean())
            mean_min_l[key].append(df_l.iloc[:,key].mean())

            max_r[key] = max(df_r.iloc[:,key])
            max_l[key] = max(df_l.iloc[:,key])

            min_r[key] = min(df_r.iloc[:, key])
            min_l[key] = min(df_l.iloc[:, key])
        n +=1
    for key in range(0,3,1):
        result['mean_r'][key] = result['mean_r'][key] /n
        result['mean_l'][key] = result['mean_l'][key] /n

        result['mean_max_r'][key] = max(mean_max_r[key])
        result['mean_max_l'][key] = max(mean_max_l[key])

        result['range_r'][key] = str(min_r[key]) + ' / ' + str(max_r[key])
        result['range_l'][key] = str(min_l[key]) + ' / ' + str(max_l[key])


    return result
    
result_df = pd.DataFrame(calculate_bodypart('scapula', 'oef6' ))
print(result_df)
    