from parserr import Parser
import pandas as pd
import oss



#TODO get test data back
directory = 'D:\\Hochschule\\5_Semester\\Orthoeyes\\Data\\test-data\\test\\'



#df = read_dataframe_from_file(directory+ filename)
#df_thorax_r = get_bodypart(df, 'thorax', 'r')
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
#print(get_mean(df, 'thorax_r_x'))


for data in get_data(directory):
    df = data.get_bodypart('thorax', 'r')
    print(get_mean(df, 'thorax_r_x'))