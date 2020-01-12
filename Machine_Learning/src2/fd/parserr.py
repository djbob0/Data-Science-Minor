import pandas as pd
import os


class Parser:
    def __init__(self, filename):
        self.filename = filename
        self.dataframe = None
        self.read_dataframe_from_file()

    def read_dataframe_from_file(self):
        # Checking if file exists
        if not os.path.exists(self.filename):
            raise FileNotFoundError("Specified CSV is not found: " + self.filename)

        # Reading file content with Pandas to a DataFrame
        # Generating column names from 0 - 29
        dataframe = pd.read_csv(self.filename, names=list(range(30)))

        # Renaming column names to bodypart, thanks to previous group for the names <3
        dataframe = dataframe.rename(columns={0: "thorax_r_x", 1: "thorax_r_y", 2: "thorax_r_z"})
        dataframe = dataframe.rename(columns={3: "clavicula_r_x", 4: "clavicula_r_y", 5: "clavicula_r_z"})
        dataframe = dataframe.rename(columns={6: "scapula_r_x", 7: "scapula_r_y", 8: "scapula_r_z"})
        dataframe = dataframe.rename(columns={9: "humerus_r_x", 10: "humerus_r_y", 11: "humerus_r_z"})
        dataframe = dataframe.rename(columns={12: "ellebooghoek_r"})
        dataframe = dataframe.rename(columns={15: "thorax_l_x", 16: "thorax_l_y", 17: "thorax_l_z"})
        dataframe = dataframe.rename(columns={18: "clavicula_l_x", 19: "clavicula_l_y", 20: "clavicula_l_z"})
        dataframe = dataframe.rename(columns={21: "scapula_l_x", 22: "scapula_l_y", 23: "scapula_l_z"})
        dataframe = dataframe.rename(columns={24: "humerus_l_x", 25: "humerus_l_y", 26: "humerus_l_z"})
        dataframe = dataframe.rename(columns={27: "ellebooghoek_l"})
        self.dataframe = dataframe

    def get_bodypart(self, bodypart, side=None) -> pd.DataFrame:
        bodyparts = []

        # Extending the bodypart with the side ("thorax" -> "thorax_l")
        if side:
            bodypart = bodypart + '_' + side

        # Looping through available DataFrame columns
        for columname in self.dataframe.columns.values:
            # Compairing column name with requested bodypart
            if bodypart in str(columname):
                # Adding it to a list for pandas to filter
                bodyparts.append(columname)

        # Extracting the correct DataFrame columns to df_bodypart
        df_bodypart = self.dataframe[bodyparts]
        return df_bodypart
