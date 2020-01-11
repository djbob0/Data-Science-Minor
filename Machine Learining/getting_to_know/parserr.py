import pandas as pd
import os


class Parser:

    def __init__(self, filename, verbose=False):
        self.dataframe = None
        self.bodypart = None
        self.verbose = verbose
        if os.path.exists(filename):
            if filename.endswith('.csv'):
                self.filename = filename
                self.read_dataframe_from_file()

                name = os.path.splitext(os.path.basename(filename))[0]

                self.catagorie = int(name.split('_')[0].replace('Cat', ''))
                self.patient = int(name.split('_')[1].replace('pat', ''))
                self.meting = int(name.split('_')[2].replace('meting', ''))
                self.oefening = int(name.split('_')[3].replace(
                    'oef', '').replace('.csv', ''))

                if self.verbose:
                    print('[Parser()] parser initalised [{cat} {pat} {met} {oef}]'.format(
                        cat=self.catagorie, pat=self.patient, met=self.meting, oef=self.oefening))
            else:
                raise FileExistsError("File is not a valid CSV file")
        else:
            raise FileNotFoundError("CSV File Not Found")

    def read_dataframe_from_file(self):
        # Checking if file exists
        if os.path.exists(self.filename):
            dataframe = pd.read_csv(self.filename, names=list(range(30)))
            # print('[Parser] setting columns.')
            # Renaming column names to bodypart, thanks to previous group for the names <3
            dataframe = dataframe.rename(
                columns={0: "thorax_r_x", 1: "thorax_r_y", 2: "thorax_r_z"})
            dataframe = dataframe.rename(
                columns={3: "clavicula_r_x", 4: "clavicula_r_y", 5: "clavicula_r_z"})
            dataframe = dataframe.rename(
                columns={6: "scapula_r_x", 7: "scapula_r_y", 8: "scapula_r_z"})
            dataframe = dataframe.rename(
                columns={9: "humerus_r_x", 10: "humerus_r_y", 11: "humerus_r_z"})
            dataframe = dataframe.rename(
                columns={12: "ellebooghoek_r"})
            dataframe = dataframe.rename(
                columns={15: "thorax_l_x", 16: "thorax_l_y", 17: "thorax_l_z"})
            dataframe = dataframe.rename(
                columns={18: "clavicula_l_x", 19: "clavicula_l_y", 20: "clavicula_l_z"})
            dataframe = dataframe.rename(
                columns={21: "scapula_l_x", 22: "scapula_l_y", 23: "scapula_l_z"})
            dataframe = dataframe.rename(
                columns={24: "humerus_l_x", 25: "humerus_l_y", 26: "humerus_l_z"})
            dataframe = dataframe.rename(columns={27: "ellebooghoek_l"})
            self.dataframe = dataframe
        else:
            raise FileNotFoundError("CSV Niet gevonden")

    def get_bodypart(self, bodypart, side=None):
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

        # Extracting the correct DataFrame columns to bodypart
        bodypart = self.dataframe[bodyparts]
        return bodypart

    def get_filename(self):
        return os.path.basename(self.filename)

    def set_bodypart(self, bodypart, side=None):
        self.bodypart = self.get_bodypart(bodypart, side)
        return self.bodypart

    def first_rows(self, count=3):
        return self.bodypart.loc[0:count, self.bodypart.columns[0]]

    def find_row_index(self, rows):
        # Create a list of True/False values for each item in bodypart
        resultlist = self.bodypart.iloc[:, 0] == rows[0]

        # Returning the first index that is a match
        return self.bodypart[resultlist].index[0]

    def dataframe_size(self):
        if self.bodypart is not None:
            return int(int(self.bodypart.size) / 3)
        else:
            return int(self.dataframe.size / len(self.dataframe.columns))
