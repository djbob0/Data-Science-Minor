import matplotlib.pyplot as plt 
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np 
import pandas as pd
from config.config import config

# READ ONLY DATA FROM OBJECTS

class Visualise_patient():
    def __init__(self, data, patientnr = None, right = True, left = False, linelabel = False):
        y = list(range(1,config.frames_counts+1))
        fig, axs = plt.subplots(4,3)
        for obj in data:
            for patient in obj.patients:
                for exer in patient.exercises:
                    #print(exer.left.shape, exer.right.shape)

                    df_r = exer.right
                    df_l = exer.left


                    """toggeling internal exerciselist"""
                    exercise_toggle = [False] * len(config.exercisetypes)
                    i = 0
                    check = 0
                    for key, exercise in config.exer_list.items():
                        if exer.exercisestype[:-1] == key and exercise:
                            exercise_toggle[i] = True
                            i = i+1
                        else:
                            check = check+1
                            i = i+1
                    


                    """change linestyle depending on cat"""
                    #print(exer.patiengroup)
                    if int(exer.patiengroup) == 1 and config.pg1:
                        linestyle = '-'                        
                    elif int(exer.patiengroup) == 2 and config.pg2:
                        linestyle = '--'
                    elif int(exer.patiengroup) == 3 and config.pg3:
                        linestyle = ':' 
                    else:
                        linestyle = None   
                        
                    linelabels = exer.patiengroup + ' '  + exer.patientid+ ' '+ exer.exercisestype

                    color = [[(1,0,0),(0.6,0,0)],
                            [(0,0,1),(0,0,0.6)],
                            [(0,1,0),(0,0.6,0)],
                            [(1,1,0),(0.6,0.6,0)],
                            [(0,0,0),(0.4,0.4,0.4)]]
                    
                    for i, toggle in enumerate(exercise_toggle):     
                        print(type(int(exer.patientid)), type(patientnr[0]))
                        if toggle and int(exer.patientid) in patientnr:
                            
                            print('hit')
                            j = 0
                            for axsi in axs:
                                for ax in axsi:
                                    print(i)
                                    ax.plot(y,df_r.iloc[:,j], c = color[4][0], linestyle= linestyle)
                                    ax.plot(y,df_l.iloc[:,j], c = color[i][1], linestyle= linestyle)
                                    ax.grid(b=True, which='major', color='k', linestyle='-', alpha=0.3)
                                    j = j+1
 


        """giving labels to subplots"""
        cols = config.exercisetypes
        rows = ['clavicula', 'scapula', 'humerus']
        for ax, col in zip(axs[0], cols):
            ax.set_title(col)
        for ax, row in zip(axs[:,0], rows):
            ax.set_ylabel(row, rotation=0, size='large')

        """ creating personalized legend """
        labels = ['AF r', 'AF l', 'EL r','EL l', 'AB r','AB l', 'RF r','RF l', 'EH r','EH l', 'Cat1 ', 'Cat2 ', 'Cat3']
        handels = [mpatches.Patch(color=color[0][0]),
                mpatches.Patch(color=color[0][1]), 
                mpatches.Patch(color=color[1][0]),
                mpatches.Patch(color=color[1][1]), 
                mpatches.Patch(color=color[2][0]),
                mpatches.Patch(color=color[2][1]), 
                mpatches.Patch(color=color[3][0]),
                mpatches.Patch(color=color[3][1]), 
                mpatches.Patch(color=color[4][0]),
                mpatches.Patch(color=color[4][1]), 
                mlines.Line2D((0,0), (1,1),linestyle='-', color = 'black'),
                mlines.Line2D((0,0), (1,1),linestyle='--', color = 'black'), 
                mlines.Line2D((0,0), (1,1),linestyle=':', color = 'black')]
        fig.legend(handles=handels, labels=labels, loc="upper left")
        
    def plot():
        plt.show()          




class Visualise_mean():
    def __init__(self, data, right = True, left = False, linelabel = False):

        y = list(range(1,config.frames_counts+1))
        fig, axs = plt.subplots(3, 5)
        

        for obj in data:
            for patient in obj.patients:
                for exer in patient.exercises:
                    #print(exer.left.shape, exer.right.shape)

                    df_r = exer.right
                    df_l = exer.left


                    """toggeling internal exerciselist"""
                    exercise_toggle = [False] * len(config.exercisetypes)
                    i = 0
                    check = 0
                    for key, exercise in config.exer_list.items():
                        if exer.exercisestype[:-1] == key and exercise:
                            exercise_toggle[i] = True
                            i = i+1
                        else:
                            check = check+1
                            i = i+1
                    
                    color = [[(1,0,0),(0.6,0,0)],
                            [(0,0,1),(0,0,0.6)],
                            [(0,1,0),(0,0.6,0)],
                            [(1,1,0),(0.6,0.6,0)],
                            [(0,0,0),(0.4,0.4,0.4)]]

                    """change linestyle depending on cat"""
                    #print(exer.patiengroup)
                    if int(exer.patiengroup) == 1 and config.pg1:
                        colour = 'orange'                       
                    elif int(exer.patiengroup) == 2 and config.pg2:
                        colour = 'blue'
                    elif int(exer.patiengroup) == 3 and config.pg3:
                        colour = 'green'
                    else:
                        linestyle = None   
                        
                    linelabels = 'Cat ' +exer.patiengroup + ', id '  + exer.patientid
                    mean = ((abs(df_r.iloc[:,3]+df_r.iloc[:,4]+df_r.iloc[:,5])/3, abs(df_l.iloc[:,3]+df_l.iloc[:,4]+df_l.iloc[:,5])/3), 
                                    (abs(df_r.iloc[:,6]+df_r.iloc[:,7]+df_r.iloc[:,8])/3, abs(df_l.iloc[:,6]+df_l.iloc[:,7]+df_l.iloc[:,8])/3),
                                    (abs(df_r.iloc[:,9]+df_r.iloc[:,10]+df_r.iloc[:,11])/3, abs(df_l.iloc[:,9]+df_l.iloc[:,10]+df_l.iloc[:,11])/3))
                    # clavicula_mean = 
                    # scapula_mean = 
                    # humerus_mean = 
                    

                    for i, toggle in enumerate(exercise_toggle):
                        if toggle:
                            
                            for j in range(len(mean)):
                                if right:
                                    axs[j,i].plot(y,mean[j][0], c = colour, linestyle= '-')
                                    if linelabel:
                                        axs[j,i].annotate(linelabels, xy=(y[2],mean[j][0].iloc[2]), xycoords='data')
                                if left:
                                    axs[j,i].plot(y,mean[j][1], c = colour, linestyle= '-')
                                    if linelabel:
                                        axs[j,i].annotate(linelabels, xy=(y[2],mean[j][1].iloc[2]), xycoords='data')
   

        """giving labels to subplots"""
        cols = config.exercisetypes
        rows = ['clavicula', 'scapula', 'humerus']
        for ax, col in zip(axs[0], cols):
            ax.set_title(col)
        for ax, row in zip(axs[:,0], rows):
            ax.set_ylabel(row, rotation=0, size='large')

        """ creating personalized legend """
        labels = ['Cat1 ', 'Cat2 ', 'Cat3']
        handels = [
                mlines.Line2D((0,0), (1,1),linestyle='-', color = 'orange'),
                mlines.Line2D((0,0), (1,1),linestyle='-', color = 'blue'), 
                mlines.Line2D((0,0), (1,1),linestyle='-', color = 'green')]
        fig.legend(handles=handels, labels=labels, loc="upper left")
        fig.suptitle('Mean of (X + Y + Z) / 3 for every bone', fontsize = 16)
    def plot():
        plt.show()                    





class Visualise2():
    def __init__(self, data):
        e = list(range(1,6))
        fig, axs = plt.subplots(4, 3)
        for obj in data:
            for patient in obj.patients:
                for exer in patient.exercises:
                    #print(exer.left.shape, exer.right.shape)

                    """changing the line color depending on exercise"""
                    if exer.exercisestype[:-1] == config.exercisetypes[0] and config.AF:
                        color_r = (1,0,0)
                        color_l = (0.6,0,0)
                    elif exer.exercisestype[:-1] == config.exercisetypes[1] and config.EL:
                        color_r = (0,0,1)
                        color_l = (0,0,0.6)
                    elif exer.exercisestype[:-1] == config.exercisetypes[2] and config.AB:
                        color_r = (0,1,0)
                        color_l = (0,0.6,0)    
                    elif exer.exercisestype[:-1] == config.exercisetypes[3] and config.RF:
                        color_r = (1,1,0)
                        color_l = (0.6,0.6,0)
                    elif exer.exercisestype[:-1] == config.exercisetypes[4] and config.EH:
                        color_r = (0,0,0)
                        color_l = (0.4,0.4,0.4)
                    else:
                        color_r = None
                        color_l = None


                    """change linestyle depending on cat"""
                    #print(exer.patiengroup)
                    if int(exer.patiengroup) == 1 and config.pg1:
                        linestyle = '-'                        
                    elif int(exer.patiengroup) == 2 and config.pg2:
                        linestyle = '--'
                    elif int(exer.patiengroup) == 3 and config.pg3:
                        linestyle = ':' 
                    else:
                        linestyle = None                   

                    """plotting the graph, but only if there is a linestyle or color"""
                    if linestyle and color_r:
                        df_r = exer.right
                        df_l = exer.left
                        i = 0
                        for axsi in axs:
                            for ax in axsi:
                                ax.plot(e,df_r.iloc[:,i], c = color_r, linestyle= linestyle)
                                ax.plot(e,df_l.iloc[:,i], c = color_r, linestyle= linestyle)
                                ax.grid(b=True, which='major', color='k', linestyle='-', alpha=0.3)
                                i = i+1      

        """giving labels to subplots"""
        cols = ['X', 'Y', 'Z']
        rows = ['thorax', 'clavicula', 'scapula', 'humerus']
        for ax, col in zip(axs[0], cols):
            ax.set_title(col)
        for ax, row in zip(axs[:,0], rows):
            ax.set_ylabel(row, rotation=0, size='large')

        """ creating personalized legend """
        labels = ['AF r', 'AF l', 'EL r','EL l', 'AB r','AB l', 'RF r','RF l', 'EH r','EH l', 'Cat1 ', 'Cat2 ', 'Cat3']
        handels = [mpatches.Patch(color=(1,0,0)),
                mpatches.Patch(color=(0.6,0,0)), 
                mpatches.Patch(color=(0,0,1)),
                mpatches.Patch(color=(0,0,0.6)), 
                mpatches.Patch(color=(0,1,0)),
                mpatches.Patch(color=(0,0.6,0)), 
                mpatches.Patch(color=(1,1,0)),
                mpatches.Patch(color=(0.6,0.6,0)), 
                mpatches.Patch(color=(0,0,0)),
                mpatches.Patch(color=(0.4,0.4,0.4)), 
                mlines.Line2D((0,0), (1,1),linestyle='-', color = 'black'),
                mlines.Line2D((0,0), (1,1),linestyle='--', color = 'black'), 
                mlines.Line2D((0,0), (1,1),linestyle=':', color = 'black')]
        fig.legend(handles=handels, labels=labels, loc="upper left")
        
    def plot():
        plt.show()                    

