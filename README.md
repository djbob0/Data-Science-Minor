# Data-Science-Minor
This is the personal portfolio for The Data Science Minor at THUAS by Raphael Pickl


# Table of contents:
- [1. Introduction](#1-Introduction)  

- [2. Learning on Machine Learning](#2-Learning-On-Machine-Learning)
  - [2.1 DataCamp and Udemy Courses](#21-DataCamp-and-Udemy-courses)  
    - [2.1.1 DataCamp](#211-DataCamp)
    - [2.1.2 Udemy](#212-Udemy)
  - [2.2 First Steps With our Data](#21-First-steps-with-our-Data)
  - [2.3 Data Preprocessing](#23-Data-Preprocessing)
    - [2.3.1 Logistic Regression](#231-Logistic-Regression)
    - [2.3.2 Neural Networks](#232-Neural-Networks)
  - [2.4 Understanding Last Groups Work](#24-Understanding-last-groups-work)
  - [2.5 Jupyter Notebooks on Machine Learing](#25-Jupyter-Notebooks-on-Machine-Learing)
- [3. Visualization](#3-Visualization)
  - [3.1 Blender](#31-Blender)
    - [3.1.1 Creating a Wireframe Model in Blender](#312-Creating-a-Wireframe-Model-in-Blender)
    - [3.1.2 Refining the Blender Model](#312-Refining-the-Blender-Model)
    - [3.1.3 Creating our own Protocol](#313-Creating-our-own-Protocol)
      - [3.1.3.1 The Protocol](#3131-The-Protocol)
      - [3.1.3.2 The Outcome](#3132-The-Outcome)
  - [3.2 Matplotlib](#32-Matplotlib)
    - [3.2.1 Plotting the CSV's in 2D](#321-Plotting-the-CSV's-in-2D)
    - [3.2.2 Plotting RAW and CSV files](#3.2.2-Plotting-RAW-and-CSV-files)
  - [3.3 Checking for Flippes Sensors](#33-Checking-for-Flippes-Sensors)
  - [3.4 Poster for Zoetemeer](#34-Poster-for-Zoetemeer)
- [4. Research](#4-Research)
  - [4.1 Answering the Subquestions](#41-Answering-the-Subquestions)
  - [4.2 Writing a Paper](#42-Writing-a-paper)
  - [4.2.1 Starting Structure](#421-Starting-Structure)
- [5. Presentations](#5-Presentations)
- [6. Conclusion and Reflection](#6-Conclusion-and-reflection)

# 1. Introduction

# 2. Learning on Machine Learning
In this chapter my personal development in Machine Learing gets discussed. <br>
Which courses and what actions did I take to improve on my Machine Leaning skills.
## 2.1 DataCamp and Udemy Courses 
## 2.1.1 DataCamp
- Screenshot of completed courses  

In the first weeks of our project, we planned one day a week for the Datacamp courses. Since I already had a small coding project in Python, which was self-thought, the year before, I already knew the basic syntax of the language. Though, having a structured course layed out, which shows one the proper techniques, which are used in the Machinelearning field.  

![completedcourses](https://github.com/djbob0/Data-Science-Minor/blob/master/Machine%20Learining/Datacamp/completed%20courses1.PNG)  

From week to week I felt more secure in the language, and I was able to make use of the API’s that got introduced to us in the course. Especially working in Data Science shows the beauty of an object orientated coding language like python. With the right API’s like Numpy and SKLearn it’s possible to attack unimaginable big problems with just a few lines of code. 

## 2.1.2 Udemy
I was able to score a good deal on one of the Udemy courses on Neural Networks with Tensorflow 2.0. Instead of over 100 Euros it was discounted to around 15 Euros. That's why I thought it wouldn't be a bad idea to improve my ML skills through a different course than the one at THUAS. I didn't want to get better with NN's for nothing, at this point of the project we wanted to start with CNN's, which this course gave a good introduction to.  

- Udemy(Udemy_tenorflow) picture  

For what it's worth, I was able to work through 67% of the course, which was enough to get a good understanding on what neural networks actually are and how they interprete the data we supply.  
By the time I was finished with the course, I was more familiar with machine learning in general. I learned a lot about Model in genral, objective functions, optimazation algorithms, Tensorflow, under- and overfitting, early stopping and of course preprocessing of the data in general.  
- Notes Udemy  
## 2.2 First Steps With our Data
Writing a script to get basic information about the data depending on exercises  

- In Local directory and Getting to know 

After I got to know our data through the experiments in Blender, where at this point I was only able to load in one file at a time, I decided to read some basic information from the data. Instead of only using one file at a time, I wanted to calculate the mean for one axes, the X-axes of the right thorax, for all of the files contained in one given folder. Eventhough the mean of a dataframe can easily be calculated with the df.mean() function, I found this exercise really helpful, because it helped me understand the datastructure and it's dimensions. Also it got me used to working with classes and functions without getting to complicated for the start.

 
```python
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
```
In this script there's only used two functions: get_data and get_mean.  
Get_data makes use of the parser class, to get back a list of all the dataframes in the directory. get_mean calculates the mean of one column in a dataframe.  
At the end of the script I simply loop trough the whole datalist to get all the means of all the files. The printout isn't really pretty, but since it was more a proof of concept it worked for me:  
```
  python3 d:/Hochschule/5_Semester/Orthoeyes/Portfolio/Machine Learining/getting_to_know/test.py  
  7.107194803858595  
  4.41344972577624  
  0.9291094629136488  
  -2.493449629363173  
  -0.9359113076469754  
  5.993412983976696  
```



## 2.3 Data Preprocessing

## 2.3.1 Logistic Regression
- what sprint?  


At this point of time, I already worked with our data quite a bit, I plotted some easy graphs, I created my own anymations in Blender, and I could load in different files at once.  

Now I wanted to load in the data to feed it to any ML model, but not by storing all the data in the memory, but by saving only the paths to the data in memory, to safe a lot of computing power. The whole groups agreed on this standard, so it was everybodys responsibility to get used to coding like this.   

In main_raphi.py I was able to make use of the already written controller scripts, that were implemented in the main branch of our GitHub repository. Unfortunatly I didn't get further than just loading in the data. 

```python
from controller.datacontroller import DataController

from ml.models import Models
from ml.logisticregression import LogisticRegressionModel
from ml.svc import SVCModel


class config:
    debug = False
    tables = True
    pp = pprint.PrettyPrinter(indent=4)

    exercises = 5
    workers = 20
    max_chunck_size = 100

    test_size = 0.2
    test_random_state = 42

    if debug:
        basepath = "src\\data\\cleaned-regrouped-small\\"
    else:
        basepath = "src\\data\\cleaned-regrouped\\"

print('ORTHO: Prepairing Dataset')

controller = DataController(config)
controller.run()
```

The moment I got this script working, our main structure in the masterbranch changed again, so I continued working with the code that was already supplied by my colleges. In src2/main. A np_combination_train and np_combination_test gets created. Those are X and y for the most of the machinelearning models. This is the first time, I used this data structure that was planned for the model from the beginning, even from the last group.  

- Explain the datastrucure here:  

Here is an explanation on the data strucure by one of my colleges. Since I’ve been working with the 65 column data structure to do visualizations and whatnot, I can say with certainty, that I understood the structure and know how it gets created.  

## 2.3.2 Neural Networks

Also to mention is that  I prepared some data together with Hassan, not for the Logistic-Regression-Model but for the CNN we’ve been training.  

- 927ce917  


At first it was thought, that we’re going to use an RNN as an unsupervised training method, but we decided to use a CNN instead. That is because of the data structure we’re working with. Since there is, for every bone in every patient, an X Y and Z Euler-Angle, we can put these three values in one RGB pixel. Eventhough it isn’t necessary for a Nural Network to have an Input like this, we thought it would be a nice way of preprocessing the data, without loosing any of the information. In order to achieve something like this, the rotation first need to be normalized and the mapped on a scale from 0 to 1  

- Picture of datastructure RGB  

As it is already described in the picture above, the “width” of the inputdata are the eight bones of the upper body, for 5 different exercises performed by one patient. Therefore one picture is one iteration of all the exercises one Patient has done( AB1, AF1,RF1,AH1…) the arrays shape is (40,100,(3)) Where 40 is 8 bones x 5 exercises and the 100 is the resampled framescount(length of exercise) the third(3) dimension is represented in the picture by the colour at the given pixel. These tensors then get stored in one bigger Tensor, which then gets fed to the CNN.

I worked with Hassan, to get a good ground for the CNN's. The moment we got a somewhat decent model running, we decided, that only one person should contiunue on the subject. That is why, the outcomes you can see here, are not the best, but they are mine!

```concole
"d:/Hochschule/5_Semester/Orthoeyes/Portfolio/Machine Learining/NN/src/CNN.py"
Importing patients from: d:\Hochschule\5_Semester\Orthoeyes\Portfolio\Machine Learining\NN\data/Category_1
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 30/30 [00:11<00:00,  2.67it/s]
Importing patients from: d:\Hochschule\5_Semester\Orthoeyes\Portfolio\Machine Learining\NN\data/Category_2
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 39/39 [00:10<00:00,  3.59it/s]
Importing patients from: d:\Hochschule\5_Semester\Orthoeyes\Portfolio\Machine Learining\NN\data/Category_3
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 37/37 [00:10<00:00,  3.58it/s]
train_combinations:  819
test_combinations:  275
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 275/275 [00:02<00:00, 126.37it/s]
(275,)
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 819/819 [00:19<00:00, 42.68it/s]
2020-01-12 12:19:31.812072: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 97, 37, 16)        784
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 48, 18, 16)        0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 45, 15, 32)        8224
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 22, 7, 32)         0
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 19, 4, 32)         16416
_________________________________________________________________
flatten (Flatten)            (None, 2432)              0
_________________________________________________________________
dense (Dense)                (None, 32)                77856
_________________________________________________________________
dense_1 (Dense)              (None, 3)                 99
=================================================================
Total params: 103,379
Trainable params: 103,379
Non-trainable params: 0
_________________________________________________________________
Train on 819 samples, validate on 275 samples
Epoch 1/10
2020-01-12 12:19:35.194058: I tensorflow/core/profiler/lib/profiler_session.cc:184] Profiler session started.
819/819 [==============================] - 2s 3ms/sample - loss: 0.6006 - accuracy: 0.7888 - val_loss: 0.3144 - val_accuracy: 0.9309
Epoch 2/10
819/819 [==============================] - 1s 2ms/sample - loss: 0.4129 - accuracy: 0.8376 - val_loss: 0.2905 - val_accuracy: 0.9200
Epoch 3/10
819/819 [==============================] - 2s 2ms/sample - loss: 0.2165 - accuracy: 0.9304 - val_loss: 0.1461 - val_accuracy: 0.9455
Epoch 4/10
819/819 [==============================] - 1s 2ms/sample - loss: 0.1013 - accuracy: 0.9597 - val_loss: 0.1298 - val_accuracy: 0.9527
Epoch 5/10
819/819 [==============================] - 1s 2ms/sample - loss: 0.0660 - accuracy: 0.9707 - val_loss: 0.3843 - val_accuracy: 0.8873
Epoch 6/10
819/819 [==============================] - 1s 2ms/sample - loss: 0.0506 - accuracy: 0.9744 - val_loss: 0.1434 - val_accuracy: 0.9200
Epoch 7/10
819/819 [==============================] - 2s 2ms/sample - loss: 0.0448 - accuracy: 0.9756 - val_loss: 0.1129 - val_accuracy: 0.9455
Epoch 8/10
819/819 [==============================] - 1s 2ms/sample - loss: 0.0306 - accuracy: 0.9866 - val_loss: 0.1464 - val_accuracy: 0.9164
Epoch 9/10
819/819 [==============================] - 1s 2ms/sample - loss: 0.0322 - accuracy: 0.9841 - val_loss: 0.1686 - val_accuracy: 0.9055
Epoch 10/10
819/819 [==============================] - 1s 2ms/sample - loss: 0.0209 - accuracy: 0.9976 - val_loss: 0.4133 - val_accuracy: 0.8945
275/1 - 0s - loss: 0.7360 - accuracy: 0.8945
0.89454544
```  

- add NN train_vs_test_acc

As I said, it is not the best possible outcome for a model, but these results let us guess, that there is sufficient informarion in the data given. Now it was in Hassans hands to find the best architecture for the CNN, eventough he often relied on my educated guess on whatever the outcome of a model was good or not. 


## 2.4 Understanding last Groups Work

- described by task 122 partly   


In order to get everybody on the same page with their machine learning skills by week 10, we created a task, as a researcher I want to understand the steps the last group took. Therefore we created a Excel sheet to keep track on progress.   

- Excel sheet with progress  

This task contains multiple subtasks, that have either been completed just to complete the subtask or in order to accomplish something else in the project. Here I will give links to the sections in my portfolio where I completed those tasks.  
- add links 

## 2.5 Jupyter Notebooks on Machine Learing

- Done till 3.3 auto MPG regression  

- What kind of proof?  
- Is it necessary?  

In preparation for the test, I started working through all the Machinelearing Notebooks provided on the datascience server. Therefore, I simultaneously had the lecture and the notebook open, this way I was able to understand and see the techniques used in the lectures by myself on my own computer. The given problems were built up gradually and of course followed the weekly lectures in the same paste.  

Since our Dataset was quite different than for example the MNIST or other examples used throughout the lectures, it was quite hard to make the transition from big and structured datasets to ours, which is really small and only contains features that are from the same type. Nevertheless it was a good practice and it helped me understand the bigger picture of Data Science. 


# 3. Visualization

## 3.1 Blender

## 3.1.1 Creating a Wireframe Model in Blender

- Master branch contains all blend files  

Before any sort of a model could be created, it was upon me, to figure out what the data actually means. With some information from the LUMC and a paper describing the WU standard for Euler Angles in bone structures, I came up with a factsheet to describe which columns is responsible for which bone and in this bone which axes.  

- factsheet
- Description of the factsheet:  

Afterwards, the struggle began to find the best “resting position” for the armature(the skeleton). This basically went down through trial and error. I first started with only one side(right side) and if I saw movement that made sense for my eyes I tried to construct the left arm based on the right one. 



## 3.1.2 Refining the Blender Model

- Commit in task  
- Add Blender Screenshots here  
- Add How to Blend

In the second phase of the refinement, I added functions, to sort the different armatures in the scene, by either their exercise- or patient-group. I also automated the script to load in multiple files at the same time. Basically the script was created to function by the click of one button.  
```python
for obj in bpy.data.collections['Result'].objects:
    if grouping == 'cat':
        name = obj.name[:4]
    if grouping == 'pat':
        name = obj.name[5:10]
    if name not in col:
        newCol = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(newCol)
        col.append(name)
    bpy.data.collections[name].objects.link(obj)
    bpy.data.collections['Result'].objects.unlink(obj)    

```

The user is able to choose which files he/she wants to read in, and by what the user wants to order the different “patients”.  
```python
rotation_mode = 'XYZ'
grouping = 'cat'
directory = 'D:\\Hochschule\\5_Semester\\Orthoeyes\\Data\\test-data\\'
```
After all the animations have been created, one has the possibility to switch into the orthographic view, in order to compare the different “patients” without having any perspective issues.  

- screenshots in PNG folder

This way of view the “patients” enables the possibility to see the exercises in 2D and 3D but also with and without perspective(with perspective helps to understand the movement, without makes it easy to compare the “patients to eachother”), all this can be done in realtime and full moveability of the camera in the same viewport.  

## Result:  
The Euler angles were in the way of getting a proper visualization. We also didn’t want to make more assumptions than we needed too. The next chapter goes more in depth on what the results were. 

- 3.1.3.2 The Outcome


## 3.1.3 Creating our own Protocol

### 3.1.3.1 The Protocol

- Scan in pages of LUMC  

In order to validate the 3D visualization made in Blender, we either needed to check back with our client at the LUMC and get a verification that the animations are correct. The other possibility was, to create out own exercises, which one of the group members would perform while hooked up to the same Flock of Birds system, that has been use to collect the patientdata we received, at the LUMC.  

To make sure, that the exercises will be performed exactly how I envisioned them beforehand, I created a protocol. This protocol contains hand drawn descriptions of the exercise and also step by step instructions for the “patient”. We didn’t want to rely on filenames and the protocol, we also filmed all the execises that had been performed by our group at the LUMC.  
- Protocol here

### 3.1.3.2 The Outcome

- Files locally   

Once we visited the LUMC and recorded our own movement data, It was about time to validate the script I created earlier. The first step was, to cut the recorded video files to the right length and name them with the corresponding filenames of the .CSV – files.  

Even though we had a protocol, to keep track in which order the tasks have been performed, it still was wearisome work, because this wasn’t the time for mistakes, since the recording should proof the reliability of the script that was planned to be used to label our dataset. To Accomplish this task I used Adobe Premiere.  
- filled protocoll scan in 

After hours of trial and error, I came to the conclusion, that I won’t be able to get a proper representation of the data, using Blender. That’s because of the Euler Angles that are used to describe the patients movement. In the Euler Angles nature lays, that the three angles can only describe ONE rotation in 3d space. This only applies, if the three axes are staying in the same order. This being said: X = 34°, Y = 45°,  Z = 180° is different to : Z = 180°, X = 34°, Y = 45°. 
For whatever reason, this order needed to be switched for different movements in Blender. But I could re the corealation and I didn't want to make any assumptions, because this script was seen as a tool to find errors in the data.

## 3.2 Matplotlib

## 3.2.1 Plotting the CSV's in 2D
- python\src2\fd\visualize.py  
-  PG1-3 vs 4  
-  Commit 42948dc5  

After I made my first contact with the data in Blender and some minor experiments with Machinelearning, I started creating some scatterplots, to further understand the data.  

```python
from fd.visualize import Visualise_patient

Visualise_patient(patient_group, patientnr=[1],right = True, left = False, linelabel= True)
Visualise_patient.plot()
```

In src2\main.py a visualization gets shown, after the training and test data gets created. This visualization shows the data exactly how it gets handed to a machine learning model.  

- PNG CSV 2D


This graph was more a proof of concept, than an actual tool for visualization. This came, when I integrated the Visualization class into the main branch. At this point of the project we realized that patientgroup 4 we received from the LUMC wasn’t what we expected it to be. The values in pg4 were completely different, then the ones for pg 1-3. 

- master2.0  tools/visualis.py
- screenshot of configurations for the Visualize class  

In order to contribute to the project, I created a class Visualize in the master branch, that anybody could make use of, just by turning it on in the config and by changing a few parameters in the definition of the class. 

```python
if config.show_visualization:
    vv = Visualize(patient_groups,catagory = [1,2,3], patients=[1,2,3,4,5,6,7,8], exercises=['RF','AF'], bones=["thorax_r_x_ext", "thorax_r_y_ax", "thorax_r_z_lat",
                      "clavicula_r_y_pro", "clavicula_r_z_ele", "clavicula_r_x_ax"])
    vv.visualise(mode='exercise')
    vv.visualise(mode = 'idle') 
```

The user can choose which Petientgroups, out of these groups which patients, what exercises and what bones should be displayed in the graphs. If none of the parameters gets set, the script will grab all the information out of the config. Also the length of the exercise doesn’t matter, since the script gets all it’s information out of the config.  

- visualize exercise  

The visualize_exercise function compares different exercises between Patientgroups and patients and bones.

- visualize idle  

The visualize_idle function was created to help validate Hassans script to remove the idle at the end and the beginning of every exercise. In this graph one patients exercise gets it's own subgraph, keep in mind there are most of the times two itterations of every exercise, thats why two line are plotted in the subplot. Therefore the two line have a different color, just as the responding "idle" lines.


## 3.2.2 Plotting RAW and CSV files

- master 2.0 tools/viusalizeRAW 

All the efforts to get a real 3D visualization in Blender didn’t bring any good results and just let us continue guessing. Since Assumptions aren’t any good in machine learing, we needed a visualization that we could trust. Therefore Eddie wrote a script, based on the last groups work, to visualize the RAW files we received from the LUMC. I used this as a base for my script. 

- GIF of visualization

This script was most and foremost created to understand fully what exercises contain which movent in Euler Angles. Therefore replacing the effort that was made to create a 3D animation in Blender.  
This script also served as the base for work that was done by Lennart to find wrongly named exercises in the files. 


## 3.3 Checking for Flippes Sensors
- answer in issue still

For anomaly detection we wanted to make sure, that the way the sensores are attached to the patients doesn't mess with the data. Therefore we did one set of exercises at the LUMC with a flipped sensor on the right humerus. 

- raphi flipped vs raphi 

Upon visual inspection there is no significant difference between the first and the second recording of Raphaels normal exercises.(keeping in mind, that the sensor was flipped on the Humerus alone)

## 3.4 Poster for Zoetemeer

Our group was asked to show our project at the campus at Zoetemeer, for this event I designed a poster. 
- poster v7  

# 4. Research

## 4.1 Answering the Subquestions
At the beginning of the project, we had some lectures about research, in these lectures we learned how to come up with a researchquestion and how we can asnwer it by using more simplyfied subquestions. 
Since we didn't have much domain knowledge in the beginning, it made a lot of sense to start working on the library and field questions.  

- subquestions.png

It was everybodys task to pick at least on of the question, find a researchpaper that describs the problem or even answers it. These Results should be saved somewhere, so that the other group members have acces to the summaries. 

- link to excel sheet  


## 4.2 Writing a paper



## 4.2.1 Starting Structure
I was the first person in the group who started working on the research paper. Thats why one of my first tasks was, to understand what a researchpaper actually is and how one goes about writing one. 

- paper here  


In Paper_guide_by_Raphi.docx is a brief summary of "how to write a good paper" with some more in depth explanation of for example difficult terms. 

I also came up with a general guide on how the writing process should go about. 

```
General Strategy:
-Gather all the information we need for the paper. 
-Start with the conclusion and work from there to the top. 
-Planned are 5 iteration in order to get to a finished state of the paper
1. structure
2. dropping anything in the paper(start writing sentences)
3. write actual sentences and add figures
4. check references etc.
5. finished paper, check typos
```
For the first point: "1. structure" I started with a very simple version and improved it over multiple itteration. These files can be found  `here`.
# 5. Presentations
# 6. Conclusion and Reflection
