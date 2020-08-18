# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 21:45:13 2020

@author: Varun
"""
##########importing libraries###################
import os
from os import listdir
from os.path import isfile, join
import struct
import numpy as np
import random
import operator
import matplotlib.pyplot as plt
import gzip
###########changing working directory################################
mypath=r'C:\Users\Varun\Desktop\Neural Networks\hw2'
os.chdir(mypath)
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
##########reading training data & labels######################################
with gzip.open('train-labels-idx1-ubyte.gz', 'rb') as f:
    zero, data_type, dims = struct.unpack('>HBB', f.read(4))
    shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
    train_labels=np.fromstring(f.read(), dtype=np.uint8).reshape(shape)
with gzip.open('train-images-idx3-ubyte.gz', 'rb') as f:
    zero, data_type, dims = struct.unpack('>HBB', f.read(4))
    shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
    train_images=np.fromstring(f.read(), dtype=np.uint8).reshape(shape)
#############labels##########################################################
train_labels_coded=list(train_labels)
train_labels_coded_n_values = np.max(train_labels_coded) + 1
OneHotEncoded_train_labels_coded=np.eye(train_labels_coded_n_values)[train_labels_coded]
######################initialise weights with random values###############################
SizeofInputVectors=train_images[0].size
SizeofOutputVectors=max(train_labels)-min(train_labels)+1
Weights=[]
for i in range(0,(SizeofOutputVectors)):
        Weights.append([round(random.uniform(-1, 1),2) for j in  range(0,SizeofInputVectors)])
##########prepared data set#######################3
training_data_processed=[]
for i in range(0,len( train_images)):
    temp=train_images[i].tolist()
    merged=[]  
    for j in temp:
        merged=merged+j   
    training_data_processed.append(merged)

#############################################################
print("Dim of Weights",len(Weights[0]),len(Weights))
print("Dim of training data",len(training_data_processed[0]),len(training_data_processed))
print("Dim of train labels",len(OneHotEncoded_train_labels_coded))
###################configure number of data points (n) and eta###################################3
###########################################################################################
###########################################################################################
eta=1
threshold=11#missclassificaton rate is alwasy less than this
n=60000#no. of datapoints
###########################################################################################
###########################################################################################
###########################################################################################

print("eta",eta)
print("threshold or maximum error rate",threshold)
print("data points or value of n",n)

training_data_processed=training_data_processed[:n]
OneHotEncoded_train_labels_coded=OneHotEncoded_train_labels_coded[:n]

epoch=0
error=0
missclassificationrate=[]
for k in range(0,len(training_data_processed)):
    trainingimage=training_data_processed[k]
    ImagePred=[(np.dot(trainingimage,Weights[j])) for j in range(0,SizeofOutputVectors)]
    maxval=max(ImagePred)
    Output=[1.0 if j==maxval and j>0 else 0.0 for j in ImagePred ]
    try:
        indexof1inOutput=Output.index(1)
    except:
        #print(ImagePred)
        indexof1inOutput=np.nan
        
    indexofcorrect=OneHotEncoded_train_labels_coded[k].tolist().index(1)
    if indexofcorrect!=indexof1inOutput:
        error=error+1
missclassificationrate.append(error)
print("Missclassifications with random initial weights",error)
print("Missclassification rate with random initial weights",(error/n)*100)

##########################BEGIN TRAINING###################################
epoch=1
while True:
    error=0
    for k in range(0,len(training_data_processed)):
        trainingimage=training_data_processed[k]
        ImagePred=[(np.dot(trainingimage,Weights[j])) for j in range(0,SizeofOutputVectors)]
        maxval=max(ImagePred)
        Output=[1.0 if j==maxval and j>0 else 0.0 for j in ImagePred ]    
        try:
            indexof1inOutput=Output.index(1)
        except:
            #print(ImagePred)
            indexof1inOutput=np.nan    
        indexofcorrect=OneHotEncoded_train_labels_coded[k].tolist().index(1)    
    
        if indexofcorrect!=indexof1inOutput:
            error=error+1
            
            D=OneHotEncoded_train_labels_coded[k].tolist()
            map_object = list(map(operator.sub, D, Output))        
            Weightsprior=Weights       
            
            #####weight adjustment####################### 
            WeightsNew=[]
            for wtx in range(0,len(Weights)):
                partb=np.dot(eta*map_object[wtx],trainingimage)
                parta=Weights[wtx]
                WeightsNew.append(list(map(operator.add, parta, partb)))
            Weights=WeightsNew
        #else:
            #don't adjust weight
    missclassificationrate.append(error)
    epoch=epoch+1
    print("epoch.....",epoch)
    print("error.....",error)
    
    if (error/n)*100 <= threshold:
        break

print("Training Error Rate", (error/n)*100)
print("Missclassifications",missclassificationrate)
print("Epochs",[i for i in range(1,epoch+1)])

#################################################################
FinalWeights=Weights
#################################################################
#################################################################
#################################################################
#################################################################
######################testing###################################

onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
with gzip.open('t10k-labels-idx1-ubyte.gz', 'rb') as f:
    zero, data_type, dims = struct.unpack('>HBB', f.read(4))
    shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
    test_labels=np.fromstring(f.read(), dtype=np.uint8).reshape(shape)
    

with gzip.open('t10k-images-idx3-ubyte.gz', 'rb') as f:
    zero, data_type, dims = struct.unpack('>HBB', f.read(4))
    shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
    test_images=np.fromstring(f.read(), dtype=np.uint8).reshape(shape)

###########labels################
test_labels_coded=list(test_labels)
test_labels_coded_n_values = np.max(test_labels_coded) + 1
OneHotEncoded_test_labels_coded=np.eye(test_labels_coded_n_values)[test_labels_coded]

##########Prepare data set#######################3
testing_data_processed=[]
for i in range(0,len( test_images)):
    temp=test_images[i].tolist()
    merged=[]  
    for j in temp:
        merged=merged+j   
    testing_data_processed.append(merged)

len(FinalWeights)
len(FinalWeights[0])

len(testing_data_processed)
len(testing_data_processed[0])

len(OneHotEncoded_test_labels_coded)
#####################################################
error=0
for k in range(0,len(testing_data_processed)):
    testingimage=testing_data_processed[k]
    ImagePred=[(np.dot(testingimage,FinalWeights[j])) for j in range(0,SizeofOutputVectors)]
    maxval=max(ImagePred)
    Output=[1.0 if j==maxval and j>0 else 0.0 for j in ImagePred ]
    
    try:
        indexof1inOutput=Output.index(1)
    except:
        #print(ImagePred)
        indexof1inOutput=np.nan
        
    indexofcorrect=OneHotEncoded_test_labels_coded[k].tolist().index(1)
    
    if indexofcorrect!=indexof1inOutput:
        error=error+1

print("TestingSet Errors", error)
print("TestingSet Error Rate", error/len(testing_data_processed)*100)
#####################questions###############################################
"""
Missclassifications = [52980, 8753, 7380, 7213, 6979, 6948, 6789, 6792, 6756, 6642, 6681, 6627, 6609, 6528]
Epochs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
plt.scatter(Epochs, Missclassifications)
plt.title('Epoch vs Misclassifications')
plt.xlabel('Missclassifications')
plt.ylabel('Epochs')
plt.show()
"""

