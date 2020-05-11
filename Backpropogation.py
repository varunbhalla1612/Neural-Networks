# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 13:23:12 2020

@author: Varun
"""
###########import libraries#############
import os
from os import listdir
from os.path import isfile, join
import struct
import numpy as np
import random
import operator
import matplotlib.pyplot as plt
import gzip
from numpy.linalg import inv
import math
###########################################
#1. Draw n = 300 real numbers uniformly at random on [0, 1], call them x1, . . . , xn.
X=[]
for i in range(0,300):
    X.append((random.uniform(0, 1)))

#Draw n real numbers uniformly at random on [-1/10,1/10] call them ν1, . . . , νn.
V=[]
for i in range(0,300):
    V.append((random.uniform((-1/10), (1/10))))

#3. Let di = sin(20xi) + 3xi + νi, i = 1, . . . , n. Plot the points (xi, di), i = 1, . . . , n.
D=[]
for i in range(0,300):
    D.append(math.sin(20*X[i])+(3*X[i])+ (V[i]))    

plt.scatter(X,D,color='red', marker='o');

"""
#########################################
def activationphi2(v):
    return(v)
def derivateactivationphi2(v):
    return(1)

def activationphi(v):
    return(np.tanh(v))
def derivativeactivationphi(v):
    return(1-(np.tanh(v)*np.tanh(v)))

########"Input Layer"#####################
"""
def activationphi2(v):
    return(v)
def derivateactivationphi2(v):
    return(1)

def activationphi1(v):
    return(0.8*np.tanh(v))
def derivateactivationphi1(v):
    return(.8*(1-(np.tanh(v)*np.tanh(v))))

#learning rate
eta2=0.05
eta1=0.01
X=np.array(X)
X=(X-np.mean(X))/(np.std(X))
#X.shape
#weights initialisation
W1 =[[ round(random.uniform(-1,1),5) for y in range(2)] for x in range(24) ]
W1=W1-np.mean(W1)
W1=W1/np.std(W1)
W2=[[ round(random.uniform(-1,1),3) for y in range(25)] for x in range(1) ]  
W2=W2-np.mean(W2)
W2=W2/np.std(W2)

W1=np.array(W1)
W2=np.array(W2)
          
MSElist=[]
c=0

while True:
    ListofDs=[]
    ListofYs=[]
    for i in range(0,len(D)):  
        ##############################FORWARD PROPOGATION############################
        ##############################FORWARD PROPOGATION############################
        ##############################FORWARD PROPOGATION############################
        ##############################FORWARD PROPOGATION############################
        ####################LAYER 1 ###########################
        ####################LAYER 1 ###########################
        ########"W1  Input Layer-Mid Layer#####################
        #W1=np.array(W1)
        W1.shape
        ########"V1" Induced Local Field ######################
        V1=np.matmul(W1,[1,X[i]])
        V1.shape
        ########"Y1" After Activation Func of Local Field #####
        Y1=np.array([activationphi1(x) for x in V1])
        Y1.shape
        ####################LAYER 2 ###########################
        ####################LAYER 2 ###########################
        ########"W2  Mid Layer-Output Layer####################
        #W2=np.array(W2)
        W2.shape
        ########"V2" Induced Local Field #####################
        V2=np.matmul(W2,np.array([1]+list(Y1)))
        V2.shape
        ########"Y2" After Activation Func of Local Field #####
        Y2=np.array([activationphi2(x) for x in V2])
        Y2.shape
        ##############################Backward PROPOGATION############################
        ##############################Backward PROPOGATION############################
        ##############################Backward PROPOGATION############################
        lambda2=(D[i]-Y2)*derivateactivationphi2(V2)
        ToUnderline=np.matmul(W2.transpose(),lambda2)
        lambda1=np.matmul(np.array(ToUnderline[1:]),derivateactivationphi1(V1))
        W2=W2+eta2*(lambda2*np.array([1]+list(Y1)).transpose())
        W1=W1+eta1*(lambda1*np.array([1]+[X[i]]).transpose())
        #############################################################################
        ListofYs.append(Y2[0])
        ListofDs.append(D[i])
        #print("Y2 ",Y2[0])
        #print("D ",D[i])
    print("Epoch  ",c )
    ##calculate the total mean square error i.e (d-y)^2
    MSE=sum([(x[0]-x[1])**2 for x in list(zip(ListofYs,ListofDs))])/(2*len(D))
    print("MSE IS",MSE)
    MSElist.append(MSE)
    c=c+1
    if MSE>MSElist[-1]:
        eta2=0.9*eta2
        eta1=0.9*eta1
    if MSE<0.02:
        break

"""##############after we get optimal weights#####################
XNew=[i for i in range(0,1)] 
XNew=np.linspace(0,1, num=1000) 
XNew=(XNew-np.mean(XNew))/np.std(XNew)
Outp=[]
i=0
for i in range(0,len(XNew)):
    V1=np.matmul(W1,[1,XNew[i]])
    Y1=np.array([activationphi1(x) for x in V1])  
    V2=np.matmul(W2,np.array([1]+list(Y1)))
    Y2=np.array([activationphi2(x) for x in V2])  
    Outp.append(Y2)   
plt.scatter(XNew,Outp,color='blue', marker='x');
plt.scatter(X,D,color='red', marker='o');
"""
print("eta value for layer 2", eta2)
print("eta value for layer 1", eta1)
Epochs=np.linspace(0,len(MSElist), num=len(MSElist))
plt.scatter(Epochs[:5000],MSElist[:5000],color='green', marker='x');
 
W1