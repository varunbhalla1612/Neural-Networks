# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 17:57:54 2020

@author: Varun
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 18:32:44 2020

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
from numpy.linalg import inv
#################Gradient Descent#####################
lr=0.0005# learning rate
x=round(random.uniform(0.1, 0.3),2)
y=round(random.uniform(0.1, 0.3),2)
print("Initial Points",(x,y))
Ix=x
Iy=y
f=-np.log(1-x-y)-np.log(x)-np.log(y)
Listoff=[]
Listofx=[]
Listofy=[]
for i in range(0,200):    
    #print("F",f)
    #print("x",x)
    #print("y",y)
    f=-np.log(1-x-y)-np.log(x)-np.log(y)
    gfx = ((2*x)+y-1)/((x)*(1-x-y)) 
    gfy = ((2*y)+(x)-1)/((y)*(1-x-y))       
    x=x-(lr*gfx)
    y=y-(lr*gfy)
    Listoff.append(f)
    Listofx.append(x)
    Listofy.append(y)
r=[i for i in range(0,len(Listoff))]
plt.scatter(r,Listoff, marker='o');
#plt.scatter(Listofx,Listofy, marker='x');

###################Newton's Method##############################
x=Ix
y=Iy
lr=lr-((lr*lr)/2)
lr=0.05
f=-np.log(1-x-y)-np.log(x)-np.log(y)
print("Initial Points",(x,y))

def GradientMat(x,y):
    A=((2*x)+y-1)/((x)*(1-x-y))
    B=((2*y)+(x)-1)/((y)*(1-x-y))
    return([[A],[B]])

def HessianCal(x,y):
    A=(1/((1-x-y)**2))+(1/x**2)
    B=(1/((1-x-y)**2))
    C=(1/((1-x-y)**2))
    D=(1/((1-x-y)**2))+(1/y**2)
    return([[A,B],[C,D]])

Listoffn=[]
Listofxn=[]
Listofyn=[]
i=1
for i in range(0,200):    
        f=-np.log(1-x-y)-np.log(x)-np.log(y)        
        tmp1=HessianCal(x,y)
        tmp2=GradientMat(x,y)
        Calc=np.matmul(inv(tmp1),tmp2)  
        x=x-(lr*Calc[0])[0]        
        y=y-(lr*Calc[1])[0]
        Listoffn.append(f)
        Listofxn.append(x)
        Listofyn.append(y)
r=[i for i in range(0,len(Listoffn))]
plt.scatter(r,Listoffn, marker='o');
#plt.scatter(Listofxn,Listofyn, marker='x');

plt.scatter(r,Listoff,color='red', marker='o');
plt.scatter(r,Listoffn,color='b', marker='x');
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 13:00:29 2020

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
from numpy.linalg import inv
#################Gradient Descent#####################

X1=[i for i in range(1,51)]
X0=[1 for i in range(1,51)]
X=[X0,X1]
Y=[[i+(random.uniform(-1, 1))] for i in range(1,51)]
savY=Y
plt.scatter(X1,Y, marker='o');
Y=np.array(Y).transpose()
X=np.array(X)

Y.shape
X.shape

"""
Function is y = w0 + w1x
Minimize (yi − (w0 + w1xi))2 over i=1 to 51
"""
W0,W1=np.matmul(Y,np.matmul(X.transpose(),inv(np.matmul(X,(X).transpose()))))[0].tolist()
print("Using Pseudo Inverse, Value of Wo and W1",(W0,W1))
print("Line Plot with the following weights")
x = np.linspace(0, 50,50)
newy=W0+(W1*x)
W0,W1
plt.scatter(X1,Y, marker='o',s=10);
plt.plot(x, newy, color='red');

###################Gradient Descent###########################
X1
Y=savY
lr=0.02
#using the same x and y values of the function
w0=random.uniform(-5, 5)
w1=random.uniform(-5, 5)

Listoffunc=[]
Listofw0val=[]
Listofw1val=[]
for i in range(0,50):    
    xtemp=X1[i]
    ytemp=Y[i][0]
    #print("F",f)
    #print("x",x)
    #print("y",y)
    w0=w0+(lr*2*(ytemp-w0-(w1*xtemp)))
    w1=w1+(lr*2*(ytemp-w0-(w1*xtemp)))
    Listoffunc.append((xtemp,ytemp))
    Listofw0val.append(w0)
    Listofw1val.append(w1)
    
xgd = np.linspace(0, 50,50)
newygd=w0+(w1*xgd)
w0,w1
plt.plot(xgd,newygd, color='green', linewidth=5);
plt.scatter(X1,Y, marker='o',s=15);
plt.plot(x, newy, color='red', linewidth=2);
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 14:04:33 2020

@author: Varun
"""
###########Just to chck gradient descent working #############
X1
Y=savY
lr=0.002
#using the same x and y values of the function
w0=random.uniform(-5, 5)
w1=random.uniform(-5, 5)

Listoffunc=[]
Listofw0val=[]
Listofw1val=[]
i=0
for i in range(0,50):    
    xtemp=X1[i]
    ytemp=Y[i][0]
    #print("F",f)
    #print("x",x)
    #print("y",y)
    w0=w0+(lr*2*(ytemp-w0-(w1*xtemp)))
    w1=w1+(lr*2*(ytemp-w0-(w1*xtemp)))
    Listoffunc= ytemp -(w0+(w1*xgd))
    Listofw0val.append(w0)
    Listofw1val.append(w1)

Listoffunc
plt.scatter(X1,Listoffunc, marker='o',s=10);
