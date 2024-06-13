# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 13:23:58 2019

@author: svupper
"""

#from PIL import Image

from matplotlib import pyplot as plt
from skimage.color import rgb2gray
from skimage import filters

import numpy as np
import matplotlib.image as mpimg
import time

#def steamConcatenate():
    
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def x_dynamic(image):
    dyn=np.copy(image)
    
    for i in range(1,len(image[:,1])):
        for j in range(0,len(image[1,:])):
            if j==0:
                dyn[i][j]=image[i][j]+np.min((dyn[i-1][j],dyn[i-1][j+1]))
            elif j==(image.shape[1]-1):
                dyn[i][j]=image[i][j]+np.min((dyn[i-1][j-1],dyn[i-1][j]))
            else:
                dyn[i][j]=image[i][j]+np.min((dyn[i-1][j-1],dyn[i-1][j],dyn[i-1][j+1]))            
            
            
    return dyn

def get_seam(image,dyn_image):
    s=np.array([],dtype=int)
    i=dyn_image.shape[0]-1
    j=np.argmin(dyn_image[-1,:])
    s=np.append(s,j)
    
    for i in reversed(range(1,dyn_image.shape[0])):
        if j==0:
            a=np.argmin((dyn_image[i-1,j],dyn_image[i-1,j+1]))
            if a==0:
                j=j
                s=np.append(s,j)
            else:
                j=j+1
                s=np.append(s,j)
        elif j==(dyn_image.shape[1]-1):
            a=np.argmin((dyn_image[i-1,j-1],dyn_image[i-1,j]))
            if a==0:
                j=j-1
                s=np.append(s,j)
            else:
                j=j
                s=np.append(s,j)
        else:
            a=np.argmin((dyn_image[i-1,j-1],dyn_image[i-1,j],dyn_image[i-1,j+1]))
            if a==0:
                j=j-1
                s=np.append(s,j)
            elif a==1:
                j=j
                s=np.append(s,j)
            else:
                j=j+1
                s=np.append(s,j)
    return s

def carving(image,dyn_image,s):
    image_s=np.zeros_like(dyn_image[:,0:(dyn_image.shape[1]-1)])
    s=s[::-1]
    j=s[-1]
    
    if j==0:
        image_s[-1,:]=image[-1,1:(image.shape[1])]
        
    elif j==(image.shape[1]-1):
        image_s[-1,:]=image[-1,0:-1]
        
    else:            
        if j==1:
            image_s[-1,:]=np.concatenate((image[-1,0],image[-1,(j+1):(image.shape[1])]),axis=None)
            
        elif j==(dyn_image.shape[1]-2):
            image_s[-1,:]=np.concatenate((image[-1,0:j],image[-1,-1]),axis=None)
            
        else:
            image_s[-1,:]=np.concatenate((image[-1,0:j],image[-1,(j+1):(image.shape[1])]),axis=None)
        
    for i in reversed(range(dyn_image.shape[0]-1)):
        j=s[i]
        if j==0:
            image_s[i,:]=image[i,1:(image.shape[1])]
            
        elif j==(image.shape[1]-1):
            image_s[i,:]=image[i,0:-1]
            
        else:            
            if j==1:
                image_s[i,:]=np.concatenate((image[i,0],image[i,(j+1):(image.shape[1])]),axis=None)
                
            elif j==(dyn_image.shape[1]-2):
                image_s[i,:]=np.concatenate((image[i,0:j],image[i,-1]),axis=None)
                
            else:
                image_s[i,:]=np.concatenate((image[i,0:j],image[i,(j+1):(image.shape[1])]),axis=None)
            
    return image_s 


def displayGradient(image):
    
    plt.gray()
    plt.figure(figsize=(10,10))
    plt.subplot(221)
    plt.imshow(image)
    plt.xticks([])
    plt.yticks([])
    plt.title('original', size=20)
    plt.subplot(222)
    edges_x = filters.sobel_h(image) 
    plt.imshow(edges_x)
    plt.xticks([])
    plt.yticks([])
    plt.title('sobel_x', size=20)
    plt.subplot(223)
    edges_y = filters.sobel_v(image)
    plt.imshow(edges_y)
    plt.xticks([])
    plt.yticks([])
    plt.title('sobel_y', size=20)
    plt.subplot(224)
    edges = filters.sobel(image)
    plt.imshow(edges)
    plt.title('sobel', size=20)
    plt.xticks([])
    plt.yticks([])
    plt.show()

def display(image):
    plt.figure(figsize=(5,5))
    plt.imshow(image)
    plt.xticks([])
    plt.yticks([])
    plt.show()

if __name__ == '__main__':
        
    image = mpimg.imread('lion.jpg')     
    gray = rgb2gray(image)    
    
#    gray=np.random.random((5, 5))
    image=np.copy(gray)
    display(image)
    
    debug=1
        
    if debug:
        displayGradient(image)
    
    for i in range(100):
        
        display(gray)
        energy = filters.sobel(gray)
        x_dyn=x_dynamic(energy) 
        display(x_dyn)
        s=get_seam(gray,x_dyn)
        
        s=s[::-1]
        
        for j in range(gray.shape[0]):
            
            gray[j][s[j]]=255
            
        display(gray)
        gray=carving(gray,x_dyn,s)
        display(gray)
        time.sleep(1)
        