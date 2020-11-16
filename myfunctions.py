import cv2
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

def rgbExclusion(image,removeChannel):
    LoadImage=cv2.imread(image,1) #Load Image using cv2 function imread
    
    if removeChannel=='B' or removeChannel=='b': #If the provided color is B or b
        LoadImage[:,:,0]=0 #remove blue from image by setting b to 0
        
    elif removeChannel=='G' or removeChannel=='g': #if the provided color is G or g
        LoadImage[:,:,1]=0 #remove green from image by setting g to 0
        
    elif removeChannel=='R' or removeChannel=='r': #if the provided color is R or r
        LoadImage[:,:,2]=0 #remove red from image by setting r to 0
    
    return LoadImage  # return the resulting image

def displayImage(imgPath):
    fig=plt.figure(figsize=(10, 10))  #Set the plot with figure size 10x10
    columns = 2 #Number of Rows in plot
    rows = 1 #Number of Cols in plot
    
    Grayimage=cv2.imread(imgPath,0) #Load Gray Scale Image (0 Indicates GrayScale)
    Colorimage=cv2.imread(imgPath,1) #Load Colored Image  (1 Indicated Colored Image)
    fig.add_subplot(rows, columns,1) #Show the image in 1st block 
    plt.imshow(cv2.cvtColor(Grayimage,cv2.COLOR_BGR2RGB))
    fig.add_subplot(rows, columns, 2) #Show the image in 2nd block 
    plt.imshow(cv2.cvtColor(Colorimage,cv2.COLOR_BGR2RGB))
    
    plt.show()

def generateMatrix(imgPath):
    gray_img = cv2.imread(imgPath, 0)
    img_shape = gray_img.shape
    img_mat=[]
    
    for i in range(0,img_shape[0]):
        row=[]
        for j in range(0,img_shape[1]):
            pvalue=gray_img.item(i,j)
            row.append(pvalue)
        img_mat.append(row)
    img_mat = np.array(img_mat)
    
    return img_mat

def convolution(imgMatrix,kernalMatrix):
    kernalflipped = np.flipud(np.fliplr(kernalMatrix))
    outMatrix = np.zeros_like(imgMatrix)
    
    Padded_Image= np.zeros((imgMatrix.shape[0] + 2, imgMatrix.shape[1] + 2))
    Padded_Image[1:-1, 1:-1] = imgMatrix
    
    for x in range(imgMatrix.shape[1]):
        for y in range(imgMatrix.shape[0]):
            outMatrix[y, x]=(kernalMatrix * Padded_Image[y: y+3, x: x+3]).sum()

    return outMatrix

def CompareImage(imgPath,imgPath1,title1,title2):
    fig=plt.figure(figsize=(15, 15))  #Set the plot with figure size 10x10
    columns = 2 #Number of Rows in plot
    rows = 1 #Number of Cols in plot
    
    image1=cv2.imread(imgPath,0) #Load Gray Scale First Image (0 Indicates GrayScale)
    image2=cv2.imread(imgPath1,0) #Load Gray Scale Second Image (0 Indicates GrayScale)
    
    fig.add_subplot(rows, columns,1) #Show the first image in 1st block 
    plt.imshow(cv2.cvtColor(image1,cv2.COLOR_BGR2RGB)) 
    plt.title(title1)
    
    fig.add_subplot(rows, columns, 2) #Show the Second image in 2nd block 
    plt.imshow(cv2.cvtColor(image2,cv2.COLOR_BGR2RGB))
    plt.title(title2)
    
    plt.show()

def showHistograms(imgPath):
    img = cv2.imread(imgPath,0)

    fig=plt.figure(figsize=(15, 15))  #Set the plot with figure size 10x10
    columns = 2 #Number of Rows in plot
    rows = 2 #Number of Cols in plot

    fig.add_subplot(rows, columns,1) #Show the image in 1st block 
    plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    plt.title('Before Histogram Equalization')
    fig.add_subplot(rows, columns, 2) #Show the Histogram in 2nd block
    plt.title('Histogram Before')
    plt.hist(img.ravel(),256,[0,256]); 

    equ = cv2.equalizeHist(img) # Apply Histogram Equalization

    fig.add_subplot(rows, columns,3) #Show the Equalized image in 3rd block 
    plt.imshow(cv2.cvtColor(equ,cv2.COLOR_BGR2RGB))
    plt.title('After Histogram Equalization')
    fig.add_subplot(rows, columns, 4) #Show the Histogram 4th block 
    plt.hist(equ.ravel(),256,[0,256]);
    plt.title('Histogram After')

    plt.show()