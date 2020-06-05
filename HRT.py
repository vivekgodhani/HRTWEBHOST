import tensorflow
import urllib
import gzip
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from keras.models import load_model
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.utils import to_categorical
from PIL import Image as im
import sys
import numpy as np
import PIL.ImageOps
from scipy.ndimage import interpolation as inter



img_path='./Save/'
path='./img/'
Binarization_Threshold = 230
HProfile_threshold = 0.85
VProfile_threshold = 0.05

def smoothening(img):
	dst = cv2.GaussianBlur(img,(5,5),0)
	return dst

def Binarization(img):
	threshold = Binarization_Threshold
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img = cv2.GaussianBlur(img,(5,5),0)
	ret,th1 = cv2.threshold(img,threshold,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	return th1


def crop_imgs(img):
	img1=img
	ui=0
	li=0
	ri=0
	di=0
	for i in range(len(img)) :
		for j in range(len(img[i])) :
			if img[i][j]==0:
				ui=i;
				break;
	for i in range(len(img)-1,0,-1) :
		for j in range(len(img[i])):
			if img[i][j]==0:
				di=i;
				break;
	img = img.T
	for i in range(len(img)) :
		for j in range(len(img[i])) :
			if img[i][j]==0:
				li=i;
				break;
	for i in range(len(img)-1,0,-1) :
		for j in range(len(img[i])) :
			if img[i][j]==0:
				ri=i;
				break;
	# print(ui)
	# print(di)
	# print(li)
	# print(ri)
	img = img.T
	a=ui
	b=di
	c=li
	d=ri
	a=ui+20
	if a>len(img):
		a=len(img)
	b=di-20
	if b<0:
		b=0
	c=li+20
	if c>len(img[1]):
		c=len(img[1])

	d=ri-20
	if d<0:
		d=0
	tempimg=img[b:a,d:c]
	img=np.asarray(tempimg)
	tempimg=img1[b:a,d:c]
	img1=np.asarray(tempimg)
	# img4=plt.imshow(img1)
	# plt.show(img4)
	return img1
model1=load_model('./model/cnn_model.h5')
def get_Prediction(img):
	# img4=plt.imshow(img)
	# plt.show(img4)
	img = np.reshape(img,[1,28,28,1])
	y_pred1=model1.predict(img) 
	prediction=np.argmax(y_pred1,axis=1)
	return prediction

#Returns the number formed after appending the digits in the image
def FindBoundary(crop_img):
	rows, cols = crop_img.shape 
	flag=False 
	vertical_sum=[]
	horizontal_sum=[]
	x=0
	y=0
	w=0
	h=0

	for j in range(cols):
		temp_sum = 0
		for i in range(rows):
			temp_sum += crop_img[i,j]
		vertical_sum.append(temp_sum)

	for i in range(rows):
		temp_sum = 0
		for j in range(cols):
			temp_sum += crop_img[i,j]
		horizontal_sum.append(temp_sum) 

	temp = 255*rows
	for i in range(len(vertical_sum)):
		if vertical_sum[i]!=temp:
			x=i
			break 
	for i in range(len(vertical_sum)-1,-1,-1):
		if vertical_sum[i]!=temp:
			w = i
			break 
	temp = 255*cols
	for i in range(len(horizontal_sum)):
		if horizontal_sum[i]!=temp:
			y=i
			break 
	for i in range(len(horizontal_sum)-1,-1,-1):
		if horizontal_sum[i]!=temp:
			h = i
			break 
	return crop_img[y:h,x:w] 

def GetDigits(img):
	rows,cols = img.shape[:] 
	cols_coordinates = [0]
	number=''
	flag = True
	for j in range(0,cols):

		temp_sum = 0
		for i in range(0,rows):
			temp_sum += img[i][j]  

		#Detecting column of all white pixels and then splitting the digits from the image	
		if temp_sum == 255*rows and not flag:
			cols_coordinates.append(j+5)
			flag = True
		if temp_sum!=255*rows and flag:
			flag = False
	if len(cols_coordinates)>0: 
		x = cols_coordinates[0] 
   
	for i in range(1,len(cols_coordinates)):
		w = cols_coordinates[i]
		crop_img = img[:,x:w]
		x = w 
		crop_img = FindBoundary(crop_img)
		crop_img = cv2.resize(crop_img, (20,20)) 
		temp_crop_img = []
		for i in range(28):
			temp = []
			for j in range(28):
				temp.append(0)
			temp_crop_img.append(temp)

		for i in range(20):
			for j in range(20):
				if crop_img[i][j]>127:
					crop_img[i][j] = 255
				else:
					crop_img[i][j] = 0

				if crop_img[i][j]==0:
					temp_crop_img[4+i][4+j] = 255
		pred_digit=get_Prediction(temp_crop_img)
		count=0
		digit=str(pred_digit[0])
		for i in range(20):
			for j in range(20): 
			    if temp_crop_img[4+i][4+j] == 255 :
				    count=count+1
		if count>=(200):
			digit="1"
		number=number + digit 
	return number	
def remove_noise(img):
	rgb_planes = cv2.split(img)
	result_planes = []
	result_norm_planes = []
	for plane in rgb_planes:
			dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
			bg_img = cv2.medianBlur(dilated_img, 21)
			diff_img = 255 - cv2.absdiff(plane, bg_img)
			norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
			result_planes.append(diff_img)
			result_norm_planes.append(norm_img)
	result = cv2.merge(result_planes)
	result_norm = cv2.merge(result_norm_planes)
	img=result
	return img


imageName = "img_8.jpeg"
temp_image_name = imageName.split('.')
if temp_image_name[1] != 'png' and temp_image_name[1] != 'PNG':
	img = im.open(path+imageName)
	img.save(img_path+'Input_img.png')
	imageName = 'Input_img.png'


img=cv2.imread(img_path+imageName)
# img=remove_noise(img)
# cv2.imwrite(img_path+"Remove_Noise.png", img)

# width = int(img.shape[1] * 10 / 100)
# height = int(img.shape[0] * 10 / 100)
# dim = (width, height)
# img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
# for i in range(len(img)-1,len(img)-80,-1):
# 		for j in range(len(img[i])):
# 				img[i][j]=img[len(img)-80][j]
# for i in range(80):
# 		for j in range(len(img[i])):
# 				img[i][j]=img[80][j]

SmoothImage = smoothening(img)
cv2.imwrite(img_path+"SmoothImage.png", SmoothImage) 

BinarizedImage = Binarization(SmoothImage)
cv2.imwrite(img_path+"BinarizedImage.png", BinarizedImage)

# _,mask=cv2.threshold(BinarizedImage,127,255,cv2.THRESH_BINARY)
# k=np.ones((3,3),np.uint8)
# mask=cv2.dilate(mask,k)
# mask=cv2.dilate(mask,k)
# for i in range(0,3):
#   mask=cv2.erode(mask,k)
# img=mask
# BinarizedImage=img

img=crop_imgs(BinarizedImage)
print("Data : ",end="")
print(GetDigits(img))