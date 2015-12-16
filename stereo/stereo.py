import cv2
import cv2.cv as cv
import sys
import numpy as np
from PIL import Image
import pylab as lab

def np_from_image(pic):
	return np.asarray(Image.open(pic), dtype=np.float32)

def imageGinput():
	x1 = Image.open('left.png')
	fig1 = lab.figure(1)
	ax1 = fig1.add_subplot(111)
	ax1.imshow(x1)
	ax1.axis('image')
	ax1.axis('off')
	x = fig1.ginput(2)

	fig1.show()
	print(x)

def sumOfAbsoluteDifferences(arr1, arr2):
	output = cv2.absdiff(arr1, arr2)
	total = 0
	shape = output.shape
	height = shape[0]
	width = shape[1]
	print("Height = " + str(height))
	print("width = " + str(width))
	for row in range(0, height-1):
		for col in range(0, width-1):
			total += output[row][col]
	print(str(total))


imgLeft = cv2.imread(sys.argv[1])
imgRight = cv2.imread(sys.argv[2])

sumOfAbsoluteDifferences(imgLeft, imgRight)