import cv2
import cv2.cv as cv
import sys
import numpy as np
from PIL import Image
import pylab as lab

def np_from_image(pic):
	return np.asarray(Image.open(pic), dtype=np.float32)

def imageGinput(anImage):
	x1 = Image.open(anImage)
	fig1 = lab.figure(1)
	ax1 = fig1.add_subplot(111)
	ax1.imshow(x1)
	ax1.axis('image')
	ax1.axis('off')
	x = fig1.ginput(2)

	fig1.show()
	print(x)
	return x

def sumOfAbsoluteDifferences(arr1, arr2):
	output = cv2.absdiff(arr1, arr2)
	total = 0
	shape = output.shape
	height = shape[0]
	width = shape[1]
	for row in range(0, height-1):
		for col in range(0, width-1):
			total += output[row][col]
	return total

# Technically array 2 should be smaller than array 1. A "template" for matching.
def sqDifferences(arr1, arr2):
	result = cv2.matchTemplate(arr1, arr2, cv.CV_TM_SQDIFF_NORMED)
	return result

def ncc(arr1, arr2):
	result = cv2.matchTemplate(arr1, arr2, cv.CV_TM_CCORR_NORMED)
	return result

# When performing matching window, first selected point should
# be top left of rectangle, second point should be bottom right of
# the rectangle. 
def getWindow(image1, image2):
	coordinates = imageGinput(image1)
	topleft = coordinates[0]
	bottomright = coordinates[1]
	startHeight = topleft[0].astype(np.int64)
	startWidth = topleft[1].astype(np.int64)
	endHeight = bottomright[0].astype(np.int64)
	endWidth = bottomright[1].astype(np.int64)
	window = [startHeight, startWidth, endHeight, endWidth]
	print(window)
	return window

def getArrays(image1, image2):
	window = getWindow(image1, image2)
	imgLeft = cv2.imread(image1)
	imgRight = cv2.imread(image2)
	array1 = []
	array2 = []
	for row in range(window[1], window[3]):
		buffer1 = []
		buffer2 = []
		for col in range(window[0], window[2]):
			buffer1.append(imgLeft[row][col])
			buffer2.append(imgRight[row][col])
		array1.append(buffer1)
		array2.append(buffer2)
	result = [np.array(array1), np.array(array2)]
	return result

def performAnalysis(image1, image2):
	arrays = getArrays(image1, image2)
	print("Sum of absolute differences: " + str(sumOfAbsoluteDifferences(arrays[0], arrays[1])))
	print("Sum of squared differences: " + str(sqDifferences(arrays[0], arrays[1])))
	print("Normalized cross correlation: " + str(ncc(arrays[0], arrays[1])))





imgLeft = cv2.imread(sys.argv[1])
imgRight = cv2.imread(sys.argv[2])

#sumOfAbsoluteDifferences(imgLeft, imgRight)
performAnalysis('left.png', 'right.png')