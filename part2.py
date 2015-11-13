from PIL import Image
from itertools import izip
import numpy as np 
from scipy import signal as sg
import scipy.stats as st
import pylab as lab

def recover_homogenous_affine_transformation(p, p_prime):
	# construct intermediate matrix
	Q = p[1:] - p[0]
	Q_prime = p_prime[1:] - p_prime[0]

	# calculate rotation matrix
	R = np.dot(np.linalg.inv(np.row_stack((Q, np.cross(*Q)))),
			   np.row_stack((Q_prime, np.cross(*Q_prime))))

	# calculate translation vector
	t = p_prime[0] - np.dot(p[0], R)

	# calculate affine transformation matrix
	return np.column_stack((np.row_stack((R, t)),
							(0, 0, 0, 1)))


def imageGinput(image1, image2):
	x1 = Image.open(image1)
	x2 = Image.open(image2)
	fig2 = lab.figure(1)
	fig1 = lab.figure(1)
	ax2 = fig2.add_subplot(141)
	ax1 = fig1.add_subplot(142)
	ax1.imshow(x1)
	ax2.imshow(x2)
	x = fig1.ginput(3)
	x22 = fig2.ginput(3)
	fig1.show()
	fig2.show()
	newArray = [np.asarray(x), np.asarray(x22)]
	print newArray
	return newArray


correspondance = imageGinput('img/portal.png', 'img/portal.png')
print recover_homogenous_affine_transformation(correspondance[0], correspondance[1])