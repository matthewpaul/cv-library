from PIL import Image
from itertools import izip
import numpy as np 
from scipy import signal as sg
import scipy.stats as st
import pylab as lab

def gaussian(kernlen=21, nsig=3):
	interval = (2*nsig+1.)/(kernlen)
	x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
	kern1d = np.diff(st.norm.cdf(x))
	kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
	kernel = kernel_raw/kernel_raw.sum()
	return kernel


# The following functions allow for converting image files into
# arrays of floats. This allows us to open the image for editing
# and saving the image after we're done. 
def np_from_image(pic):
	return np.asarray(Image.open(pic), dtype=np.float32)

def save_as_image(ar, pic):
	Image.fromarray(ar.round().astype(np.uint8)).save(pic)

def norm(ar):
	return 255.*np.absolute(ar)/np.max(ar)

########################################### PART 1 Function Definitions ####################################################
def convolve(image, kernel):
	imgArray = np_from_image(image)
	return norm(sg.convolve(imgArray, kernel))

# Performs a gaussian convolution, and then reduces image size by half the width and height.
def reduce(image):
	width, height = Image.open(image).size
	img = Image.fromarray(convolve(image, gaussian()).round().astype(np.uint8))
	return img.resize((width / 2, height / 2))

# Returns an image that is 2 times the height and width of the original image. 
def expand(image):
	width, height = Image.open(image).size
	return Image.open(image).resize((width * 2, height * 2))

# Returns an image reduced by a Gaussian Pyramid given an image i and number of levels n using the reduce(image) function.
def gaussianPyramid(image, n, filename):
	tempImage = image
	for x in range(0, n):
		reduce(tempImage).save(filename)
		tempImage = filename
	return Image.open(filename)

# Returns an image expanded by a Laplacian Pyramid given an image I and a number of levels n using the expand(image) function.
def laplacianPyramid(image, n, filename):
	tempImage = image
	for x in range(0, n):
		expand(tempImage).save(filename)
		tempImage = filename
	return Image.open(filename)

# Returns the reconstructed image from the image returned by the Laplacian Pyramid function.
def Reconstruct(L1, n):
	originalImage = Image.open(L1)
	laplacianImage = laplacianPyramid(L1, n, 'img/reconLaplacian.png')
	reconstructedImage = gaussianPyramid('img/reconLaplacian.png', n, 'img/reconImage.png')
 
	pairs = izip(originalImage.getdata(), reconstructedImage.getdata())
	if len(originalImage.getbands()) == 1:
		# for gray-scale jpegs
		dif = sum(abs(p1-p2) for p1,p2 in pairs)
	else:
		dif = sum(abs(c1-c2) for p1,p2 in pairs for c1,c2 in zip(p1,p2))
 
	ncomponents = originalImage.size[0] * originalImage.size[1] * 3
	print ("Difference (percentage):", (dif / 255.0 * 100) / ncomponents)
	return reconstructedImage

def imageGinput():
	x1 = Image.open('img/portal.png')
	fig1 = lab.figure(1)
	ax1 = fig1.add_subplot(111)
	ax1.imshow(x1)
	ax1.axis('image')
	ax1.axis('off')
	x = fig1.ginput(2)

	fig1.show()
	print(x)

#############################################################################################################################


# Use the scipy convolve function to convolve the kernel around the image.
# This sets up a basis for testing our convolve function
singleDGaussian = [0.000229, 0.005977, 0.060598, 0.241732, 0.382928, 0.241732, 0.060598, 0.005977, 0.000229]

#save_as_image(convolve('img/portal.png', gaussian()), 'img/gportal.png')
#reduce('img/portal.png').save('img/rportal.png')
#expand('img/portal.png').save('img/bigportal.png')
#gaussianPyramid('img/portal.png', 2, 'img/gpportal.png')
#laplacianPyramid('img/portal.png', 2, 'img/laplacianportal.png')
#Reconstruct('img/portal.png', 3).save('img/reconImage.png')
imageGinput()

