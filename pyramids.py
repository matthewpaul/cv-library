from PIL import Image
import numpy as np 
from scipy import signal as sg
import scipy.stats as st

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

#############################################################################################################################


# Use the scipy convolve function to convolve the kernel around the image.
# This sets up a basis for testing our convolve function
singleDGaussian = [0.000229, 0.005977, 0.060598, 0.241732, 0.382928, 0.241732, 0.060598, 0.005977, 0.000229]

save_as_image(convolve('img/portal.png', gaussian()), 'img/gportal.png')
reduce('img/portal.png').save('img/rportal.png')
expand('img/portal.png').save('img/bigportal.png')

