from PIL import Image
import numpy as np 
from scipy import signal as sg


# The following functions allow for converting image files into
# arrays of floats. This allows us to open the image for editing
# and saving the image after we're done. 
def np_from_image(pic):
	return np.asarray(Image.open(pic), dtype=np.float32)

def save_as_image(ar, pic):
	Image.fromarray(ar.round().astype(np.uint8)).save(pic)

def norm(ar):
	return 255.*np.absolute(ar)/np.max(ar)

# Use the scipy convolve function to convolve the kernel around the image.
# This sets up a basis for testing our convolve function
img = np_from_image('img/portal.png')
newimg = save_as_image(norm(sg.convolve(img, [[1.], [-1.]])), 'img/portal-v.png')

