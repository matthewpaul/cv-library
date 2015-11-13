from PIL import Image
from itertools import izip
import numpy as np 
from scipy import signal as sg
import scipy.stats as st
import pylab as lab
from skimage import data
from skimage import transform as tf

def Affine_Fit( from_pts, to_pts ):
	q = from_pts
	p = to_pts
	if len(q) != len(p) or len(q)<1:
		print "from_pts and to_pts must be of same size."
		return false

	dim = len(q[0]) # num of dimensions
	if len(q) < dim:
		print "Too few points => under-determined system."
		return false

	# Make an empty (dim) x (dim+1) matrix and fill it
	c = [[0.0 for a in range(dim)] for i in range(dim+1)]
	for j in range(dim):
		for k in range(dim+1):
			for i in range(len(q)):
				qt = list(q[i]) + [1]
				c[k][j] += qt[k] * p[i][j]

	# Make an empty (dim+1) x (dim+1) matrix and fill it
	Q = [[0.0 for a in range(dim)] + [0] for i in range(dim+1)]
	for qi in q:
		qt = list(qi) + [1]
		for i in range(dim+1):
			for j in range(dim+1):
				Q[i][j] += qt[i] * qt[j]

	# Ultra simple linear system solver. Replace this if you need speed.
	def gauss_jordan(m, eps = 1.0/(10**10)):
	  """Puts given matrix (2D array) into the Reduced Row Echelon Form.
		 Returns True if successful, False if 'm' is singular.
		 NOTE: make sure all the matrix items support fractions! Int matrix will NOT work!
		 Written by Jarno Elonen in April 2005, released into Public Domain"""
	  (h, w) = (len(m), len(m[0]))
	  for y in range(0,h):
		maxrow = y
		for y2 in range(y+1, h):    # Find max pivot
		  if abs(m[y2][y]) > abs(m[maxrow][y]):
			maxrow = y2
		(m[y], m[maxrow]) = (m[maxrow], m[y])
		if abs(m[y][y]) <= eps:     # Singular?
		  return False
		for y2 in range(y+1, h):    # Eliminate column y
		  c = m[y2][y] / m[y][y]
		  for x in range(y, w):
			m[y2][x] -= m[y][x] * c
	  for y in range(h-1, 0-1, -1): # Backsubstitute
		c  = m[y][y]
		for y2 in range(0,y):
		  for x in range(w-1, y-1, -1):
			m[y2][x] -=  m[y][x] * m[y2][y] / c
		m[y][y] /= c
		for x in range(h, w):       # Normalize row y
		  m[y][x] /= c
	  return True

	# Augement Q with c and solve Q * a' = c by Gauss-Jordan
	M = [ Q[i] + c[i] for i in range(dim+1)]
	if not gauss_jordan(M):
		print "Error: singular matrix. Points are probably coplanar."
		return false

	# Make a result object
	class Transformation:
		"""Result object that represents the transformation
		   from affine fitter."""

		def To_Str(self):
			res = ""
			for j in range(dim):
				str = "x%d' = " % j
				for i in range(dim):
					str +="x%d * %f + " % (i, M[i][j+dim+1])
				str += "%f" % M[dim][j+dim+1]
				res += str + "\n"
			return res

		def Transform(self, pt):
			res = [0.0 for a in range(dim)]
			for j in range(dim):
				for i in range(dim):
					res[j] += pt[i] * M[i][j+dim+1]
				res[j] += M[dim][j+dim+1]
			return res
	return Transformation()

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
trn = Affine_Fit(correspondance[0], correspondance[1])
print trn.To_Str()