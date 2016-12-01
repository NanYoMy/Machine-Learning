import numpy as np
import math, random

from skimage import io, img_as_float, color
from skimage.exposure import equalize_hist

class Image():
	"""
	holds a picture
	can either supply a filename or
	data as a numpy array
	"""
	def __init__(self, filename=None, data=None):
		assert any([filename is not None, data is not None]), "you need to supply an image file or pass a picture array"

		if filename is not None:
			self._data = io.imread(filename)
		else:
			self._data = data

		# preprocessing
		if self._data.ndim > 2:
			self._data = equalize_hist(color.rgb2gray(self._data)) # convert to grayscale
		self._data = img_as_float(self._data)

		(self.height, self.width) = self._data.shape

	def __getitem__(self, item):
	# piggyback off of numpy's array indexing
		return self._data.__getitem__(item)

		
class MRF():
	def __init__(self, image, means, variances):
		# mean, var comes from gmm
		self.image = image
		self.means, self.variances = means, variances
		self.no_classes = len(means)

		self.labels = self.init_labels_from_gmm()

	### initialization helper functions

	def beta(self, i, j, coeff=-1):
		# possible to overwrite this function to modify our energies
		return coeff
	
	def log_like(self, intensity, mu, sig):
		# since we will be comparing likelihoods can ignore the term that includes pi in it
		# sig is the variance (so in math-land it is actually sigma^2)
		return -0.5 * math.log(sig) - (intensity - mu)**2 / (2 * sig)

	def estimate_label(self, i, j):
		# maximum log-likelihood
		lls = [self.log_like(self.image[i,j],self.means[k],self.variances[k]) 
			   for k in range(self.no_classes)]
		return np.argmax(lls) # returns index of maximum as apposed to python's max which would return maximum value

	def init_labels_from_gmm(self):
		label_output = np.empty_like(self.image._data, dtype='int8')
		for i in range(self.image.height):
			for j in range(self.image.width):
				label_output[i,j] = self.estimate_label(i,j)
		return label_output

	### energy calculation functions

	def singleton(self, i, j, label):
		# this is just the negative log-likelihood
		return math.log(math.sqrt(2.0 * math.pi * self.variances[label])) + \
			(self.image[i,j] - self.means[label])**2 / (2.0 * self.variances[label])
	
	# change to multivariate gaussian

	def doubleton(self, i, j, label):
		energy = 0.0

		for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
			if all([i + dy < self.image.height,
					i + dy >= 0,
					j + dx < self.image.width,
					j + dx >= 0]):
				if label == self.labels[i+dy,j+dx]: 
					energy -= self.beta(i,j)
				else:
					energy += self.beta(i,j)

		return energy

	def global_energy(self):
		singletons = 0
		doubletons = 0

		for i in range(self.image.height):
			for j in range(self.image.width):
				k = self.labels[i,j]
				singletons += self.singleton(i,j,k)
				doubletons += self.doubleton(i,j,k)

		return singletons + doubletons/2

	def local_energy(self, i, j, label):
		return self.singleton(i,j,label) + self.doubleton(i,j,label)

	### estimation algos for updating labels
	# efficient graph cut

	def icm(self, thresh=0.05):
		# basically loop through everything picking minimizing labeling until "convergence"
		E_old = self.global_energy()
		delta_E = 999 # no do-while in python :v[
		counter = 0
		# generate all indices of the image
		indices = [(i,j) for i in range(self.image.height) for j in range(self.image.width)]

		while delta_E > thresh and counter < 10: # threshold for convergence
			delta_E = 0
			# mix up the order the indices are visited
			random.shuffle(indices)

			for i, j in indices:
				local_energies = [self.local_energy(i,j,k) for k in range(self.no_classes)]
				self.labels[i,j] = np.argmin(local_energies)

			energy = self.global_energy()
			delta_E = math.fabs(E_old - energy)
			E_old = energy
			counter += 1

		print('took {} iterations\n final energy was {:6.6f}'.format(counter,E_old))

	def gibbs(self, thresh=0.05, temp=4):

		E_old = self.global_energy()
		delta_E = 999 # no do-while in python :v[
		counter = 0

		# generate all indices of the image
		indices = [(i,j) for i in range(self.image.height) for j in range(self.image.width)]

		while delta_E > thresh and counter < 10: # threshold for convergence
			random.shuffle(indices)
			for i, j in indices:
				local_energies = [math.exp(-1*self.local_energy(i,j,k)/temp) for k in range(self.no_classes)]
				sum_energies = sum(local_energies)
				r = random.uniform(0,1)
				z = 0
				for k in range(self.no_classes):
					z += local_energies[k] / sum_energies
					if z > r:
						self.labels[i,j] = k
						break
			energy = self.global_energy()
			delta_E = math.fabs(E_old - energy)
			E_old = energy
			counter += 1

		print('took {} iterations\n final energy was {:6.6f}'.format(counter,E_old))

# gmm with estimate_parameters function
# -> mle to get initial segmentation

if __name__ == '__main__':
	import matplotlib.pyplot as plt

	# test_img = Image() # should raise error
	test_img = Image('./test_resized.jpg')
	test_mrf = MRF(test_img,[.3,.5,.7],[.25, .66, .25])
	plt.imshow(test_mrf.labels,cmap='gist_gray_r')
	plt.savefig('before.png')
	test_mrf.icm()
	plt.imshow(test_mrf.labels)
	plt.savefig('after_icm.png',cmap='gist_gray_r')
	# lol, for my 640 x 425 px image this code took 140 seconds to run on my desktop