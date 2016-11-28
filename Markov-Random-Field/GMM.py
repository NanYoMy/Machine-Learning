import numpy as np
from random import randint
from scipy import stats
from scipy import misc
import math
import pdb
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
class GaussianMixtureModel(object):
	def __init__(self, data, K):#, pi_init, mu_init, sigma_init):
		self.data = data
		self.K = K
		# self.pi = pi_init
		# self.mu = mu_init
		# self.sigma = sigma_init

	# def init_labels_params(self):
	# self.labels = np.zeros(self.data.shape)
	# pdb.set_trace()
	# for i in range(self.labels.size):
	# 	self.labels[i] = randint(0, self.K - 1);
	# self.pi = np.tile(0.0, self.K)
	# self.mu = [np.array([0.0, 0, 0]) for i in range(self.K)]  # row vectors...
	# self.sigma = [np.matrix([[1, 0.0, 0], [0, 1.0, 0], [0, 0.0, 1]]) for i in range(K)]

	def E_step(self, pi_old, mu_old, sigma_old):
		# pdb.set_trace()
		normals = [stats.multivariate_normal(mean=mu_old[i], cov=sigma_old[i]) for i in range(self.K)]
		gamma_z = np.zeros([self.K, self.data.shape[0], self.data.shape[1]])
		for i in range(self.data.shape[0]):
			for j in range(self.data.shape[1]):
				gamma_denom = 0.0
				for k in range(self.K):
					# pdb.set_trace()
					gamma_denom += pi_old[k] * normals[k].pdf(x=self.data[i, j, 0:3])  # ignore alpha channel
				for l in range(self.K):
					gamma_z[l, i, j] = pi_old[l] * normals[l].pdf(x=self.data[i, j, 0:3]) / gamma_denom
		return gamma_z

	def M_step(self, gamma_z):
		# estimate pi
		N = np.zeros(self.K)
		for i in range(self.data.shape[0]):  # TODO:vectorize
			for j in range(self.data.shape[1]):
				for k in range(self.K):
					N[k] += gamma_z[k, i, j]
		# print(gamma_z)
		pi_new = N / (self.data.shape[0]*self.data.shape[1])

		# estimate mu
		mu_new = [np.array([0.0, 0, 0]) for i in range(self.K)]
		for i in range(self.data.shape[0]):  # TODO:vectorize
			for j in range(self.data.shape[1]):
				for k in range(self.K):
					# pdb.set_trace()
					mu_new[k] += gamma_z[k, i, j] * self.data[i, j, 0:3] / N[k]

		sigma_new = [np.matrix([[0.0, 0, 0], [0.0, 0, 0], [0.0, 0, 0]]) for i in range(self.K)]
		for i in range(self.data.shape[0]):  # TODO:vectorize
			for j in range(self.data.shape[1]):
				diffs = [np.matrix(self.data[i, j, 0:3] - mu_new[k]) for k in range(self.K)]
				for k in range(self.K):
					sigma_new[k] += gamma_z[k, i, j] * (np.matrix.transpose(diffs[k]).dot(diffs[k])) / N[k]
		return [pi_new, mu_new, sigma_new]

	# def likelihood(self,gamma_z,pi,mu,sigma):
	# 	total = 0;
	# 	for i in range(self.data.shape[0]):#TODO:vectorize
	# 		for j in range(self.data.shape[1]):
	# 			diffs = [np.matrix(self.data[i, j, 0:3] - mu[k]) for k in range(self.K)]
	# 			log_arg = 0
	# 			for k in range(self.K):
	# 				log_arg+= pi[k](math.log(pi[k]) - math.log

	def estimate_parameters(self, max_iter, pi_init,mu_init, sigma_init):
		i = 0
		pi_est = pi_init
		mu_est = mu_init
		sigma_est = sigma_init
		while i < max_iter:
			gamma_z = self.E_step(pi_est, mu_est, sigma_est)
			# pdb.set_trace()
			[pi_est, mu_est, sigma_est] = self.M_step(gamma_z)
			i += 1
		return [pi_est, mu_est, sigma_est]


#### Main Code ####
NN = 30
K = 2
# m = [0,0]
# sigma = [0,0]
# m[0] = (np.array([4,4]))
# m[1] = (np.array([0,0]))
# sigma[0] = np.matrix([[1,0],[0,1]])
# sigma[1] = np.matrix([[0.5,0.25],[0.25,1.5]])

m = np.array([0.0, 0.0, 1.0])
mu = [m for i in range(K)]  # row vectors...
sigma = [np.matrix([[2.0, 10.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]]), np.matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]) ]
pi = [0.75, 0.25]
z = np.random.choice(size=[100, 100], a=[0, 1], p=pi)
im = np.zeros([NN, NN, 3])
for ii in range(NN):
	for jj in range(NN):
		# plt.hold(true)
		im[ii, jj, 0:3] = np.random.multivariate_normal(mean=mu[z[ii, jj]], cov=sigma[z[ii, jj]])

init_pi = [0.5, 0.5]
init_mu = [np.array([10.0, 10.0, 1.0]), np.array([0.0, 0.0, 0.0])]
init_sigma = [np.matrix([[1000.0, 0.0, 0.0], [0.0, 157.0, 0.0], [0.0, 0.0, 1.0]]) for i in range(K)]

# GMM.init_labels_params()
# im = misc.imresize(mpimg.imread("/Users/Oliver/Desktop/watershed.png"),5)/255
GMM = GaussianMixtureModel(im, K)
ests = GMM.estimate_parameters(10, init_pi, init_mu, init_sigma)
# print(ests)
# plt.imshow(im)
# plt.show()
