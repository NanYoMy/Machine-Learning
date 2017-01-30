import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pdb
from random import randint

class node(object):
	def __init__(self,value,observed):#TODO: add label when repurposed for seg
		self.value = value
		# self.label = label
		self.observed = observed
		self.neighbors = []
	def add_neighbor(self,new_node):
		self.neighbors.append(new_node)
	def isObserved(self):
		return self.observed
	def node_energy(self):
		sum=singleton_energy(self)
		for n in self.neighbors:
			sum+=pair_energy(self,n)
		return sum

def singleton_energy(unobserved_node): #biases denoised image towards pixeLs with value of opposite sign to h
	h = 0
	if unobserved_node.isObserved:
		return 0 #raise NameError('Node comes from noisy image')
	return h*unobserved_node.value

def pair_energy(node1,node2):
	eta = 2.1
	beta = 1.0
	if( node1.isObserved() and node2.isObserved()):
		# pdb.set_trace()
		raise NameError('Nodes both from noisy image')
	elif not ( node1.isObserved() or node2.isObserved() ):
		# print("beta")
		return -beta*node1.value*node2.value
	# print("eta")
	return -eta*node1.value*node2.value


class MRF(object):
	nodes = []


	def loadIm(self,im):
		if len(im.shape) >2:
			raise NameError("Dimensional error: Is image 2d gray?") #for denoising should be binary black/white
		#load in noisy image and add corresponding latent nodes and the edges between.

		image_nodes = [[None]*im.shape[1] for i in range(0,im.shape[0])]

		for i in range(0,im.shape[0]):
			for j in range(im.shape[1]):
				y = node(value=im[i,j],observed=True)#in true graph these should be connected to x, but this is already encoded in x's neighbors.
				image_nodes[i][j] = y
				self.add_node(y)

		latent_nodes = 	[[None]*im.shape[1] for i in range(0,im.shape[0])]
		for i in range(0,im.shape[0]):
			for j in range(0,im.shape[1]):
				x = node(value=im[i,j],observed=False)
				latent_nodes[i][j] = x

		for i in range(0,im.shape[0]):
			for j in range(0,im.shape[1]):

				latent_nodes[i][j].add_neighbor(image_nodes[i][j])

				#boundary cases!!! :)))))))))))
				# pdb.set_trace()
				if i == 0 and j==0:
					neigh_idx = np.array([[i+1,j],[i,j+1]]) # top left corner
				elif i==0 and j==im.shape[1]-1:# top right corner
					neigh_idx = np.array([[i,j-1],[i+1,j]])
				elif i==im.shape[0]-1 and j==im.shape[1]-1: # bottom right corner
					neigh_idx = np.array([[i-1,j],[i,j-1]])
				elif i==im.shape[0]-1 and j==0:#bottom left corner
					neigh_idx = np.array([[i-1,j],[i,j+1]])
				elif i==0:
					neigh_idx = np.array([[i+1,j],[i,j+1],[i,j-1]]) # top edge
				elif j==im.shape[1]-1:# right edge
					neigh_idx = np.array([[i,j-1],[i-1,j],[i+1,j]])
				elif i==im.shape[0]-1:#bottom edge
					neigh_idx = np.array([[i-1,j],[i,j-1],[i,j+1]])
				elif j==0:
					neigh_idx = np.array([[i+1,j],[i,j+1],[i-1,j]]) # left edge
				else:
					neigh_idx = np.array([[i+1,j],[i,j+1],[i-1,j],[i,j-1]])
				for idx in neigh_idx:
					latent_nodes[i][j].add_neighbor(latent_nodes[idx[0]][idx[1]])
				self.add_node(latent_nodes[i][j])

	def add_node(self,n):
		self.nodes.append(n)
	# def compute_energy(self):
	# 	total=0
	# 	checked_pair = []
	# 	for n in self.nodes:
	# 		total += singleton_energy(n)
	# 		for neighbor in n.neighbors:
	# 			if [n,neighbor] not in checked_pair and [neighbor,n] not in checked_pair:#TODO: check this line for weird bugs
	# 				total += pair_energy(n,neighbor)
	# 				checked_pair.append([n,neighbor])
	def print_nodes(self):#TODO: for debugging, remove later
		for n in self.nodes:
			print(str(n.value) + " " + str(n.observed))
	def minimize_energy(self):
		num_change = 1# TODO: hacky, rewrite
		while num_change >0:
			num_change = 0
			for n in self.nodes:
				current_energy = n.node_energy()
				n.value = -n.value
				hyp_energy = n.node_energy()
				if hyp_energy < current_energy:
					num_change+=1
					# print(num_change)
				else:
					n.value = -n.value
	def denoise(self, im):
		self.loadIm(im)
		self.minimize_energy()

		barrier = im.shape[0]*im.shape[1]-1
		rowlen = im.shape[1]
		for i in range(0,im.shape[0]):
			for j in range(0,im.shape[1]):
				# print(str(i)+","+str(j))
				if self.nodes[i*rowlen+j].value != self.nodes[barrier+i*rowlen+j].value: #see if change made in final minimized graph. Weird indexing is because all stored in one array.
					im[i][j] = -im[i][j]
		return im




# n = node(value=1,observed=True)
# m = node(value=-1,observed=False)
# s = node(value=-1,observed=False)
# n.add_neighbor(m)
# m.add_neighbor(n)
# m.add_neighbor(s)
# print(m.node_energy())
# m.value=-m.value
# print(m.node_energy())
#
# #
# nlist =[i.value for i in n.neighbors]
# mlist = [i.value for i in m.neighbors]
# print(nlist)
# print(mlist)
# pdb.set_trace()

####### Create MRF from image ########

if __name__ == "__main__":


	## Demo: Using ynthetic image, displays original (Fig 1), noised (Fig2), and denoised (Fig 3). 
	im = np.ones([200,200])
	im[100:150,100:150] = -1
	plt.figure()
	plt.imshow(im)
	for i in range(0,1000):
		x = randint(0,199)
		y = randint(0,199)
		im[y][x] = -im[y][x]
	plt.figure()
	plt.imshow(im)
	imageField = MRF()
	imageField.loadIm(im)
	denoised_im=imageField.denoise(im)
	plt.figure()
	plt.imshow(denoised_im)
	plt.show()






# plt.imshow(im)
# plt.show()
