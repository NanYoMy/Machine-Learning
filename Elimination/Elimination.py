import operator
import numpy as np



#probability definitions, and their corresponding classes indexed by dependency
def marg_x1(x):
	if x!=0 and x!=1:
		raise NameError("Input must be 0,1")
		return
	else:
		return 0.5
def cond21(x2,x1):
	prob_table = [[0.6,0.2],[0.4,0.8]]
	return prob_table[x2][x1]
def cond32(x3,x2):
	prob_table = [[0.6,0.2],[0.4,0.8]]
	return prob_table[x3][x2]
def cond43(x4,x3):
	prob_table = [[0.6,0.2],[0.4,0.8]]
	return prob_table[x4][x3]
def ev_potential(arg, index):
	if arg==index:
		return 1
	else:
		return 0
marg1 = indexed_function([1],marg_x1)
cp2_1 = indexed_function([2,1],cond21)
cp3_2 = indexed_function([3,2],cond32)
cp4_3 = indexed_function([4,3],cond43)
delta4 = indexed_function([4],ev_potential)

class indexed_function: #simple class containing function and its node dependencies
	deplist = []
	def __init__(self, deplist, prob):
		self.deplist = deplist
		self.prob = prob
	def marginalize(self,varindex):
		self.prob = 

	def eval(self,*args):
		return self.prob[tuple(args)]














def DAG_Eliminate():
	order  = [4,3,2,1]
	m = []
	activelist = [marg1, cp2_1,cp3_2,cp4_3, delta4]
	for node in order:
		productlist = []
		for func in activelist:
			if node in func.deplist:
				productlist.append(func)
				activelist.remove(func)
			prod0list = []
			prod1list = []
			for func in funclist:
				prod0list[funclist.index(func)] = m_eval(func,node,0)
			for func in funclist:
				prod1list[funclist.index(func)] = m_eval(func,node,1)
			def prod0():
				return reduce(operator.mul,prod0list,1) #doesn't work for lists of functions! Oh, python :'( 
			def prod1():
				return reduce(operator.mul,prod1list,1)

			m.append(prodsum)
			activelist.append(prodsum)






# def DAG_Eliminate():
# 	order = [4,3,2,1]
# 	m=[]
# 	pre_m =[]
# 	activelist = [marg1, cp2_1,cp3_2,cp4_3, delta4]
# 	for node in order:
# 		productlist = []
# 		for func in activelist:
# 			print(func.prob.func_name)
# 			if node in func.deplist:
# 					productlist.append(func)
# 					activelist.remove(func)
# 		pre_m.append(productlist)








# def DAG_Eliminate(*args): #must input node indices in elimination order before functions
# 	nodelist = []
# 	factorlist = []
# 	for arg in args:
# 		if isinstance(arg, int):
# 			nodelist.append(arg)
# 		else:
# 			factorlist.append(arg)
# 	activelist = factorlist
# 	def summand():
# 		....
# 		.
# 		.
# 		.
# 		.
# 		.
# 	for func in activelist:




