import operator
from functools import partial



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
	def eval(*args):
		return prob(args)

def m_eval(func,depnode,val):
	varname = "x" + string(depnode)
	return partial(func, varname=val)

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

#My sad little hard-coded-but-functional code
def DAG_Eliminate():
	def m4(x3):
		return cond43(1,x3)
	def m3(x2):
		return cond32(0,x2)*m4(0)+cond32(1,x2)*m4(1)
	def m2(x1):
		return marg_x1(x1)*cond21(0,x1)*m3(0)+marg_x1(x1)*cond21(1,x1)*m3(1)
	m1 = marg_x1(0)*m2(0)+marg_x1(1)*m2(1)
	res = [m1,m2,m3,m4]
	return res
result = DAG_Eliminate()
#p(x1=0,x4=1) for x4=1 is result[1](0)
#p(x4) for x4=1 is result[0]
#so p(x1=0 | x4 =1) is result[1](0)/result[0] 



# def divide(n,m):
# 	return n/m
# def half():
# 	return partial(divide,m=2)
# print(partial(divide,m=2)(10))
# def quarter():
# 	return partial(divide,m=4)
# a = [half,quarter]
# def eighth():
# 	return functools.reduce(operator.mul,a)
# eighth()

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




