from BlackJackEngine import *
import numpy as np
import random
import time
import pickle
import matplotlib.pyplot as plt
from sklearn import linear_model
np.set_printoptions(threshold= 1000000)
np.set_printoptions(suppress=True) #printset
mut_rate = 0.15 #mutation rate
crossrate = 0.7 #corss rate
score = {}
population = {}


'''
def individual():
    return np.random.uniform(-1,1,10)
def population(n):
    return [individual() for x in range(n)]
'''
def fitness(weights_list, rounds):
	with open(get_file(rounds), 'rb') as handle:
		frequencies, results = pickle.load(handle)
	i = 0
	best_score = 0.0
	for weights in weights_list:
		thresh = 1
		counts = (np.dot( weights, frequencies.T )/6)
		counts[ counts > thresh] *= 5
		counts *= 5 # these counts are now bets
		counts = counts.clip(min = 5, max = 70) # max bet
		score[i] =  np.sum(np.dot(counts, results))*100/(rounds*5)
		population[i] = weights
		print ("weights", weights, "edge", score[i], "%")
		if best_score <= score[i]:
			best_score = score[i]
		i += 1
	return best_score

def fitness_first (weights_list,rounds,best_score):
	with open(get_file(rounds), 'rb') as handle:
		frequencies, results = pickle.load(handle)
	i = 0

	for i in range(0,len(weights_list)):
		thresh = 1
		counts = (np.dot( weights_list[i], frequencies.T )/6)
		counts[ counts > thresh] *= 5
		counts *= 5 # these counts are now bets
		counts = counts.clip(min = 5, max = 50) # max bet
		population[i] = weights_list[i]
		score[i] =  np.sum(np.dot(counts, results))*100/(rounds*5)
		#print ("weights", population[i], "edge", score[i], "%")
		if best_score <= score[i]:
			best_score = score[i]
	return best_score

def fitness_new (rounds,best_score,frequencies, results):

	for i in range(0,len(population)):
		thresh = 1
		counts = (np.dot( population[i], frequencies.T )/6)
		counts[ counts > thresh] *= 5
		counts *= 5 # these counts are now bets
		counts = counts.clip(min = 5, max = 50) # max bet
		score[i] =  np.sum(np.dot(counts, results))*100/(rounds*5)
		#print ("weights", population[i], "edge", score[i], "%")
		if best_score <= score[i]:
			best_score = score[i]
			print("weights", population[i] )
	return best_score


def selection (popul):
	total = 0
	chance = {}
	point = {}
	new_gen = {}
	for i in range(0, popul):
		total += score[i]
	for i in range(0, popul):
		chance[i] =  score[i]/total
		if i == 0:
			point[i] = chance[i]
		else:
			point[i] = point[i-1] + chance[i]
		#print(point[i])
	a = np.random.uniform(0, 1, 10)
	for i in range(0,10): #generate
		for j in range(0, popul):
			if a[i]<=point[j] :
				new_gen[i] = population[j]
				population[popul+i] = new_gen[i]
				break
	#for i in range(0,len(population)):
	#	print(population[i])
	if np.random.uniform(0,1)< crossrate:
		st,nd = np.random.randint(0, len(population), size=2)
		crossover(st,nd)
	mutation()

	#for i in range(0,len(population)):
	#	print(population[i])
	return


def mutation():
	length = 10 # length of individual
	seeds = np.random.uniform(0,1,len(population))
	for i in range(0,len(population)):
		if seeds[i] <= mut_rate:
			j = np.random.randint(0,length)
			population[i][j] = np.random.randint(-1,2)
	#one point mutation
	return


def crossover(st, nd):
	a, b = np.random.randint(0, 10,size=2)
	if (a < b):
		for i in range(a,b):
			temp = population[st][i]
			population[nd][i] = temp
			population[st][i] = population[nd][i] #sweep
	elif(a > b):
		for i in range(b,a):
			temp = population[st][i]
			population[nd][i] = temp
			population[st][i] = population[nd][i] #sweep
	return #double point crossover

def get_file(rounds): #name the file
	if rounds == 100:
		return ('bjdata100.pickle')
	elif rounds == 100000:
		return ('bjdata100k.pickle')
	elif rounds == 1000000:
		return 'bjdata1m.pickle'
	elif rounds == 10000000:
		return ('bjdata10m-2.pickle')


def generate_data( rounds ):
	frequencies, results =  game(shoe_size = 6, rounds = rounds )

	with open(get_file(rounds), 'wb') as handle:
		pickle.dump([frequencies, results], handle)


def generate_random(num):
	a = {}
	for i in range(0, num):
		a[i] = np.random.randint(-1, 2, 10)
	return a


def regression():
	with open('bjdata100k.pickle', 'rb') as handle:
		frequencies, results = pickle.load(handle)

	regr = linear_model.LinearRegression()
	regr.fit( frequencies, results )

	print('Coefficients:')
	print(regr.coef_/np.min(np.absolute(regr.coef_)))
	for x in zip(['A',2,3,4,5,6,7,8,9,10], (regr.coef_/np.min(np.absolute(regr.coef_))).astype(int)):
		print (x)
	'''

	
	'''
def compare_established(rounds):
	print (str(rounds) + " rounds")
	print ("----------------")
	print ("No counting")
	fitness(np.array([0,0,0,0,0,0,0,0,0,0]), rounds)
	print ("")

	print ("Hi Lo")
	fitness(np.array([-1,1,1,1,1,1,0,0,0,-1]), rounds)
	print ("")

	print ("Hi-Opt I")
	fitness(np.array([0,0,1,1,1,1,0,0,0,-1]), rounds)
	print ("")

	print ("Hi-Opt II")
	fitness(np.array([0,1,1,2,2,1,0,0,0,-2]), rounds)
	print ("")

	print ("KO")
	fitness(np.array([-1,1,1,1,1,1,1,0,0,-1]), rounds)
	print ("")

	print ("Omega II")
	fitness(np.array([-1,1,1,2,2,2,1,0,-1,-2]), rounds)
	print ("")

def compare(rounds):
	print (str(rounds) + " rounds")
	fitness([np.array([0,0,0,0,0,0,0,0,0,0]),   # no count
			 np.array([-1,1,1,1,1,1,0,0,0,-1]), # Hi Lo
			 np.array([0,0,1,1,1,1,0,0,0,-1]),  # Hi Opt I
			 np.array([0,1,1,2,2,1,0,0,0,-2]),	# Hi Opt II
			 np.array([-1,1,1,1,1,1,1,0,0,-1]),	# KO
			 np.array([-1,1,1,2,2,2,1,0,-1,-2]),# Omega 
			 ], rounds)

def main():
	REGENERATE = False
	best_score = 0.0
	x_label = []
	y_label = []
	if REGENERATE:
		generate_data(100000)
	with open('bjdata100k.pickle', 'rb') as handle:
		frequencies, results = pickle.load(handle)

		best_score = fitness([np.array([0,0,0,0,0,0,0,0,0,0]),   # no count
			 np.array([-1,1,1,1,1,1,0,0,0,-1]), # Hi Lo
			 np.array([0,0,1,1,1,1,0,0,0,-1]),  # Hi Opt I
			 np.array([0,1,1,2,2,1,0,0,0,-2]),	# Hi Opt II
			 np.array([-1,1,1,1,1,1,1,0,0,-1]),	# KO
			 np.array([-1,1,1,2,2,2,1,0,-1,-2]),# Omega
			 ], 100000)
		print("the best score is:", best_score)
		'''
		original = generate_random(10)
		best_score = fitness_first(original, 100000, best_score)
		'''
		selection(len(population))
		for num in range(0,100): #number of generation:
			best_score = fitness_new(100000,best_score, frequencies, results)
			selection(len(population))
			x_label.append(num)
			y_label.append(best_score)
			plt.scatter(num, best_score,s=75, alpha=.5)
			print("the best score is:", best_score,"now, This is generation: ", num)
		#compare_established(100000)
		#regression()

		plt.plot(x_label, y_label, 'r')
		plt.xlabel('generation')
		plt.ylabel('best score now')
		plt.grid()
		#plt.xlim(0, 100)
		#plt.xticks(())
		#plt.ylim(0, 2)
		#plt.yticks(())
		plt.show()


if __name__ == "__main__":
	main()




#a, b = game(shoe_size = 6, rounds = 10 )
#print a.shape
#print b.shape

