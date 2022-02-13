import numpy as np

class FCNN:
	def __init__(self):
		self.layers = []
		self.weights = []

		# Build input layer
		n1 = int(input("Enter number of input nodes: "))
		self.layers.append(np.zeros(n1))

		# Start adding hidden layers
		k = int(input("Enter number of hidden layers: "))

		i = 1
		while i <= k:
			# Build hidden layer i
			ni = int(input(f"Enter number of nodes in hidden layer {i}: "))
			self.layers.append(np.zeros(ni))
			i += 1

		# Build output layer
		nf = int(input("Enter number of nodes in output layer: "))
		self.layers.append(np.zeros(nf))

		self.size = len(self.layers)

		# Build weights
		i = 0
		while i < self.size-1:
			wr = self.layers[i+1].shape[0]
			wc = self.layers[i].shape[0]
			self.weights.append(np.random.rand(wr, wc))
			i += 1


	def forProp(self, inp):
		print(inp)
		self.layers[0] = inp
		i = 1
		while i < self.size:
			self.layers[i] = np.dot(self.weights[i-1], self.layers[i-1])
			print(self.layers[i])
			i += 1
		return self.layers[-1]


	def backProp(self, inp):
		pass


	def trainOnExample(self, ex):
		# ex: (exampleInput, expectedOutput)
		pass



	def __repr__(self):
		return str(self.layers)
