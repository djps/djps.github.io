import numpy as np
from typing import Union

# Setup a class for the Jacobi algorithm

class Jacobi:
	def __init__(self, *args, **kwargs) -> None:
		self.x = None
		self.name = "Jacobi Iteration"

	def iteration(self, x, b, L, U, D):
		self.x = np.matmul(np.linalg.inv(D), (b - np.matmul((L + U), x)))
		return self.x


# Define the Gauss-Seidel iteration

class Gauss_Seidel:
	def __init__(self, *args, **kwargs) -> None:
		self.x = None
		self.name = "Gauss-Seidel Iteration"
		pass

	def iteration(self, x, b, L, U, D):
		self.x = np.matmul(np.linalg.inv(D + L), (b - np.matmul(U, x)))
		return self.x


# Define the SOR iteration

class Successive_Overrelaxation:
	def __init__(self, *args, **kwargs) -> None:
		self.x = None
		self.name = "Successive Overrelaxation"
		self.w = kwargs["w"]
		print("w: ", self.w)

	def iteration(self, x, b, L, U, D):
		first_part = np.linalg.inv(((1.0 / self.w) * D) + L)
		second_part = b - np.matmul((U + (1 - (1 / self.w)) * D), x)
		self.x = np.matmul(first_part, second_part)
		return self.x


# Interface through which the algorithms will be applied:

class Interface:
	def __init__(self,
				A: np.matrix,
				 x: np.matrix,
				 b: np.matrix,
				 algo: Union[Jacobi, Gauss_Seidel, Successive_Overrelaxation],
				 w: float = None) -> None:

		self.A = A
		self.x = x
		self.b = b
		self.D = np.diag(np.diag(A))	# Automating extraction of diagonal
		self.L = np.tril(A) - self.D	# Extract lower triangular part
		self.U = np.triu(A) - self.D	# Extract upper triangular part
		self.step = 0					# Keep track of step number
		self.w = w						# Used for SOR
		self.algo = algo(w = w)			# Switch between algorithms here

	def iteration(self, step_count = True):				# Apply chosen algorithm
		self.x = self.algo.iteration(x = self.x, b = self.b, L = self.L, U = self.U, D = self.D)
		if (step_count == True):
			self.step += 1
		return self.x

	def iterate_n_times(self, steps = 5):
		i = 0
		while (i < steps):
			self.step += 1
			self.print_step()
			self.iteration(step_count=False)
			self.print_x()
			i += 1

	def print_step(self):
		print(f"{self.algo.name}::Step {self.step}:")

	def print_values(self):
		self.print_step()
		print("A: \n", self.A)
		print()
		print("x: \n", self.x)
		print()
		print("b: \n", self.b)
		print()
		print("D: \n", self.D)
		print()
		print("D inverse: \n", np.linalg.inv(self.D))
		print()
		print("L: \n", self.L)
		print()
		print("U: \n", self.U)
		print()

	def print_x(self):
		print("x: \n", self.x)
		print()


# Compute Jacobi scheme

A = np.matrix([[5, -2, 1],
		       [-3, 9, 1],
			   [2, -1, -7]])
b = np.transpose(np.matrix([-1, 2, 3]))
x0 = np.transpose(np.matrix([0, 0, 0]))

num_steps = 10
jacobi = Interface(A, x0, b, Jacobi)
jacobi.print_values()
jacobi.iterate_n_times(num_steps)

# GS scheme
gsi = Interface(A, x0, b, Gauss_Seidel)
gsi.print_values()
gsi.iterate_n_times(num_steps)


print("Performing some steps of Successive Overrelaxation...")
sor = Interface(A, x0, b, w = 1.1, algo = Successive_Overrelaxation)
sor.print_values()
sor.iterate_n_times(num_steps)
