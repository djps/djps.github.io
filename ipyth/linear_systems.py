import numpy as np
import matplotlib.pyplot as plt
import copy


# # Solver linear equations:
#
# Tridiagonal solver (Thomas algorithm); input has to be a tridiagonal square matrix and a solution vector. If the matrix is singular (i.e. determinant = 0), the algorithm will break (division by zero).

# The algorithm is not generally stable (i.e. does not magnify small errors in the original matrix via, for example, rounding errors), but it is stable in special cases such as diagonally dominant matrices or symmetric positive definite matrices. In practice, one of the two is often the case.

# Function for solving tridiagonal matrix problem
def tridiag(B, z):
    s=np.shape(B)
    x=np.zeros(len(z))
    A=copy.deepcopy(B)
    y=copy.deepcopy(z)
    if s[0]==s[1]:
        n=s[0]-1
        for i in range(1, n+1):
            #print i
            w=A[i,i-1]/A[i-1,i-1]
            A[i,i]=A[i,i]-w*A[i-1,i]
            y[i]=y[i]-w*y[i-1]
        x[n]=y[n]/A[n,n]
        for i in range(n,0,-1):
            j=i-1
            x[j]=(y[j]-A[j,j+1]*x[j+1])/A[j,j]
    else:
        print('Not a square matrix')
    return x


# We first check if potential conditions for termination of the algorithm hold. This step might be as costly as or even more costly than the algorithm itself, which is why one wouldn't do this in general. In many practical cases, there might be indications prior to using the algorithm that the conditions hold. Sometimes, one just simply applies the algorithm without checking at all and just checks if it fails. Below, we first apply the checks and then apply the algorithm.

#Matrix A and right-hand side y of Ax=y
A=np.array([[2.,1., 0.],[1.,3., 1.], [0.,1.,2.]])
y=np.array([1.,2.,5.])

#Check for positive definiteness and symmetry
print('Matrix is pos. def. ', np.all(np.linalg.eigvals(A) > 0))
At=np.matrix.transpose(A)
print('Matrix is symmetric ',np.all(A==At))

#Check for diagonal dominance
diag = np.diag(np.abs(A)) # Absolute value of diagonal coefficients
rowsum = np.sum(np.abs(A), axis=1) - diag # Row sum without diagonal element
print('Matrix is diagonally dominant ',np.all(diag>rowsum))


x=tridiag(A,y)
print('Solution vector of the problem is ', x)


# # Cholesky decomposition:
#
# Cholesky decomposition of an input matrix A, which must be positive definite and symmetric. It returns the lower triangular matrix L. The upper one is given by the transpose of L.

from math import sqrt
from pprint import pprint

def cholesky(A):
    """Performs a Cholesky decomposition of A, which must
    be a symmetric and positive definite matrix. The function
    returns the lower variant triangular matrix, L."""
    n = len(A)

    # Create zero matrix for L
    L = np.zeros([n,n])

    # Perform the Cholesky decomposition
    for i in range(0,n):
        for k in range(0,i+1):
            tmp = sum(L[i,:] * L[k,:])

            if (i == k): # Diagonal elements
                L[i,k] = sqrt(A[i,i] - tmp)
            else:
                L[i,k] = (1.0 / L[k,k] * (A[i,k] - tmp))
    return L


# We apply the Cholsky decomposition to the same matrix A as before for the Thomas algorithm. For solving an actual system of Ax=y one now only needs to do substitution with the matrix L and its transpose to find x.

L=cholesky(A)
print('Lower triangular matrix of Cholesky decomposition ')
print(L)
K=L.dot( L.transpose())
print('Reconstruction of original matrix by LL^t')
print(K)
