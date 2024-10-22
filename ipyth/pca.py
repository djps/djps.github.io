
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
from sklearn.utils.extmath import svd_flip
from sklearn.decomposition import PCA

'''
Principal component analysis based on:
https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html
'''

plt.style.use('seaborn-v0_8-whitegrid')

def draw_vector(v0, v1, ax=None):
    """Helper function to plot principal component directions"""
    ax = ax or plt.gca()
    arrowprops=dict(arrowstyle='->', linewidth=2, shrinkA=0, shrinkB=0)
    ax.annotate('', v1, v0, arrowprops=arrowprops)

# Create random data
n_samples: int = 200
n_features: int = 2
rng = np.random.RandomState(1)
X = np.dot(rng.rand(n_features, n_features), rng.randn(n_features, n_samples)).T

# plot data
fig0, ax0 = plt.subplots(1, 1)
ax0.scatter(X[:, 0], X[:, 1])
ax0.axis('equal')

# perform PCA extracting all the components, i.e. n_components = n_features
pca = PCA(n_components=2)
pca.fit(X)

fig1, ax1 = plt.subplots(1, 1)
ax1.scatter(X[:, 0], X[:, 1], alpha=0.2)
for length, vector in zip(pca.explained_variance_, pca.components_):
    v = vector * 3 * np.sqrt(length)
    draw_vector(pca.mean_, pca.mean_ + v)
ax1.axis('equal')

# centre the data
mean_ = np.mean(X, axis=0)
Z = (X - mean_)

# perform the singular value decomposition
U, s, Vt = scipy.linalg.svd(Z, full_matrices=False)

# flip eigenvectors signs to enforce consistent output
U, Vt = svd_flip(U, Vt, u_based_decision=False)

explained_variance_ = (s**2) / (n_samples - 1)
components_ = Vt

print("\nmean:", mean_, pca.mean_)
print("\nvariance:", explained_variance_, pca.explained_variance_)
print("\ncomponents:", components_, pca.components_)

fig2, ax2 = plt.subplots(1, 1)
ax2.scatter(Z[:, 0], Z[:, 1], alpha=0.2)
for length, vector in zip(explained_variance_, components_):
    v = vector * 3.0 * np.sqrt(length)
    draw_vector(mean_, mean_ + v, ax=ax2)
ax2.axis('equal')

plt.show()