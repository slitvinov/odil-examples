import scipy
import matplotlib.pyplot as plt
import math
import numpy as np


def cappend(i, j, d):
    row.append(len(rhs))
    col.append(i * nx + j)
    data.append(d)


row = []
col = []
rhs = []
data = []
nx = 50
nt = 10
L = 1.0
T = 10.0
alpha = 0.01
sigma = 0.2
dx = 2 * L / (nx - 1)
dt = T / nt
c0 = -alpha / (dx**2)
c1 = 1.0 / dt + 2.0 * alpha / (dx**2)
c2 = -1.0 / dt
x = np.linspace(-L, L, nx)
for j in range(nx):
    cappend(0, j, 1)
    rhs.append(math.exp(-(x[j] / sigma)**2))
for i in range(1, nt):
    for j in range(nx):
        if j == 0 or j == nx - 1:
            cappend(i, j, 1)
            rhs.append(0.0)
        else:
            cappend(i, j - 1, c0)
            cappend(i, j, c1)
            cappend(i, j + 1, c0)
            cappend(i - 1, j, c2)
            rhs.append(0.0)
A = scipy.sparse.csr_matrix((data, (row, col)), dtype=float)
sol = scipy.sparse.linalg.spsolve(A, rhs)
u = np.asarray(sol).reshape(nt, nx)
for k in 0, nt // 4, nt // 2, 3 * nt // 4, nt - 1:
    plt.plot(x, u[k, :], 'o-', label=f"t={k*dt:.2f}")
plt.show()
