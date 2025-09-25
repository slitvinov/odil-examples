import scipy
import matplotlib.pyplot as plt
import math
import numpy as np


def cappend(i, j, d):
    row.append(len(rhs))
    col.append(i * nx + j)
    data.append(d)


nx = 50
nt = 50
L = 1.0
T = 1.0
sigma = 0.2
dx = 2 * L / (nx - 1)
dt = T / nt
c0 = -1 / (2 * dx**2)
c1 = 1 / (2 * dt**2)
c2 = -((dx**2 - dt**2) / (dt**2 * dx**2))
x = np.linspace(-L, L, nx)
row = []; col = []; rhs = []; data = []
for i in range(nt):
    for j in range(nx):
        if i == 0:
            cappend(0, j, 1)
            rhs.append(math.exp(-(x[j] / sigma)**2))
        elif i == nt - 1:
            cappend(nt - 1, j, 1)
            rhs.append(0)
        elif j == 0 or j == nx - 1:
            cappend(i, j, 1)
            rhs.append(0)
        else:
            cappend(i - 1, j, c1)
            cappend(i, j - 1, c0)
            cappend(i, j, c2)
            cappend(i, j + 1, c0)
            cappend(i + 1, j, c1)
            rhs.append(0)
A = scipy.sparse.csr_matrix((data, (row, col)), dtype=float)
sol = scipy.sparse.linalg.spsolve(A, rhs)
u = np.asarray(sol).reshape(nt, nx)
for k in 0, nt // 4, nt // 2, 3 * nt // 4, nt - 1:
    plt.plot(x, u[k, :], 'o-', label=f"t={k*dt:.2f}")
plt.legend()
plt.show()
