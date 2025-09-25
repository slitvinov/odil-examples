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
sigma = 0.2
dx = 2 * L / (nx - 1)
dt = T / nt
c0 = -1/(2*dx**2)
c1 = 1/(2*dt**2)
c2 = -((dx**2-dt**2)/(dt**2*dx**2))
x = np.linspace(-L, L, nx)
for j in range(nx):
    cappend(0, j, 1)
    rhs.append(math.exp(-(x[j] / sigma)**2))
for j in range(nx):
    cappend(nt - 1, j, 1)
    rhs.append(0)
for i in range(1, nt - 1):
    for j in range(nx):
        if j == 0 or j == nx - 1:
            cappend(i, j, 1)
            rhs.append(0)
        else:
            cappend(i-1, j, c1)
            cappend(i, j-1, c0)
            cappend(i, j, c2)
            cappend(i, j+1, c0)
            cappend(i+1, j, c1)
            rhs.append(0)
A = scipy.sparse.csr_matrix((data, (row, col)), dtype=float)
sol = scipy.sparse.linalg.spsolve(A, rhs)
u = np.asarray(sol).reshape(nt, nx)
for k in 0, nt // 4, nt // 2, 3 * nt // 4, nt - 1:
    plt.plot(x, u[k, :], 'o-', label=f"t={k*dt:.2f}")
plt.show()
