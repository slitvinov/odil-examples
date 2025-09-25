import scipy
import matplotlib.pyplot as plt
import math
import numpy as np

class G0:
    pass

class F0:
    pass

def cappend(M, i, j, d):
    M.row.append(len(M.rhs))
    M.col.append(i * nx + j)
    M.data.append(d)

def ini(M):
    M.row = []
    M.col = []
    M.rhs = []
    M.data = []

ini(G0)
ini(F0)

nx = 50
nt = 50
L = 1.0
T = 1.0
sigma = 0.2
dx = 2 * L / (nx - 1)
dt = T / (nt - 1)
c0 = -1 / (2 * dx**2)
c1 = 1 / (2 * dt**2)
c2 = -((dx**2 - dt**2) / (dt**2 * dx**2))
x = np.linspace(-L, L, nx)
for j in range(nx):
    cappend(G0, 0, j, 1)
    G0.rhs.append(math.exp(-(x[j] / sigma)**2))
for j in range(nx):
    cappend(G0, nt - 1, j, 1)
    G0.rhs.append(0)
for i in range(1, nt - 1):
    for j in range(nx):
        if j == 0 or j == nx - 1:
            cappend(F0, i, j, 1)
            F0.rhs.append(0)
        else:
            cappend(F0, i - 1, j, c1)
            cappend(F0, i, j - 1, c0)
            cappend(F0, i, j, c2)
            cappend(F0, i, j + 1, c0)
            cappend(F0, i + 1, j, c1)
            F0.rhs.append(0)
            
dF = scipy.sparse.csr_matrix((F0.data, (F0.row, F0.col)), dtype=float)
dG = scipy.sparse.csr_matrix((G0.data, (G0.row, G0.col)), dtype=float)

# sol = scipy.sparse.linalg.spsolve(dF + dG, rhs)
u = np.asarray(sol).reshape(nt, nx)
for k in 0, nt // 4, nt // 2, 3 * nt // 4, nt - 1:
    plt.plot(x, u[k, :], 'o-', label=f"t={k*dt:.2f}")
plt.legend()
plt.show()
