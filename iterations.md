---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.3
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

+++ {"id": "2829bc3c"}

[![Open in
Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/slitvinov/odil-examples/blob/main/iterations.ipynb)

+++ {"id": "3V7O0Eco4DF7"}

# Explanation of the Wave Equation Solver

This notebook cell implements a **least-squares solution** of the **1D wave equation** on a space–time grid.

---

## 1. The PDE

We want to solve the **1D wave equation**:

$$
u_{tt} - u_{xx} = 0, \quad (t,x) \in [0,T]\times[-L,L]
$$

with conditions:

- **Initial condition (at \(t=0\)):**
$$
u(0,x) = \exp\!\Big(-\frac{x^2}{\sigma^2}\Big)
$$

- **Boundary conditions (at \(x=-L,\,x=L\)):**
$$
u(t, -L) = u(t, L) = 0
$$

- **Terminal condition (at \(t=T\)):**
$$
u(T, x) = 0
$$

---

## 2. Discretization

We discretize the domain with:
- \(nx\) spatial points, spacing
$$
dx = \frac{2L}{nx-1}
$$
- \(nt\) time points, spacing
$$
dt = \frac{T}{nt}
$$

The centered finite-difference approximation of the wave operator is:

$$
\frac{u_{i+1,j} - 2u_{i,j} + u_{i-1,j}}{dt^2}
-
\frac{u_{i,j+1} - 2u_{i,j} + u_{i,j-1}}{dx^2} = 0
$$

which, after rearrangement, gives coefficients:

$$
c_1 = \frac{1}{2 dt^2}, \quad
c_0 = -\frac{1}{2 dx^2}, \quad
c_2 = -\frac{dx^2 - dt^2}{dt^2 dx^2}.
$$

Thus, each interior equation couples five unknowns:

$$
c_1 u_{i-1,j} + c_0 u_{i,j-1} + c_2 u_{i,j} + c_0 u_{i,j+1} + c_1 u_{i+1,j} = 0
$$

---

## 3. Sparse matrix construction

The code builds two sparse matrices, `dF` and `dG`, to represent the discrete operators for the PDE and the associated conditions.

- **`dF` (PDE operator):** Encodes the 5-point stencil for the wave equation at all **interior** grid points.

- **`dG` (Conditions operator):** Enforces all other conditions by creating identity rows that select the corresponding `u` values:
  - **Initial condition ($t=0$):** enforces $u(0,x_j) = \exp(-x_j^2/\sigma^2)$.
  - **Boundary conditions ($x=\pm L$):** enforces $u(t, \pm L) = 0$.
  - **Terminal condition ($t=T$):** enforces $u(T, x_j) = 0$.

Both matrices are stored in the efficient **CSR (Compressed Sparse Row)** format.

---

## 4. Least-squares formulation

Following the ODIL (Operator Discretization and Inference Library) framework, we formulate the problem as minimizing a loss function $L(u)$, which is the sum of squared residuals:

$$
L(u) = \|F[u]\|_2^2 + \|G[u]\|_2^2
$$

Here, $F[u]$ and $G[u]$ are discrete operators representing the residuals of the PDE and the associated conditions. For this linear problem, they take the form:

- $F[u] = dF \cdot u - f$: The residual of the wave equation at interior points. `dF` is the sparse matrix for the PDE, and the right-hand-side vector `f` is zero.
- $G[u] = dG \cdot u - g$: The residual for the initial, boundary, and terminal conditions. `dG` selects grid points on the domain boundary, and `g` contains the target values for these conditions.

The vector $u$ represents the solution at all grid points, flattened into a single vector. We seek the $u$ that minimizes $L(u)$. This is a linear least-squares problem, and its solution is found by solving the **normal equations**:

$$
(dF^T dF + dG^T dG) u = dF^T f + dG^T g
$$

The code implements a Newton-Raphson iterative solver. For a general (non-linear) problem, the update step is:

$$
M (u_{new} - u_{old}) = - (dF^T F_s + dG^T G_s)
$$

where:
- $M = dF^T dF + dG^T dG$ is the Hessian matrix.
- $F_s = dF \cdot u_{old} - f$ is the PDE residual at the current iteration (`Fs` in the code).
- $G_s = dG \cdot u_{old} - g$ is the conditions residual at the current iteration (`Gs` in the code).

The right-hand side of the update is the negative gradient of the loss function. Since our problem is linear, the solver converges to the exact solution in a single step (if starting from $u_{old}=0$).

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 518
id: 6da60b8c
outputId: b7d58b2d-fb82-4f36-b6cc-2226711db497
---
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

# dF for PDE at interior points
row = []
col = []
rhs = []
data = []
for i in range(1, nt - 1):
    for j in range(1, nx - 1):
	cappend(i - 1, j, c1)
	cappend(i, j - 1, c0)
	cappend(i, j, c2)
	cappend(i, j + 1, c0)
	cappend(i + 1, j, c1)
	rhs.append(0)
dF = scipy.sparse.csr_matrix((data, (row, col)), shape=(len(rhs), nt * nx), dtype=float)
f = np.array(rhs, dtype=float)

# dG for initial, boundary and terminal conditions
row = []
col = []
rhs = []
data = []
for i in range(nt):
    for j in range(nx):
	if i == 0:
	    # Initial condition
	    cappend(i, j, 1)
	    rhs.append(math.exp(-(x[j] / sigma)**2) *
		       math.cos(math.pi * x[j] / L))
	elif i == nt - 1:
	    # Terminal condition
	    cappend(i, j, 1)
	    rhs.append(0)
	elif j == 0 or j == nx - 1:
	    # Boundary conditions
	    cappend(i, j, 1)
	    rhs.append(0)
dG = scipy.sparse.csr_matrix((data, (row, col)), shape=(len(rhs), nt * nx), dtype=float)
g = np.array(rhs, dtype=float)

us = np.zeros(nt * nx)
for i in range(5):
    Fs = dF @ us - f
    Gs = dG @ us - g
    M = dF.T @ dF + dG.T @ dG
    rhs = M @ us - dF.T @ Fs - dG.T @ Gs
    usp = scipy.sparse.linalg.spsolve(M, rhs)
    print(f"diff: {np.mean((usp - us)**2):8.4e}")
    us = usp
u = np.asarray(us).reshape(nt, nx)
for k in 0, nt // 4, nt // 2, 3 * nt // 4, nt - 1:
    plt.plot(x, u[k, :], 'o-', label=f"t={k*dt:.2f}")
plt.legend();
```
