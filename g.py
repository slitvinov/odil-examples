import numpy as np
import matplotlib.pyplot as plt

L = 1.0
T = 10.0
nx = 50
nt = 10000
oalpha = 0.01
u = np.zeros((nt + 1, nx))
x = np.linspace(-L, L, nx)
sigma = 0.2
u[0, :] = np.exp(-(x / sigma)**2)
dx = 2 * L / (nx - 1)
dt = T / nt
q = alpha * dt / dx**2
if 2 * q > 1:
    print(
        "Warning: Stability condition not met. The solution may be unstable.")


def total_cost_2d(u_2d):
    cost = 0.0
    for n in range(nt):
        for i in range(1, nx - 1):
            term = (u_2d[n + 1, i] - u_2d[n, i] + alpha * dt / dx**2 *
                    (u_2d[n, i + 1] - 2 * u_2d[n, i] + u_2d[n, i - 1]))
            cost += term**2
    return cost


def grad_total_cost_2d(u_2d):
    R = residual(u_2d)
    G = np.zeros_like(u_2d)
    G[1:, 1:-1] += 2 * R
    G[:-1, 1:-1] += -2 * (1 + 2 * c) * R
    G[:-1, 2:] += 2 * c * R
    G[:-1, :-2] += 2 * c * R
    G[1:nt, 1:-1] += 2 * R[:-1, :]
    G[0, :] = 0.0
    G[:, 0] = 0.0
    G[:, -1] = 0.0
    return G


from scipy.optimize import minimize

initial_guess_flat = np.tile(u[0, :], nt + 1)
bounds = []
for n in range(nt + 1):
    for i in range(nx):
        if n == 0:
            bounds.append((u[0, i], u[0, i]))
        elif i == 0 or i == nx - 1:
            bounds.append((0.0, 0.0))
        else:
            bounds.append((None, None))


def objective_function(u_flat):
    print(len(u_flat))
    u_2d = u_flat.reshape((nt + 1, nx))
    return total_cost_2d(u_2d)


def gradient_function(u_flat):
    print("preved")
    u_2d = u_flat.reshape((nt + 1, nx))
    g2d = grad_total_cost_2d(u_2d)
    return g2d.ravel()


result = minimize(objective_function,
                  initial_guess_flat,
                  method='Newton-CG',
                  jac=gradient_function,
                  bounds=bounds,
                  options={
                      'disp': True,
                      'maxiter': 10
                  })
u = result.x.reshape((nt + 1, nx))
print(f"Final cost after optimization: {result.fun}")
time_steps_to_plot = [0, nt // 4, nt // 2, 3 * nt // 4, nt]
for i in time_steps_to_plot:
    plt.plot(x, u[i, :], label=f'Time step {i}')
T_grid, X_grid = np.meshgrid(np.linspace(0, T, nt + 1), x)
plt.contourf(X_grid, T_grid, u.T, 100, cmap='viridis')
plt.colorbar(label='u')
