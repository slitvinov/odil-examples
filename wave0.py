import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
def solve_wave_equation_odil(nx, nt, dx, dt, g_conditions,
boundary_mask, iterations=10):
     num_vars = nt * nx
     u_flat = g_conditions.flatten()
     grid_indices = np.arange(num_vars).reshape((nt, nx))
     interior_indices = grid_indices[~boundary_mask]
     boundary_indices = grid_indices[boundary_mask]
     num_interior = len(interior_indices)
     num_boundary = len(boundary_indices)
     Jg_row = np.arange(num_boundary)
     Jg_col = boundary_indices
     Jg_data = np.ones(num_boundary)
     J_g = sp.coo_matrix((Jg_data, (Jg_row, Jg_col)),
shape=(num_boundary, num_vars)).tocsr()
     Jf_rows, Jf_cols, Jf_data = [], [], []
     c_dt2 = 1.0 / (dt**2)
     c_dx2 = 1.0 / (dx**2)
     for k, flat_idx in enumerate(interior_indices):
         n, i = np.unravel_index(flat_idx, (nt, nx))
         stencil = {
             (n, i):     -2*c_dt2 - (-2*c_dx2), (n+1, i): c_dt2,
             (n-1, i):   c_dt2,                 (n, i+1): -c_dx2,
             (n, i-1):   -c_dx2
         }
         for (row_idx, col_idx), val in stencil.items():
             Jf_rows.append(k)
             Jf_cols.append(grid_indices[row_idx, col_idx])
             Jf_data.append(val)
     J_f = sp.coo_matrix((Jf_data, (Jf_rows, Jf_cols)),
shape=(num_interior, num_vars)).tocsr()
     A = (J_f.T @ J_f) + (J_g.T @ J_g)
     A = A.tocsc()
     g_flat = g_conditions.flatten()
     loss_history = []
     # print("Starting Newton iterations...")
     for it in range(iterations):
         F = J_f @ u_flat
         G = u_flat[boundary_indices] - g_flat[boundary_indices]
         loss = 0.5 * (np.dot(F, F) + np.dot(G, G))
         loss_history.append(loss)
         b = J_f.T @ F + J_g.T @ G
         du = spsolve(A, -b)
         u_flat += du
         if np.linalg.norm(du) < 1e-8:
             break
     return u_flat.reshape((nt, nx)), loss_history

# (The exact solver function 'solve_wave_equation_exact' remains the same)
def solve_wave_equation_exact(nx, nt, dx, dt, initial_condition_func):
     # print("Calculating exact solution...")
     L = (nx - 1) * dx
     x = np.linspace(0, L, nx)
     u_exact = np.zeros((nt, nx))
     def f_extended(y):
         y_mapped = (y + L) % (2 * L) - L
         return np.sign(y_mapped) * initial_condition_func(np.abs(y_mapped))
     for n in range(nt):
         t = n * dt
         u_exact[n, :] = 0.5 * (f_extended(x + t) + f_extended(x - t))
     # print("Finished exact solution.")
     return u_exact

if __name__ == '__main__':
     nx, nt, L, c = 101, 201, 1.0, 1.0
     dx = L / (nx - 1)
     dt = 0.5 * dx / c

     # Define the initial condition as a function
     def initial_gaussian(x_coords):
         # This function is now only used for the exact solver's logic
         return np.exp(-((x_coords - L/2)**2) / 0.01)

     x = np.linspace(0, L, nx)
     g_conditions = np.zeros((nt, nx))
     boundary_mask = np.zeros((nt, nx), dtype=bool)

     # --- Set up conditions for the numerical solver ---

     # --- THIS IS THE FIX ---
     # 1. Create the initial shape
     initial_shape = initial_gaussian(x)
     # 2. Force it to be exactly zero at the boundaries
     initial_shape[0] = 0.0
     initial_shape[-1] = 0.0
     # 3. Assign it to the conditions
     g_conditions[0, :] = initial_shape
     # --- END OF FIX ---

     boundary_mask[0, :] = True
     g_conditions[1, :] = g_conditions[0, :] # Initial velocity = 0
     boundary_mask[1, :] = True
     boundary_mask[:, 0] = True   # u(0, t) = 0
     boundary_mask[:, -1] = True  # u(L, t) = 0
     boundary_mask[-1, :] = True  # Final time is a boundary

     # --- Solve both numerically and exactly ---
     u_numerical, losses = solve_wave_equation_odil(
         nx, nt, dx, dt, g_conditions, boundary_mask, iterations=15
     )
     u_exact = solve_wave_equation_exact(nx, nt, dx, dt, initial_gaussian)

     # --- Plotting Comparison ---
     fig = plt.figure(figsize=(15, 10))

     ax1 = fig.add_subplot(2, 2, 1)
     im = ax1.imshow(u_numerical, aspect='auto', origin='lower',
extent=[0, L, 0, nt*dt])
     fig.colorbar(im, ax=ax1, label='u(x,t)')
     ax1.set_title('Numerical Solution (ODIL)')
     ax1.set_xlabel('Position (x)')
     ax1.set_ylabel('Time (t)')

     ax2 = fig.add_subplot(2, 2, 2)
     im = ax2.imshow(u_exact, aspect='auto', origin='lower', extent=[0,
L, 0, nt*dt])
     fig.colorbar(im, ax=ax2, label='u(x,t)')
     ax2.set_title('Exact Solution (d\'Alembert)')
     ax2.set_xlabel('Position (x)')
     ax2.set_ylabel('Time (t)')

     ax3 = fig.add_subplot(2, 2, 3)
     time_indices_to_plot = [0, int(nt/4), int(nt/2), int(3*nt/4)]
     for n_idx in time_indices_to_plot:
         time = n_idx * dt
         ax3.plot(x, u_numerical[n_idx, :], 'o', markersize=4,
label=f'Numerical t={time:.2f}')
         ax3.plot(x, u_exact[n_idx, :], '-', label=f'Exact t={time:.2f}')
     ax3.set_title('Solution Profiles at Different Times')
     ax3.set_xlabel('Position (x)')
     ax3.set_ylabel('Amplitude u(x)')
     ax3.legend()
     ax3.grid(True)

     ax4 = fig.add_subplot(2, 2, 4)
     error = np.abs(u_numerical - u_exact)
     im = ax4.imshow(error, aspect='auto', origin='lower', extent=[0, L,
0, nt*dt], vmax=error.max())
     fig.colorbar(im, ax=ax4, label='|Error|')
     ax4.set_title('Absolute Error |Numerical - Exact|')
     ax4.set_xlabel('Position (x)')
     ax4.set_ylabel('Time (t)')

     plt.tight_layout()
     plt.show()
