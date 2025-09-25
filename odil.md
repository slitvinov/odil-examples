In the following, we formulate the ODIL framework for the one-dimensional wave equation \(u_{tt} = u_{xx}\) discretized with finite differences. Instead of solving this equation by marching in time from known initial conditions, we rewrite the problem as a minimization of the loss function

\[
L(u) = \sum_{(i,n)\in\Omega_h} 
\left( 
\frac{u^{n+1}_i - 2u^n_i + u^{n-1}_i}{\Delta t^2} 
- 
\frac{u^n_{i+1} - 2u^n_i + u^n_{i-1}}{\Delta x^2}
\right)^2
+ 
\sum_{(i,n)\in\Gamma_h} (u^n_i - g^n_i)^2,
\]

where the unknown parameter is the solution itself, a discrete field \(u^n_i\) on a Cartesian grid. The loss function includes the residual of the discrete equation in grid points \(\Omega_h\) and imposes boundary and initial conditions \(u = g\) in grid points \(\Gamma_h\). To solve this problem with a gradient-based method, such as Adam [8] or L-BFGS-B [22], we only need to compute the gradient of the loss which is also a discrete field \(\partial L / \partial u\). To apply Newton’s method, we assume that the loss function is a sum of quadratic terms such as \(L(u) = \|F[u]\|_2^2 + \|G[u]\|_2^2\) with discrete operators \(F[u]\) and \(G[u]\) and linearize the operators about the current solution \(u^s\) to obtain a quadratic loss

\[
L^s(u^{s+1}) = \|F^s + (\partial F^s/\partial u)(u^{s+1} - u^s)\|_2^2 
+ \|G^s + (\partial G^s/\partial u)(u^{s+1} - u^s)\|_2^2,
\]

where \(F^s\) and \(G^s\) denote \(F[u^s]\) and \(G[u^s]\). A minimum of this function provides the solution \(u^{s+1}\) at the next iteration and satisfies a linear system

\[
\Big(
(\partial F^s/\partial u)^T (\partial F^s/\partial u) 
+ (\partial G^s/\partial u)^T (\partial G^s/\partial u)
\Big)(u^{s+1} - u^s) 
+ (\partial F^s/\partial u)^T F^s + (\partial G^s/\partial u)^T G^s = 0.
\]

Cases with terms other than \(F\) and \(G\) are handled similarly. We further assume that \(F[u]\) and \(G[u]\) at each grid point depend only on the value of \(u\) in the neighboring points. This makes the derivatives \(\partial F / \partial u\) and \(\partial G / \partial u\) sparse matrices. To implement this procedure, we use automatic differentiation in TensorFlow [1] and solve the linear system with either a direct [20] or multigrid sparse linear solver [2].

