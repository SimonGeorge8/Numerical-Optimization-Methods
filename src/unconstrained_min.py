import numpy as np

class UnconstrainedMinimizer:
    """
    A class for unconstrained optimization using gradient descent or Newton methods.
    """
    
    def __init__(self, method='gradient_descent'):
        """
        Initialize the minimizer.
        
        Parameters:
        -----------
        method : str
            The optimization method to use ('gradient_descent' or 'newton')
        """
        self.method = method
    
    def minimize(self, f, x0, obj_tol, param_tol, max_iter):
        """
        Minimize the given function using the selected optimization method.
        
        Parameters:
        -----------
        f : object
            The objective function object with methods:
            - f.func(x): evaluate function at x
            - f.grad(x): evaluate gradient at x  
            - f.hessian(x): evaluate Hessian matrix at x
        x0 : array-like
            Starting point for optimization
        obj_tol : float
            Tolerance for objective function convergence
        param_tol : float
            Tolerance for parameter convergence
        max_iter : int
            Maximum number of iterations
            
        Returns:
        --------
        result : dict
            Dictionary containing optimization results
        """
        x_curr = np.array(x0, dtype=float)
        x_prev = x_curr.copy()
        f_curr = f.func(x_curr)
        f_prev = f_curr
        
        iterations = 0
        converged = False
        
        while iterations < max_iter and not converged:
            # Store previous values
            x_prev = x_curr.copy()
            f_prev = f_curr
            
            # Update step based on selected method
            if self.method == 'gradient_descent':
                # Use line search to find optimal step size
                direction = -f.grad(x_curr)
                step_size = self._line_search(f, x_curr, direction)
                x_curr = x_curr + step_size * direction
            elif self.method == 'newton':
                x_curr = self._newton_step(f, x_curr)
            
            # Evaluate function at new point
            f_curr = f.func(x_curr)
            
            # Check convergence
            converged = self._check_convergence(f_curr, f_prev, x_curr, x_prev, obj_tol, param_tol)
            iterations += 1
        
        return {
            'x': x_curr,
            'f': f_curr,
            'iterations': iterations,
            'converged': converged
        }
    
    def _gradient_descent_step(self, f, x, step_size):
        """
        Perform a single gradient descent step.
        
        Parameters:
        -----------
        f : object
            The objective function object with analytic derivatives
        x : array-like
            Current point
        step_size : float
            Step size for the update
            
        Returns:
        --------
        x_new : array-like
            Updated point
        """
        return x - step_size * f.grad(x)
    
    def _newton_step(self, f, x):
        """
        Perform a single Newton step.
        
        Parameters:
        -----------
        f : object
            The objective function object with analytic derivatives
        x : array-like
            Current point
            
        Returns:
        --------
        x_new : array-like
            Updated point
        """
        try:
            # Solve H * p = -g for the Newton direction p
            hessian = f.hessian(x)
            gradient = f.grad(x)
            
            # Check if Hessian is positive definite by trying Cholesky decomposition
            try:
                np.linalg.cholesky(hessian)
                # If successful, solve the linear system
                newton_direction = np.linalg.solve(hessian, -gradient)
            except np.linalg.LinAlgError:
                # If Hessian is not positive definite, use gradient descent direction
                newton_direction = -gradient
                
            return x + newton_direction
            
        except np.linalg.LinAlgError:
            # If Hessian is singular, fallback to gradient descent
            return x - 0.01 * f.grad(x)
    
    
    def _check_convergence(self, f_curr, f_prev, x_curr, x_prev, obj_tol, param_tol):
        """
        Check if convergence criteria are met.
        
        Parameters:
        -----------
        f_curr : float
            Current function value
        f_prev : float
            Previous function value
        x_curr : array-like
            Current parameter values
        x_prev : array-like
            Previous parameter values
        obj_tol : float
            Objective function tolerance
        param_tol : float
            Parameter tolerance
            
        Returns:
        --------
        converged : bool
            True if converged, False otherwise
        """
        # Check relative change in objective function
        obj_change = abs(f_curr - f_prev) / max(abs(f_prev), 1)
        
        # Check relative change in parameters
        param_change = np.linalg.norm(x_curr - x_prev) / max(np.linalg.norm(x_prev), 1)
        
        # Return True if both conditions are met
        return obj_change < obj_tol and param_change < param_tol
    
    
    def _line_search(self, f, x, direction, alpha_init=1.0, c1=1e-4, c2=0.9, max_iter=50):
        """
        Perform line search to find optimal step size using Wolfe conditions.
        
        Parameters:
        -----------
        f : object
            The objective function object with analytic derivatives
        x : array-like
            Current point
        direction : array-like
            Search direction
        alpha_init : float
            Initial step size
        c1 : float
            Parameter for Armijo condition
        c2 : float
            Parameter for curvature condition
        max_iter : int
            Maximum iterations for line search
            
        Returns:
        --------
        step_size : float
            Optimal step size
        """
        alpha = alpha_init
        f0 = f.func(x)
        grad0 = np.dot(f.grad(x), direction)
        
        # If direction is not a descent direction, return small step
        if grad0 >= 0:
            return 0.01
        
        alpha_low = 0.0
        alpha_high = None
        
        for i in range(max_iter):
            x_new = x + alpha * direction
            f_new = f.func(x_new)
            
            # Armijo condition (sufficient decrease)
            if f_new > f0 + c1 * alpha * grad0:
                alpha_high = alpha
                alpha = (alpha_low + alpha_high) / 2.0
            else:
                grad_new = np.dot(f.grad(x_new), direction)
                
                # Curvature condition
                if abs(grad_new) <= c2 * abs(grad0):
                    return alpha
                
                if grad_new >= 0:
                    alpha_high = alpha_low
                    alpha_low = alpha
                else:
                    alpha_low = alpha
                
                if alpha_high is not None:
                    alpha = (alpha_low + alpha_high) / 2.0
                else:
                    alpha *= 2.0
        
        # If line search fails, return a conservative step size
        return 0.01


class ObjectiveFunction:
    """
    Base class template for objective functions with analytic derivatives.
    
    Each objective function should inherit from this class and implement
    the exact analytic formulas for f, gradient, and Hessian.
    """
    
    def func(self, x):
        """
        Evaluate the objective function at point x.
        
        Parameters:
        -----------
        x : array-like
            Point at which to evaluate function
            
        Returns:
        --------
        value : float
            Function value f(x)
        """
        raise NotImplementedError("Must implement analytic function formula")
    
    def grad(self, x):
        """
        Evaluate the gradient of the objective function at point x.
        
        Parameters:
        -----------
        x : array-like
            Point at which to evaluate gradient
            
        Returns:
        --------
        gradient : array-like
            Gradient vector ∇f(x)
        """
        raise NotImplementedError("Must implement analytic gradient formula")
    
    def hessian(self, x):
        """
        Evaluate the Hessian matrix of the objective function at point x.
        
        Parameters:
        -----------
        x : array-like
            Point at which to evaluate Hessian
            
        Returns:
        --------
        hessian : array-like
            Hessian matrix ∇²f(x)
        """
        raise NotImplementedError("Must implement analytic Hessian formula")
