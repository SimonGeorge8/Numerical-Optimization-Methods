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
        pass
    
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
        pass
    
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
        pass
    
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
        pass
    
    def _line_search(self, f, x, direction):
        """
        Perform line search to find optimal step size.
        
        Parameters:
        -----------
        f : object
            The objective function object with analytic derivatives
        x : array-like
            Current point
        direction : array-like
            Search direction
            
        Returns:
        --------
        step_size : float
            Optimal step size
        """
        pass
