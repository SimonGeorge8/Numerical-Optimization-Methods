import numpy as np

class UnconstrainedMinimizer:
    """
    A class for unconstrained optimization using gradient descent or Newton methods.
    
    Works with objective functions that follow the assignment interface:
    func(x, hessian_needed) -> (f, g, h)
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
        self.path = []
        self.f_values = []
    
    def minimize(self, f, x0, obj_tol, param_tol, max_iter):
        """
        Minimize the given function using the selected optimization method.
        
        Parameters:
        -----------
        f : callable
            The objective function with interface: f(x, hessian_needed) -> (f_val, grad, hess)
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
        
        # Evaluate function at starting point
        f_curr, _, _ = f(x_curr, hessian_needed=False)
        
        # Initialize storage for path and values
        self.path = [x_curr.copy()]
        self.f_values = [f_curr]
        
        iterations = 0
        converged = False
        
        # Print initial iteration
        print(f"Iteration {iterations}: x = {x_curr}, f(x) = {f_curr:.8f}")
        
        while iterations < max_iter and not converged:
            x_prev = x_curr.copy()
            f_prev = f_curr
            
            # Update step based on selected method
            if self.method == 'gradient_descent':
                # Get gradient for direction
                _, gradient, _ = f(x_curr, hessian_needed=False)
                direction = -gradient
                # Use Wolfe line search with specified constants
                step_size = self._wolfe_line_search(f, x_curr, direction, c1=0.01, backtrack_factor=0.5)
                x_curr = x_curr + step_size * direction
            elif self.method == 'newton':
                x_curr = self._newton_step(f, x_curr)
            else:
                raise ValueError(f"Unknown method: {self.method}")
            
            # Evaluate function at new point
            f_curr, _, _ = f(x_curr, hessian_needed=False)
            iterations += 1
            
            # Store path and values
            self.path.append(x_curr.copy())
            self.f_values.append(f_curr)
            
            # Print iteration info
            print(f"Iteration {iterations}: x = {x_curr}, f(x) = {f_curr:.8f}")
            
            # Check convergence
            converged = self._check_convergence(f_curr, f_prev, x_curr, x_prev, obj_tol, param_tol)
        
        success_flag = converged
        if not converged:
            print(f"Maximum iterations ({max_iter}) reached without convergence")
        else:
            print("Converged successfully!")
        
        return {
            'x': x_curr,
            'f': f_curr,
            'iterations': iterations,
            'converged': success_flag,
            'path': np.array(self.path),
            'f_values': self.f_values,
            'method': self.method
        }
    
    def _newton_step(self, f, x):
        """
        Perform a single Newton step.
        
        Parameters:
        -----------
        f : callable
            The objective function with interface: f(x, hessian_needed) -> (f_val, grad, hess)
        x : array-like
            Current point
            
        Returns:
        --------
        x_new : array-like
            Updated point
        """
        try:
            # Get both gradient and Hessian
            _, gradient, hessian = f(x, hessian_needed=True)
            
            # Check if Hessian is positive definite by trying Cholesky decomposition
            try:
                np.linalg.cholesky(hessian)
                # If successful, solve the linear system
                newton_direction = np.linalg.solve(hessian, -gradient)
                return x + newton_direction
            except np.linalg.LinAlgError:
                # If Hessian is not positive definite, use modified Newton
                # Add small diagonal regularization
                regularized_hessian = hessian + 1e-6 * np.eye(len(hessian))
                try:
                    newton_direction = np.linalg.solve(regularized_hessian, -gradient)
                    return x + newton_direction
                except np.linalg.LinAlgError:
                    # Fallback to gradient descent step
                    return x - 0.01 * gradient
                
        except np.linalg.LinAlgError:
            # If Hessian computation fails, fallback to gradient descent
            _, gradient, _ = f(x, hessian_needed=False)
            return x - 0.01 * gradient
    
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
        # Check absolute change in objective function
        obj_change = abs(f_curr - f_prev)
        
        # Check Euclidean distance between parameters
        param_change = np.linalg.norm(x_curr - x_prev)
        
        # Check if both criteria are met
        obj_converged = obj_change < obj_tol
        param_converged = param_change < param_tol
        
        return obj_converged or param_converged
    
    def _wolfe_line_search(self, f, x, direction, c1=0.01, backtrack_factor=0.5, max_iter=50):
        """
        Perform backtracking line search with Armijo condition (first Wolfe condition).
        
        Parameters:
        -----------
        f : callable
            The objective function with interface: f(x, hessian_needed) -> (f_val, grad, hess)
        x : array-like
            Current point
        direction : array-like
            Search direction
        c1 : float
            Parameter for Armijo condition (should be 0.01 per assignment)
        backtrack_factor : float
            Backtracking factor (should be 0.5 per assignment)
        max_iter : int
            Maximum iterations for line search
            
        Returns:
        --------
        step_size : float
            Step size satisfying Armijo condition
        """
        alpha = 1.0
        f0, grad0, _ = f(x, hessian_needed=False)
        directional_derivative = np.dot(grad0, direction)
        
        # If direction is not a descent direction, return small step
        if directional_derivative >= 0:
            return 1e-4
        
        for i in range(max_iter):
            x_new = x + alpha * direction
            f_new, _, _ = f(x_new, hessian_needed=False)
            
            # Armijo condition (sufficient decrease)
            if f_new <= f0 + c1 * alpha * directional_derivative:
                return alpha
            
            # Backtrack
            alpha *= backtrack_factor
        
        # If line search fails, return a conservative step size
        return 1e-4


# Wrapper class for backward compatibility with old interface
class ObjectiveFunction:
    """
    Base class template for objective functions with analytic derivatives.
    
    This is kept for backward compatibility, but the assignment requires
    functions with interface: func(x, hessian_needed) -> (f, g, h)
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


def create_function_wrapper(objective_function_class):
    """
    Convert an ObjectiveFunction class to the required function interface.
    
    Parameters:
    -----------
    objective_function_class : ObjectiveFunction
        Instance of ObjectiveFunction class
        
    Returns:
    --------
    func : callable
        Function with interface: func(x, hessian_needed) -> (f, g, h)
    """
    def wrapper(x, hessian_needed=False):
        f_val = objective_function_class.func(x)
        grad = objective_function_class.grad(x)
        hess = objective_function_class.hessian(x) if hessian_needed else None
        return f_val, grad, hess
    
    return wrapper