"""
Example implementations of objective functions with analytic derivatives.

Each function class should implement exact analytic formulas for:
1. f(x) - the objective function
2. ∇f(x) - the gradient (first derivatives)  
3. ∇²f(x) - the Hessian matrix (second derivatives)

Note: Derive these formulas on paper first, then implement the exact expressions.
"""

import numpy as np
from unconstrained_min import ObjectiveFunction


class QuadraticFunction(ObjectiveFunction):
    """
    Example: Quadratic function f(x) = (1/2) * x^T * Q * x + b^T * x + c
    
    Analytic derivatives:
    - ∇f(x) = Q * x + b
    - ∇²f(x) = Q
    """
    
    def __init__(self, Q, b, c):
        """
        Parameters:
        -----------
        Q : array-like, shape (n, n)
            Quadratic term matrix (should be positive definite for minimization)
        b : array-like, shape (n,)
            Linear term vector
        c : float
            Constant term
        """
        pass
    
    def func(self, x):
        """
        Evaluate f(x) = (1/2) * x^T * Q * x + b^T * x + c
        """
        pass
    
    def grad(self, x):
        """
        Evaluate ∇f(x) = Q * x + b
        """
        pass
    
    def hessian(self, x):
        """
        Evaluate ∇²f(x) = Q
        """
        pass


class RosenbrockFunction(ObjectiveFunction):
    """
    Example: Rosenbrock function f(x,y) = (a-x)² + b(y-x²)²
    Default: a=1, b=100 gives f(x,y) = (1-x)² + 100(y-x²)²
    
    Analytic derivatives (derive these on paper):
    - ∂f/∂x = ?  # TODO: derive and implement
    - ∂f/∂y = ?  # TODO: derive and implement
    - ∂²f/∂x² = ?  # TODO: derive and implement
    - ∂²f/∂x∂y = ?  # TODO: derive and implement
    - ∂²f/∂y² = ?  # TODO: derive and implement
    """
    
    def __init__(self, a=1, b=100):
        """
        Parameters:
        -----------
        a : float
            Parameter for (a-x)² term
        b : float  
            Parameter for b(y-x²)² term
        """
        pass
    
    def func(self, x):
        """
        Evaluate f(x,y) = (a-x)² + b(y-x²)²
        """
        pass
    
    def grad(self, x):
        """
        Evaluate gradient vector [∂f/∂x, ∂f/∂y]
        """
        pass
    
    def hessian(self, x):
        """
        Evaluate Hessian matrix:
        [[∂²f/∂x², ∂²f/∂x∂y],
         [∂²f/∂x∂y, ∂²f/∂y²]]
        """
        pass


class CircleFunction(ObjectiveFunction):
    """
    Example: Circle function f(x,y) = x² + y²
    
    Analytic derivatives:
    - ∇f(x,y) = [2x, 2y]
    - ∇²f(x,y) = [[2, 0], [0, 2]]
    """
    
    def func(self, x):
        """
        Evaluate f(x,y) = x² + y²
        """
        pass
    
    def grad(self, x):
        """
        Evaluate ∇f(x,y) = [2x, 2y]
        """
        pass
    
    def hessian(self, x):
        """
        Evaluate ∇²f(x,y) = [[2, 0], [0, 2]]
        """
        pass


class CustomFunction(ObjectiveFunction):
    """
    Template for implementing your own objective function.
    
    Steps:
    1. Define your function f(x) mathematically
    2. Compute ∇f(x) by hand (partial derivatives)
    3. Compute ∇²f(x) by hand (second partial derivatives)
    4. Implement the exact analytic formulas below
    """
    
    def func(self, x):
        """
        TODO: Implement your function f(x)
        """
        pass
    
    def grad(self, x):
        """
        TODO: Implement your gradient ∇f(x)
        """
        pass
    
    def hessian(self, x):
        """
        TODO: Implement your Hessian ∇²f(x)
        """
        pass