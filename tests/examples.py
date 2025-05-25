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
    Example: Quadratic function f(x) = (1/2) * x^T * Q * x
    
    Analytic derivatives:
    - ∇f(x) = Q * x
    - ∇²f(x) = Q
    """
    
    def __init__(self, Q):
        """
        Parameters:
        -----------
        Q : array-like, shape (n, n)
            Quadratic term matrix (should be positive definite for minimization)
        """
        self.Q = Q

        pass
    
    def func(self, x):
        """
        Evaluate f(x) = (1/2) * x^T * Q * x 
        """
        return float((0.5) * (x.T) @ self.Q @ x)
    
    def grad(self, x):
        """
        Evaluate ∇f(x) = Q * x 
        """
        return self.Q @ x
    
    def hessian(self):
        """
        Evaluate ∇²f(x) = Q
        """
        return self.Q

class LinearFunction:
    """
    Implements a linear function f(x) = a^T x
    where a is a constant nonzero vector.
    """

    def __init__(self, a=np.array([1.0, -2.0])):
        """
        Parameters:
        -----------
        a : np.array
            Nonzero vector of shape (2,)
        """
        self.a = a

    def func(self, x1, x2):
        """
        Evaluate f(x) = a^T x
        """
        x = np.array([x1, x2])
        return float(np.dot(self.a, x))

    def grad(self, x1, x2):
        """
        Gradient of a linear function is constant and equals vector a
        """
        return self.a.copy()

    def hessian(self, x1, x2):
        """
        Hessian of a linear function is always zero matrix
        """
        return np.zeros((2, 2))
    
class RosenbrockFunction(ObjectiveFunction):
    """
    Example: Rosenbrock function f(x,y) = (a-x)² + b(y-x²)²
    Default: a=1, b=100 gives f(x,y) = (1-x)² + 100(y-x²)²
    
    Analytic derivatives (derive these on paper):
    - ∂f/∂x = -4*b*(y-x^2) - 2(a-x)
    - ∂f/∂y = 2*b*(y-x^2)
    - ∂²f/∂x² = 2-4b(y-x^2)+8bx^2
    - ∂²f/∂x∂y = - 4*b*x
    - ∂²f/∂y² = 2*b
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
        self.a = a
        self.b = b
    
    def func(self, x, y):
        """
        Evaluate f(x,y) = (a-x)² + b(y-x²)²
        """
        a = self.a
        b = self.b
        return float((a-x)**2 + b*(y-x**2)**2)
    
    def grad(self, x, y):
        """
        Evaluate gradient vector [∂f/∂x, ∂f/∂y]
        """
        a = self.a
        b = self.b
        rep = y - x**2
        df_dx = -4*b*x*rep - 2*(a - x)
        df_dy = 2*b*rep
        return np.array([df_dx, df_dy])

    def hessian(self, x, y):
        """
        Evaluate Hessian matrix:
        [[∂²f/∂x², ∂²f/∂x∂y],
         [∂²f/∂x∂y, ∂²f/∂y²]]
        """
        b = self.b
        rep = y - x**2
        d2f_dx2 = 2 - 4*b*rep + 8*b*x**2
        d2f_dxdy = -4*b*x
        d2f_dy2 = 2*b
        return np.array([[d2f_dx2, d2f_dxdy],
                         [d2f_dxdy, d2f_dy2]])

class SmoothedCornerTrianglesFunction:
    """
    Implements the function:
        f(x1, x2) = exp(x1 + 3x2 - 0.1) + exp(x1 - 3x2 - 0.1) + exp(-x1 - 0.1)
    As seen in Boyd, p. 470, example 9.20.
    """

    def func(self, x1, x2):
        """
        Compute f(x1, x2)
        """
        term1 = np.exp(x1 + 3*x2 - 0.1)
        term2 = np.exp(x1 - 3*x2 - 0.1)
        term3 = np.exp(-x1 - 0.1)
        return float(term1 + term2 + term3)

    def grad(self, x1, x2):
        """
        Gradient: [∂f/∂x1, ∂f/∂x2]
        """
        term1 = np.exp(x1 + 3*x2 - 0.1)
        term2 = np.exp(x1 - 3*x2 - 0.1)
        term3 = np.exp(-x1 - 0.1)

        df_dx1 = term1 + term2 - term3
        df_dx2 = 3*term1 - 3*term2
        return np.array([df_dx1, df_dx2])

    def hessian(self, x1, x2):
        """
        Hessian matrix:
        [[∂²f/∂x1², ∂²f/∂x1∂x2],
         [∂²f/∂x2∂x1, ∂²f/∂x2²]]
        """
        term1 = np.exp(x1 + 3*x2 - 0.1)
        term2 = np.exp(x1 - 3*x2 - 0.1)
        term3 = np.exp(-x1 - 0.1)

        d2f_dx1x1 = term1 + term2 + term3
        d2f_dx1x2 = 3*term1 - 3*term2
        d2f_dx2x2 = 9*term1 + 9*term2

        return np.array([[d2f_dx1x1, d2f_dx1x2],
                         [d2f_dx1x2, d2f_dx2x2]])
    

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