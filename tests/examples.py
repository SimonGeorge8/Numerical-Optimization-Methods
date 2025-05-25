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
        self.Q = np.array(Q)
    
    def func(self, x):
        """
        Evaluate f(x) = (1/2) * x^T * Q * x 
        """
        x = np.array(x)
        return float(0.5 * x.T @ self.Q @ x)
    
    def grad(self, x):
        """
        Evaluate ∇f(x) = Q * x 
        """
        x = np.array(x)
        return self.Q @ x
    
    def hessian(self, x):
        """
        Evaluate ∇²f(x) = Q
        """
        return self.Q


class LinearFunction(ObjectiveFunction):
    """
    Implements a linear function f(x) = a^T x
    where a is a constant nonzero vector.
    
    Note: Linear functions have no minimum (unbounded below)
    unless constrained. This is mainly for testing.
    """

    def __init__(self, a=np.array([1.0, -2.0])):
        """
        Parameters:
        -----------
        a : array-like
            Nonzero vector
        """
        self.a = np.array(a)

    def func(self, x):
        """
        Evaluate f(x) = a^T x
        """
        x = np.array(x)
        return float(np.dot(self.a, x))

    def grad(self, x):
        """
        Gradient of a linear function is constant and equals vector a
        """
        return self.a.copy()

    def hessian(self, x):
        """
        Hessian of a linear function is always zero matrix
        """
        x = np.array(x)
        return np.zeros((len(x), len(x)))


class RosenbrockFunction(ObjectiveFunction):
    """
    Example: Rosenbrock function f(x) = (a-x₁)² + b(x₂-x₁²)²
    Default: a=1, b=100 gives f(x) = (1-x₁)² + 100(x₂-x₁²)²
    
    Analytic derivatives (derive these on paper):
    - ∂f/∂x₁ = -2(a-x₁) - 4bx₁(x₂-x₁²)
    - ∂f/∂x₂ = 2b(x₂-x₁²)
    - ∂²f/∂x₁² = 2 - 4b(x₂-x₁²) + 8bx₁²
    - ∂²f/∂x₁∂x₂ = -4bx₁
    - ∂²f/∂x₂² = 2b
    """
    
    def __init__(self, a=1, b=100):
        """
        Parameters:
        -----------
        a : float
            Parameter for (a-x₁)² term
        b : float  
            Parameter for b(x₂-x₁²)² term
        """
        self.a = a
        self.b = b
    
    def func(self, x):
        """
        Evaluate f(x) = (a-x₁)² + b(x₂-x₁²)²
        """
        x = np.array(x)
        if len(x) != 2:
            raise ValueError("Rosenbrock function requires 2D input")
        
        x1, x2 = x[0], x[1]
        return float((self.a - x1)**2 + self.b * (x2 - x1**2)**2)
    
    def grad(self, x):
        """
        Evaluate gradient vector [∂f/∂x₁, ∂f/∂x₂]
        """
        x = np.array(x)
        if len(x) != 2:
            raise ValueError("Rosenbrock function requires 2D input")
        
        x1, x2 = x[0], x[1]
        a, b = self.a, self.b
        
        df_dx1 = -2*(a - x1) - 4*b*x1*(x2 - x1**2)
        df_dx2 = 2*b*(x2 - x1**2)
        
        return np.array([df_dx1, df_dx2])

    def hessian(self, x):
        """
        Evaluate Hessian matrix:
        [[∂²f/∂x₁², ∂²f/∂x₁∂x₂],
         [∂²f/∂x₁∂x₂, ∂²f/∂x₂²]]
        """
        x = np.array(x)
        if len(x) != 2:
            raise ValueError("Rosenbrock function requires 2D input")
        
        x1, x2 = x[0], x[1]
        b = self.b
        
        d2f_dx1x1 = 2 - 4*b*(x2 - x1**2) + 8*b*x1**2
        d2f_dx1x2 = -4*b*x1
        d2f_dx2x2 = 2*b
        
        return np.array([[d2f_dx1x1, d2f_dx1x2],
                         [d2f_dx1x2, d2f_dx2x2]])


class SmoothedCornerTrianglesFunction(ObjectiveFunction):
    """
    Implements the function:
        f(x) = exp(x₁ + 3x₂ - 0.1) + exp(x₁ - 3x₂ - 0.1) + exp(-x₁ - 0.1)
    As seen in Boyd, p. 470, example 9.20.
    
    This is a convex function that approximates a piecewise linear function.
    """

    def func(self, x):
        """
        Compute f(x) = exp(x₁ + 3x₂ - 0.1) + exp(x₁ - 3x₂ - 0.1) + exp(-x₁ - 0.1)
        """
        x = np.array(x)
        if len(x) != 2:
            raise ValueError("SmoothedCornerTriangles function requires 2D input")
        
        x1, x2 = x[0], x[1]
        term1 = np.exp(x1 + 3*x2 - 0.1)
        term2 = np.exp(x1 - 3*x2 - 0.1)
        term3 = np.exp(-x1 - 0.1)
        
        return float(term1 + term2 + term3)

    def grad(self, x):
        """
        Gradient: [∂f/∂x₁, ∂f/∂x₂]
        """
        x = np.array(x)
        if len(x) != 2:
            raise ValueError("SmoothedCornerTriangles function requires 2D input")
        
        x1, x2 = x[0], x[1]
        term1 = np.exp(x1 + 3*x2 - 0.1)
        term2 = np.exp(x1 - 3*x2 - 0.1)
        term3 = np.exp(-x1 - 0.1)

        df_dx1 = term1 + term2 - term3
        df_dx2 = 3*term1 - 3*term2
        
        return np.array([df_dx1, df_dx2])

    def hessian(self, x):
        """
        Hessian matrix:
        [[∂²f/∂x₁², ∂²f/∂x₁∂x₂],
         [∂²f/∂x₂∂x₁, ∂²f/∂x₂²]]
        """
        x = np.array(x)
        if len(x) != 2:
            raise ValueError("SmoothedCornerTriangles function requires 2D input")
        
        x1, x2 = x[0], x[1]
        term1 = np.exp(x1 + 3*x2 - 0.1)
        term2 = np.exp(x1 - 3*x2 - 0.1)
        term3 = np.exp(-x1 - 0.1)

        d2f_dx1x1 = term1 + term2 + term3
        d2f_dx1x2 = 3*term1 - 3*term2
        d2f_dx2x2 = 9*term1 + 9*term2

        return np.array([[d2f_dx1x1, d2f_dx1x2],
                         [d2f_dx1x2, d2f_dx2x2]])



# Example usage and testing
if __name__ == "__main__":
    # Test all functions
    x_test = np.array([0.5, -0.3])
    
    functions = [
        ("Quadratic", QuadraticFunction(np.array([[2, 1], [1, 2]]))),
        ("Linear", LinearFunction(np.array([1, -2]))),
        ("Rosenbrock", RosenbrockFunction()),
        ("SmoothedCornerTriangles", SmoothedCornerTrianglesFunction())
    ]
    
    for name, func in functions:
        print(f"\n{name} Function at x = {x_test}:")
        print(f"  f(x) = {func.func(x_test):.6f}")
        print(f"  ∇f(x) = {func.grad(x_test)}")
        print(f"  ∇²f(x) = \n{func.hessian(x_test)}")