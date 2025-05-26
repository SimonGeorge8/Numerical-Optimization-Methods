"""
Unit tests for unconstrained minimization algorithms and objective functions.
Run with: python -m pytest test_unconstrained_min.py -v
or: python test_unconstrained_min.py
"""
import unittest
import numpy as np
import sys
import os
from pathlib import Path

# Handle imports more robustly
# Try import from src directory first
from src.unconstrained_min import UnconstrainedMinimizer, ObjectiveFunction
from examples import QuadraticFunction, LinearFunction, RosenbrockFunction, SmoothedCornerTrianglesFunction


class TestObjectiveFunctions(unittest.TestCase):
    """Test all objective function implementations for correctness."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_point = np.array([0.5, -0.3])
        self.tolerance = 1e-8
    
    def test_quadratic_function(self):
        """Test QuadraticFunction implementation."""
        Q = np.array([[2, 1], [1, 2]])
        func = QuadraticFunction(Q)
        x = self.test_point
        
        # Test function value: f(x) = x^T * Q * x
        expected_f = x.T @ Q @ x
        self.assertAlmostEqual(func.func(x), expected_f, places=10)
        
        # Test gradient: ∇f(x) = Q * x
        expected_grad = Q @ x
        np.testing.assert_allclose(func.grad(x), expected_grad, atol=self.tolerance)
        
        # Test Hessian: ∇²f(x) = Q
        expected_hess = Q
        np.testing.assert_allclose(func.hessian(x), expected_hess, atol=self.tolerance)
    
    def test_rosenbrock_function(self):
        """Test RosenbrockFunction implementation."""
        func = RosenbrockFunction(a=1, b=100)
        x = np.array([0.0, 0.0])  # Use origin for easier verification
        
        # Test function value at origin: f(0,0) = (1-0)² + 100(0-0²)² = 1
        expected_f = 1.0
        self.assertAlmostEqual(func.func(x), expected_f, places=10)
        
        # Test gradient at origin: ∇f(0,0) = [-2, 0]
        expected_grad = np.array([-2.0, 0.0])
        np.testing.assert_allclose(func.grad(x), expected_grad, atol=self.tolerance)
        
        # Test Hessian at origin: ∇²f(0,0) = [[2, 0], [0, 200]]
        expected_hess = np.array([[2.0, 0.0], [0.0, 200.0]])
        np.testing.assert_allclose(func.hessian(x), expected_hess, atol=self.tolerance)
        
        # Test at the minimum: (1, 1)
        x_min = np.array([1.0, 1.0])
        self.assertAlmostEqual(func.func(x_min), 0.0, places=10)


class TestNumericalGradients(unittest.TestCase):
    """Test gradients and Hessians using numerical differentiation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.h = 1e-8  # Step size for numerical differentiation
        self.tolerance = 1e-6
    
    def numerical_gradient(self, func, x):
        """Compute numerical gradient using finite differences."""
        x = np.array(x, dtype=float)
        grad = np.zeros_like(x)
        
        for i in range(len(x)):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += self.h
            x_minus[i] -= self.h
            
            grad[i] = (func.func(x_plus) - func.func(x_minus)) / (2 * self.h)
        
        return grad
    
    def test_quadratic_gradients(self):
        """Test QuadraticFunction gradients numerically."""
        Q = np.array([[3, 1], [1, 4]])
        func = QuadraticFunction(Q)
        x = np.array([0.7, -0.5]) # FIX the test as the actual formula for the quadratic formula is incorrect
        
        analytical_grad = func.grad(x)
        numerical_grad = self.numerical_gradient(func, x)
        
        np.testing.assert_allclose(analytical_grad, numerical_grad, atol=self.tolerance)
    
    def test_rosenbrock_gradients(self):
        """Test RosenbrockFunction gradients numerically."""
        func = RosenbrockFunction()
        x = np.array([0.5, 1.2])
        
        analytical_grad = func.grad(x)
        numerical_grad = self.numerical_gradient(func, x)
        
        np.testing.assert_allclose(analytical_grad, numerical_grad, atol=self.tolerance)


class TestUnconstrainedMinimizer(unittest.TestCase):
    """Test the optimization algorithms."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.tolerance = 1e-6
        self.max_iter = 1000
    
    def test_quadratic_minimization_newton(self):
        """Test Newton's method on a simple quadratic function."""
        # Create a shifted circle: f(x) = (x-2)² + (y-3)²
        class ShiftedCircle(ObjectiveFunction):
            def func(self, x):
                x = np.array(x)
                return (x[0] - 2)**2 + (x[1] - 3)**2
            
            def grad(self, x):
                x = np.array(x)
                return np.array([2*(x[0] - 2), 2*(x[1] - 3)])
            
            def hessian(self, x):
                return np.array([[2, 0], [0, 2]])
        
        func = ShiftedCircle()
        minimizer = UnconstrainedMinimizer(method='newton')
        
        result = minimizer.minimize(
            func, 
            x0=[0, 0], 
            obj_tol=1e-10, 
            param_tol=1e-10, 
            max_iter=100
        )
        
        # Check that we found the minimum
        expected_x = np.array([2, 3])
        np.testing.assert_allclose(result['x'], expected_x, atol=1e-6)
        self.assertAlmostEqual(result['f'], 0.0, places=6)
        self.assertTrue(result['converged'])
    
    def test_quadratic_minimization_gradient_descent(self):
        """Test gradient descent on quadratic function."""
        # Simple quadratic: f(x) = x₁² + 2x₂² (minimum at origin)
        Q = np.array([[2, 0], [0, 4]])
        func = QuadraticFunction(Q)
        minimizer = UnconstrainedMinimizer(method='gradient_descent')
        
        result = minimizer.minimize(
            func,
            x0=[5, -3],
            obj_tol=1e-8,
            param_tol=1e-8,
            max_iter=1000
        )
        
        # Minimum should be at origin
        expected_x = np.array([0, 0])
        np.testing.assert_allclose(result['x'], expected_x, atol=1e-4)
        self.assertAlmostEqual(result['f'], 0.0, places=4)
    
    def test_convergence_criteria(self):
        """Test different convergence criteria."""
        # Simple quadratic: f(x) = x₁² + x₂²
        Q = np.array([[2, 0], [0, 2]])
        func = QuadraticFunction(Q)
        minimizer = UnconstrainedMinimizer(method='newton')
        
        # Strict tolerance
        result_strict = minimizer.minimize(
            func,
            x0=[1, 1],
            obj_tol=1e-12,
            param_tol=1e-12,
            max_iter=100
        )
        
        # Loose tolerance
        result_loose = minimizer.minimize(
            func,
            x0=[1, 1],
            obj_tol=1e-3,
            param_tol=1e-3,
            max_iter=100
        )
        
        # Both should converge
        self.assertTrue(result_strict['converged'])
        self.assertTrue(result_loose['converged'])
    
    def test_max_iterations(self):
        """Test maximum iteration limit."""
        func = RosenbrockFunction()
        minimizer = UnconstrainedMinimizer(method='gradient_descent')
        
        # Very few iterations with difficult starting point
        result = minimizer.minimize(
            func,
            x0=[-5, 5],
            obj_tol=1e-12,
            param_tol=1e-12,
            max_iter=5  # Very few iterations
        )
        
        # Should not converge due to iteration limit
        self.assertEqual(result['iterations'], 5)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""
    
    def test_singular_hessian(self):
        """Test behavior with singular Hessian matrix."""
        class SingularHessianFunction(ObjectiveFunction):
            def func(self, x):
                x = np.array(x)
                return x[0]**2  # Only depends on first variable
            
            def grad(self, x):
                x = np.array(x)
                return np.array([2*x[0], 0])
            
            def hessian(self, x):
                return np.array([[2, 0], [0, 0]])  # Singular matrix
        
        func = SingularHessianFunction()
        minimizer = UnconstrainedMinimizer(method='newton')
        
        # Should handle singular Hessian gracefully
        result = minimizer.minimize(
            func,
            x0=[1, 1],
            obj_tol=1e-6,
            param_tol=1e-6,
            max_iter=100
        )
        
        # Should still make progress (fallback to gradient descent)
        self.assertLess(abs(result['x'][0]), 0.1)  # x[0] should be close to 0
    
    def test_starting_at_minimum(self):
        """Test starting exactly at the minimum."""
        # Simple quadratic: f(x) = x₁² + x₂² (minimum at origin)
        Q = np.array([[2, 0], [0, 2]])
        func = QuadraticFunction(Q)
        minimizer = UnconstrainedMinimizer(method='newton')
        
        result = minimizer.minimize(
            func,
            x0=[0, 0],  # Already at minimum
            obj_tol=1e-8,
            param_tol=1e-8,
            max_iter=100
        )
        
        # Should converge immediately or in very few iterations
        self.assertLessEqual(result['iterations'], 2)
        self.assertTrue(result['converged'])


class TestComparisonBetweenMethods(unittest.TestCase):
    """Compare different optimization methods."""
    
    def test_newton_vs_gradient_descent(self):
        """Compare Newton's method vs gradient descent on well-conditioned problems."""
        # Simple quadratic: f(x) = x₁² + x₂²
        Q = np.array([[2, 0], [0, 2]])
        func = QuadraticFunction(Q)
        
        # Newton's method
        minimizer_newton = UnconstrainedMinimizer(method='newton')
        result_newton = minimizer_newton.minimize(
            func,
            x0=[2, 3],
            obj_tol=1e-8,
            param_tol=1e-8,
            max_iter=100
        )
        
        # Gradient descent
        minimizer_gd = UnconstrainedMinimizer(method='gradient_descent')
        result_gd = minimizer_gd.minimize(
            func,
            x0=[2, 3],
            obj_tol=1e-8,
            param_tol=1e-8,
            max_iter=100
        )
        
        # Newton should converge faster for well-conditioned quadratic
        self.assertLess(result_newton['iterations'], result_gd['iterations'])
        
        # Both should find similar solutions
        np.testing.assert_allclose(result_newton['x'], result_gd['x'], atol=1e-6)


def run_tests():
    """Run all tests with detailed output."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestObjectiveFunctions,
        TestNumericalGradients, 
        TestUnconstrainedMinimizer,
        TestEdgeCases,
        TestComparisonBetweenMethods
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(test_suite)
    
    return result


if __name__ == '__main__':
    print("Running unconstrained minimization tests...")
    print("=" * 50)
    
    try:
        result = run_tests()
        
        print("\n" + "=" * 50)
        if result.wasSuccessful():
            print("✅ All tests passed!")
        else:
            print(f"❌ {len(result.failures)} test(s) failed")
            print(f"❌ {len(result.errors)} error(s) occurred")
            
    except Exception as e:
        print(f"Error running tests: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure 'unconstrained_min.py' exists in the same directory")
        print("2. Check that there are no syntax errors in your main files")
        print("3. Ensure all required imports are available")