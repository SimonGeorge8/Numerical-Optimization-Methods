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
try:
    from src.unconstrained_min import UnconstrainedMinimizer, ObjectiveFunction, create_function_wrapper
    from examples import (quadratic_circles, quadratic_axis_aligned, quadratic_rotated, 
                         rosenbrock, linear, smoothed_corner_triangles, get_assignment_functions)
    from src.utils import plot_contour, plot_function_values, print_optimization_summary
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all required files exist in the correct directories")
    sys.exit(1)


class TestAssignmentFunctions(unittest.TestCase):
    """Test all assignment functions with the correct interface."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_point = np.array([0.5, -0.3])
        self.tolerance = 1e-8
    
    def test_function_interface(self):
        """Test that all functions follow the required interface: func(x, hessian_needed) -> (f, g, h)"""
        functions = get_assignment_functions()
        
        for name, func_data in functions.items():
            with self.subTest(function=name):
                func = func_data['func']
                x = self.test_point
                
                # Test without Hessian
                f, g, h = func(x, hessian_needed=False)
                
                # Check return types
                self.assertIsInstance(f, float)
                self.assertIsInstance(g, np.ndarray)
                self.assertIsNone(h)  # Should be None when hessian_needed=False
                
                # Test with Hessian
                f2, g2, h2 = func(x, hessian_needed=True)
                
                # Check that function and gradient are the same
                self.assertAlmostEqual(f, f2, places=10)
                np.testing.assert_allclose(g, g2, atol=self.tolerance)
                
                # Check that Hessian is returned when requested
                self.assertIsNotNone(h2)
                self.assertIsInstance(h2, np.ndarray)
                self.assertEqual(h2.shape, (2, 2))
    
    def test_quadratic_circles(self):
        """Test the quadratic circles function."""
        x = np.array([1.0, 1.0])
        f, g, h = quadratic_circles(x, hessian_needed=True)
        
        # For Q = I, f(x) = x^T * x = 1 + 1 = 2
        expected_f = 2.0
        self.assertAlmostEqual(f, expected_f, places=10)
        
        # Gradient should be 2*x = [2, 2]
        expected_g = np.array([2.0, 2.0])
        np.testing.assert_allclose(g, expected_g, atol=self.tolerance)
        
        # Hessian should be 2*I
        expected_h = 2 * np.eye(2)
        np.testing.assert_allclose(h, expected_h, atol=self.tolerance)
    
    def test_quadratic_axis_aligned(self):
        """Test the axis-aligned ellipses function."""
        x = np.array([1.0, 1.0])
        f, g, h = quadratic_axis_aligned(x, hessian_needed=True)
        
        # For Q = [[1, 0], [0, 100]], f([1,1]) = 1 + 100 = 101
        expected_f = 101.0
        self.assertAlmostEqual(f, expected_f, places=10)
        
        # Gradient should be 2*Q*x = [2, 200]
        expected_g = np.array([2.0, 200.0])
        np.testing.assert_allclose(g, expected_g, atol=self.tolerance)
    
    def test_rosenbrock_function(self):
        """Test Rosenbrock function."""
        # Test at origin: f(0,0) = 100(0-0)² + (1-0)² = 1
        x = np.array([0.0, 0.0])
        f, g, h = rosenbrock(x, hessian_needed=True)
        
        expected_f = 1.0
        self.assertAlmostEqual(f, expected_f, places=10)
        
        # Test gradient at origin: ∇f(0,0) = [-2, 0]
        expected_g = np.array([-2.0, 0.0])
        np.testing.assert_allclose(g, expected_g, atol=self.tolerance)
        
        # Test at the minimum: (1, 1)
        x_min = np.array([1.0, 1.0])
        f_min, g_min, _ = rosenbrock(x_min, hessian_needed=False)
        self.assertAlmostEqual(f_min, 0.0, places=10)
        
        # Gradient at minimum should be zero
        np.testing.assert_allclose(g_min, np.array([0.0, 0.0]), atol=self.tolerance)
    
    def test_linear_function(self):
        """Test linear function."""
        x = self.test_point
        f, g, h = linear(x, hessian_needed=True)
        
        # For a = [1, -2], f(x) = 1*0.5 + (-2)*(-0.3) = 0.5 + 0.6 = 1.1
        expected_f = 1.0 * 0.5 + (-2.0) * (-0.3)
        self.assertAlmostEqual(f, expected_f, places=10)
        
        # Gradient should be constant = a = [1, -2]
        expected_g = np.array([1.0, -2.0])
        np.testing.assert_allclose(g, expected_g, atol=self.tolerance)
        
        # Hessian should be zero matrix
        expected_h = np.zeros((2, 2))
        np.testing.assert_allclose(h, expected_h, atol=self.tolerance)
    
    def test_smoothed_corner_triangles(self):
        """Test smoothed corner triangles function."""
        x = np.array([0.0, 0.0])
        f, g, h = smoothed_corner_triangles(x, hessian_needed=True)
        
        # Test that function runs without error
        self.assertIsInstance(f, float)
        self.assertFalse(np.isnan(f))
        
        # Test gradient
        self.assertEqual(len(g), 2)
        self.assertFalse(np.any(np.isnan(g)))
        
        # Test Hessian
        self.assertEqual(h.shape, (2, 2))
        self.assertFalse(np.any(np.isnan(h)))


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
            
            f_plus, _, _ = func(x_plus, hessian_needed=False)
            f_minus, _, _ = func(x_minus, hessian_needed=False)
            
            grad[i] = (f_plus - f_minus) / (2 * self.h)
        
        return grad
    
    def test_quadratic_gradients(self):
        """Test quadratic function gradients numerically."""
        x = np.array([0.7, -0.5])
        
        analytical_f, analytical_grad, _ = quadratic_circles(x, hessian_needed=False)
        numerical_grad = self.numerical_gradient(quadratic_circles, x)
        
        np.testing.assert_allclose(analytical_grad, numerical_grad, atol=self.tolerance)
    
    def test_rosenbrock_gradients(self):
        """Test Rosenbrock function gradients numerically."""
        x = np.array([0.5, 1.2])
        
        analytical_f, analytical_grad, _ = rosenbrock(x, hessian_needed=False)
        numerical_grad = self.numerical_gradient(rosenbrock, x)
        
        np.testing.assert_allclose(analytical_grad, numerical_grad, atol=self.tolerance)


class TestUnconstrainedMinimizer(unittest.TestCase):
    """Test the optimization algorithms."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.tolerance = 1e-6
        self.max_iter = 1000
    
    def test_quadratic_minimization_newton(self):
        """Test Newton's method on a simple quadratic function."""
        minimizer = UnconstrainedMinimizer(method='newton')
        
        result = minimizer.minimize(
            quadratic_circles, 
            x0=[2, 3], 
            obj_tol=1e-10, 
            param_tol=1e-10, 
            max_iter=100
        )
        
        # Check that we found the minimum (should be at origin for this function)
        expected_x = np.array([0, 0])
        np.testing.assert_allclose(result['x'], expected_x, atol=1e-6)
        self.assertAlmostEqual(result['f'], 0.0, places=6)
        self.assertTrue(result['converged'])
    
    def test_quadratic_minimization_gradient_descent(self):
        """Test gradient descent on quadratic function."""
        minimizer = UnconstrainedMinimizer(method='gradient_descent')
        
        result = minimizer.minimize(
            quadratic_circles,
            x0=[1, 1],
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
        minimizer = UnconstrainedMinimizer(method='newton')
        
        # Strict tolerance
        result_strict = minimizer.minimize(
            quadratic_circles,
            x0=[1, 1],
            obj_tol=1e-12,
            param_tol=1e-12,
            max_iter=100
        )
        
        # Loose tolerance
        result_loose = minimizer.minimize(
            quadratic_circles,
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
        minimizer = UnconstrainedMinimizer(method='gradient_descent')
        
        # Very few iterations with difficult starting point
        result = minimizer.minimize(
            rosenbrock,
            x0=[-5, 5],
            obj_tol=1e-12,
            param_tol=1e-12,
            max_iter=5  # Very few iterations
        )
        
        # Should not converge due to iteration limit
        self.assertEqual(result['iterations'], 5)
        self.assertFalse(result['converged'])


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""
    
    def test_singular_hessian(self):
        """Test behavior with singular Hessian matrix."""
        def singular_hessian_func(x, hessian_needed=False):
            x = np.array(x)
            f = x[0]**2  # Only depends on first variable
            g = np.array([2*x[0], 0])
            h = np.array([[2, 0], [0, 0]]) if hessian_needed else None  # Singular matrix
            return f, g, h
        
        minimizer = UnconstrainedMinimizer(method='newton')
        
        # Should handle singular Hessian gracefully
        result = minimizer.minimize(
            singular_hessian_func,
            x0=[1, 1],
            obj_tol=1e-6,
            param_tol=1e-6,
            max_iter=100
        )
        
        # Should still make progress
        self.assertLess(abs(result['x'][0]), 0.1)  # x[0] should be close to 0
    
    def test_starting_at_minimum(self):
        """Test starting exactly at the minimum."""
        minimizer = UnconstrainedMinimizer(method='newton')
        
        result = minimizer.minimize(
            quadratic_circles,
            x0=[0, 0],  # Already at minimum
            obj_tol=1e-8,
            param_tol=1e-8,
            max_iter=100
        )
        
        # Should converge immediately or in very few iterations
        self.assertLessEqual(result['iterations'], 3)
        self.assertTrue(result['converged'])


class TestComparisonBetweenMethods(unittest.TestCase):
    """Compare different optimization methods."""
    
    def test_newton_vs_gradient_descent(self):
        """Compare Newton's method vs gradient descent on well-conditioned problems."""
        
        # Use moderately ill-conditioned quadratic to show Newton's advantage
        minimizer_newton = UnconstrainedMinimizer(method='newton')
        result_newton = minimizer_newton.minimize(
            quadratic_axis_aligned,  # Condition number 100 - challenging for GD
            x0=[1, 1],
            obj_tol=1e-8,
            param_tol=1e-6,
            max_iter=50
        )
        
        # Gradient descent - will struggle with ill-conditioning
        minimizer_gd = UnconstrainedMinimizer(method='gradient_descent')
        result_gd = minimizer_gd.minimize(
            quadratic_axis_aligned,
            x0=[1, 1],
            obj_tol=1e-8,
            param_tol=1e-6,
            max_iter=50  # Same iteration budget
        )
        
        # Key test: Newton should converge much faster (or GD may not converge at all)
        # This demonstrates Newton's superiority on ill-conditioned problems
        if result_newton['converged'] and result_gd['converged']:
            # If both converged, Newton should be faster
            self.assertLess(result_newton['iterations'], result_gd['iterations'])
            np.testing.assert_allclose(result_newton['x'], result_gd['x'], atol=1e-4)
        elif result_newton['converged'] and not result_gd['converged']:
            # This is the expected outcome: Newton succeeds, GD struggles
            # This demonstrates Newton's advantage on ill-conditioned problems
            self.assertTrue(True, "Newton converged while GD didn't - demonstrates Newton's advantage")
        else:
            # Newton should at least converge on this problem
            self.assertTrue(result_newton['converged'], "Newton should converge on quadratic problems")
        
        # Newton should definitely converge for quadratic problems
        self.assertTrue(result_newton['converged'])
        
        # Newton should achieve better objective value within same iteration budget
        self.assertLess(result_newton['f'], result_gd['f'])


def run_all_assignment_tests():
    """Run optimization tests on all assignment functions."""
    print("Running Assignment Tests with Correct Interface")
    print("=" * 60)
    
    functions = get_assignment_functions()
    
    # Assignment parameters
    obj_tol = 1e-12
    param_tol = 1e-8
    
    for name, func_data in functions.items():
        print(f"\nTesting {name}:")
        print("-" * 40)
        
        func = func_data['func']
        x0 = func_data['x0']
        
        # Set max_iter based on function and method
        if name == 'Rosenbrock':
            max_iter_gd = 10000  # Special case for Rosenbrock with GD
            max_iter_newton = 100
        else:
            max_iter_gd = 100
            max_iter_newton = 100
        
        # Test Gradient Descent
        print(f"Gradient Descent:")
        minimizer_gd = UnconstrainedMinimizer(method='gradient_descent')
        result_gd = minimizer_gd.minimize(func, x0, obj_tol, param_tol, max_iter_gd)
        print(f"  Final: x={result_gd['x']}, f={result_gd['f']:.8e}, converged={result_gd['converged']}")
        
        # Test Newton's Method  
        print(f"Newton's Method:")
        minimizer_newton = UnconstrainedMinimizer(method='newton')
        result_newton = minimizer_newton.minimize(func, x0, obj_tol, param_tol, max_iter_newton)
        print(f"  Final: x={result_newton['x']}, f={result_newton['f']:.8e}, converged={result_newton['converged']}")


def run_tests():
    """Run all tests with detailed output."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestAssignmentFunctions,
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
            print("\nRunning assignment-specific tests...")
            run_all_assignment_tests()
        else:
            print(f"❌ {len(result.failures)} test(s) failed")
            print(f"❌ {len(result.errors)} error(s) occurred")
            
            if result.failures:
                print("\nFailures:")
                for test, traceback in result.failures:
                    print(f"  {test}: {traceback}")
            
            if result.errors:
                print("\nErrors:")
                for test, traceback in result.errors:
                    print(f"  {test}: {traceback}")
            
    except Exception as e:
        print(f"Error running tests: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure all files exist in the correct directories")
        print("2. Check that there are no syntax errors in your main files")
        print("3. Ensure all required imports are available")