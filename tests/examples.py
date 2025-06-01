
import numpy as np

def quadratic_circles(x, hessian_needed=False):
    """
    Quadratic function f(x) = x^T * Q * x where Q = [[1, 0], [0, 1]]
    Contour lines are circles.
    """
    x = np.array(x, dtype=float)
    Q = np.array([[1, 0], [0, 1]])
    
    # Function value: f(x) = x^T * Q * x
    f = float(x.T @ Q @ x)
    
    # Gradient: g(x) = 2 * Q * x
    g = 2 * Q @ x
    
    # Hessian: H(x) = 2 * Q (only if needed)
    h = 2 * Q if hessian_needed else None
    
    return f, g, h


def quadratic_axis_aligned(x, hessian_needed=False):
    """
    Quadratic function f(x) = x^T * Q * x where Q = [[1, 0], [0, 100]]
    Contour lines are axis-aligned ellipses.
    """
    x = np.array(x, dtype=float)
    Q = np.array([[1, 0], [0, 100]])
    
    # Function value: f(x) = x^T * Q * x
    f = float(x.T @ Q @ x)
    
    # Gradient: g(x) = 2 * Q * x
    g = 2 * Q @ x
    
    # Hessian: H(x) = 2 * Q (only if needed)
    h = 2 * Q if hessian_needed else None
    
    return f, g, h


def quadratic_rotated(x, hessian_needed=False):
    """
    Quadratic function f(x) = x^T * Q * x where Q = R^T * D * R
    R = [[√3/2, -0.5], [0.5, √3/2]] (rotation matrix)
    D = [[100, 0], [0, 1]] (diagonal matrix)
    Contour lines are rotated ellipses.
    """
    x = np.array(x, dtype=float)
    
    # Rotation matrix
    sqrt3_2 = np.sqrt(3) / 2
    R = np.array([[sqrt3_2, -0.5],
                  [0.5, sqrt3_2]])
    
    # Diagonal matrix
    D = np.array([[100, 0],
                  [0, 1]])
    
    # Q = R^T * D * R
    Q = R.T @ D @ R
    
    # Function value: f(x) = x^T * Q * x
    f = float(x.T @ Q @ x)
    
    # Gradient: g(x) = 2 * Q * x
    g = 2 * Q @ x
    
    # Hessian: H(x) = 2 * Q (only if needed)
    h = 2 * Q if hessian_needed else None
    
    return f, g, h


def rosenbrock(x, hessian_needed=False):
    """
    Rosenbrock function: f(x) = 100(x₂ - x₁²)² + (1 - x₁)²
    Contour lines are banana-shaped ellipses. This is NOT a convex function.
    """
    x = np.array(x, dtype=float)
    if len(x) != 2:
        raise ValueError("Rosenbrock function requires 2D input")
    
    x1, x2 = x[0], x[1]
    
    # Function value: f(x) = 100(x₂ - x₁²)² + (1 - x₁)²
    f = float(100 * (x2 - x1**2)**2 + (1 - x1)**2)
    
    # Gradient: 
    # ∂f/∂x₁ = -400x₁(x₂ - x₁²) - 2(1 - x₁)
    # ∂f/∂x₂ = 200(x₂ - x₁²)
    df_dx1 = -400*x1*(x2 - x1**2) - 2*(1 - x1)
    df_dx2 = 200*(x2 - x1**2)
    g = np.array([df_dx1, df_dx2])
    
    # Hessian (only if needed):
    # ∂²f/∂x₁² = -400(x₂ - x₁²) + 800x₁² + 2
    # ∂²f/∂x₁∂x₂ = -400x₁
    # ∂²f/∂x₂² = 200
    if hessian_needed:
        d2f_dx1x1 = -400*(x2 - x1**2) + 800*x1**2 + 2
        d2f_dx1x2 = -400*x1
        d2f_dx2x2 = 200
        h = np.array([[d2f_dx1x1, d2f_dx1x2],
                      [d2f_dx1x2, d2f_dx2x2]])
    else:
        h = None
    
    return f, g, h


def linear(x, hessian_needed=False):
    """
    Linear function: f(x) = a^T * x for some nonzero vector a.
    Contour lines are straight lines.
    """
    x = np.array(x, dtype=float)
    a = np.array([1.0, -2.0])  # Chosen nonzero vector
    
    # Function value: f(x) = a^T * x
    f = float(np.dot(a, x))
    
    # Gradient: g(x) = a (constant)
    g = a.copy()
    
    # Hessian: H(x) = 0 (zero matrix, only if needed)
    h = np.zeros((len(x), len(x))) if hessian_needed else None
    
    return f, g, h


def smoothed_corner_triangles(x, hessian_needed=False):
    """
    Function: f(x₁, x₂) = exp(x₁ + 3x₂ - 0.1) + exp(x₁ - 3x₂ - 0.1) + exp(-x₁ - 0.1)
    From Boyd's book, p. 470, example 9.20.
    Contour lines look like smoothed corner triangles.
    """
    x = np.array(x, dtype=float)
    if len(x) != 2:
        raise ValueError("Smoothed corner triangles function requires 2D input")
    
    x1, x2 = x[0], x[1]
    
    # Compute exponential terms
    term1 = np.exp(x1 + 3*x2 - 0.1)
    term2 = np.exp(x1 - 3*x2 - 0.1)
    term3 = np.exp(-x1 - 0.1)
    
    # Function value
    f = float(term1 + term2 + term3)
    
    # Gradient:
    # ∂f/∂x₁ = term1 + term2 - term3
    # ∂f/∂x₂ = 3*term1 - 3*term2
    df_dx1 = term1 + term2 - term3
    df_dx2 = 3*term1 - 3*term2
    g = np.array([df_dx1, df_dx2])
    
    # Hessian (only if needed):
    # ∂²f/∂x₁² = term1 + term2 + term3
    # ∂²f/∂x₁∂x₂ = 3*term1 - 3*term2
    # ∂²f/∂x₂² = 9*term1 + 9*term2
    if hessian_needed:
        d2f_dx1x1 = term1 + term2 + term3
        d2f_dx1x2 = 3*term1 - 3*term2
        d2f_dx2x2 = 9*term1 + 9*term2
        h = np.array([[d2f_dx1x1, d2f_dx1x2],
                      [d2f_dx1x2, d2f_dx2x2]])
    else:
        h = None
    
    return f, g, h


def get_assignment_functions():
    """
    Returns all functions required by the assignment with their metadata.
    """
    functions = {
        'Quadratic Circles': {
            'func': quadratic_circles,
            'x0': np.array([1.0, 1.0]),
            'title': 'Quadratic Function - Circles (Q = I)'
        },
        'Quadratic Axis-Aligned': {
            'func': quadratic_axis_aligned, 
            'x0': np.array([1.0, 1.0]),
            'title': 'Quadratic Function - Axis-Aligned Ellipses'
        },
        'Quadratic Rotated': {
            'func': quadratic_rotated,
            'x0': np.array([1.0, 1.0]), 
            'title': 'Quadratic Function - Rotated Ellipses'
        },
        'Rosenbrock': {
            'func': rosenbrock,
            'x0': np.array([-1.0, 2.0]),  # Special starting point for Rosenbrock
            'title': 'Rosenbrock Function'
        },
        'Linear': {
            'func': linear,
            'x0': np.array([1.0, 1.0]),
            'title': 'Linear Function'
        },
        'Smoothed Corner Triangles': {
            'func': smoothed_corner_triangles,
            'x0': np.array([1.0, 1.0]),
            'title': 'Smoothed Corner Triangles'
        }
    }
    return functions


    # Test all functions
    x_test = np.array([0.5, -0.3])
    
    functions = get_assignment_functions()
    
    print("Testing all assignment functions:")
    print("=" * 50)
    
    for name, func_data in functions.items():
        func = func_data['func']
        print(f"\n{name}:")
        try:
            # Test without Hessian
            f, g, h = func(x_test, hessian_needed=False)
            print(f"  f(x) = {f:.6f}")
            print(f"  ∇f(x) = {g}")
            print(f"  Hessian computed: {h is not None}")
            
            # Test with Hessian
            f2, g2, h2 = func(x_test, hessian_needed=True)
            print(f"  With Hessian requested:")
            print(f"    Hessian shape: {h2.shape if h2 is not None else 'None'}")
            
        except Exception as e:
            print(f"  Error: {e}")
    
    print(f"\n{'='*50}")
    print("All functions tested successfully!")
    print("Interface: func(x, hessian_needed) -> (f, g, h)")