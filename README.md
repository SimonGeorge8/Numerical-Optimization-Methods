# Numerical Optimization Project

A Python implementation of unconstrained optimization algorithms including gradient descent and Newton's method with exact analytic derivatives.

## Project Structure

```
numerical_optimization/
├── README.md
├── requirements.txt
├── src/
│   ├── unconstrained_min.py    # Main optimization algorithms
│   └── utils.py                # Plotting and utility functions
└── tests/
    ├── test_unconstrained_min.py    # Unit tests
    └── examples.py                  # Example usage and test functions
```

## Features

- **Unconstrained Optimization Methods:**
  - Gradient Descent with line search
  - Newton's Method with exact Hessian
  
- **Exact Analytic Derivatives:**
  - All objective functions implement exact mathematical formulas
  - No numerical approximations for gradients or Hessians
  - Hand-derived derivatives for maximum accuracy

- **Visualization Tools:**
  - Contour plots of objective functions
  - Algorithm path visualization with legends
  - Convergence curve comparisons
  - Function value tracking across iterations

- **Convergence Criteria:**
  - Objective function tolerance (`obj_tol`)
  - Parameter tolerance (`param_tol`) 
  - Maximum iteration limits (`max_iter`)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd numerical_optimization
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Requirements

- Python 3.7+
- NumPy
- Matplotlib
- (Add other dependencies as needed)

## Usage

### Basic Optimization

```python
from src.unconstrained_min import UnconstrainedMinimizer
from src.examples import QuadraticFunction

# Create an objective function with analytic derivatives
f = QuadraticFunction(Q=[[2, 0], [0, 2]], b=[0, 0], c=0)

# Initialize optimizer
optimizer = UnconstrainedMinimizer(method='gradient_descent')

# Run optimization
result = optimizer.minimize(
    f=f,
    x0=[1.0, 1.0],          # Starting point
    obj_tol=1e-12,          # Objective tolerance
    param_tol=1e-8,         # Parameter tolerance  
    max_iter=100            # Maximum iterations
)

print(f"Optimal point: {result['x']}")
print(f"Optimal value: {result['f']}")
print(f"Converged: {result['converged']}")
```

### Visualization

```python
from src.utils import plot_contour, plot_function_values

# Plot contour with optimization path
plot_contour(
    objective_func=f,
    x_limits=(-2, 2),
    y_limits=(-2, 2),
    algorithm_paths=[result['path']],
    path_names=['Gradient Descent'],
    title="Optimization Path"
)

# Plot convergence curve
plot_function_values(
    iteration_data=[result['f_values']],
    method_names=['Gradient Descent'],
    title="Convergence Analysis"
)
```

## Implementing Custom Objective Functions

All objective functions must inherit from `ObjectiveFunction` and implement exact analytic derivatives:

```python
from src.unconstrained_min import ObjectiveFunction
import numpy as np

class MyFunction(ObjectiveFunction):
    def func(self, x):
        """Implement f(x) - derive on paper first!"""
        # TODO: Your function implementation
        pass
    
    def grad(self, x):
        """Implement ∇f(x) - exact analytic gradient"""
        # TODO: Your gradient implementation (derived by hand)
        pass
    
    def hessian(self, x):
        """Implement ∇²f(x) - exact analytic Hessian"""
        # TODO: Your Hessian implementation (derived by hand)
        pass
```

### Important Notes for Objective Functions

1. **Derive derivatives by hand** - Compute gradients and Hessians analytically on paper before implementing
2. **Exact formulas only** - No numerical approximations allowed
3. **Test your derivatives** - Verify your analytic formulas against known solutions
4. **Handle edge cases** - Ensure your functions are well-defined at optimization points

## Example Objective Functions

The project includes several pre-implemented test functions:

- **QuadraticFunction**: `f(x) = (1/2) * x^T * Q * x + b^T * x + c`
- **RosenbrockFunction**: `f(x,y) = (1-x)² + 100(y-x²)²`
- **CircleFunction**: `f(x,y) = x² + y²`

## Testing

Run the test suite:

```bash
python -m pytest tests/
```

Run examples:

```bash
python tests/examples.py
```

## Optimization Methods

### Gradient Descent
- Uses exact analytic gradients
- Includes line search for step size optimization
- Suitable for large-scale problems

### Newton's Method  
- Uses exact analytic Hessian matrices
- Quadratic convergence near optimal points
- Requires positive definite Hessian

## Convergence Criteria

The optimizer stops when any of these conditions are met:

1. **Objective tolerance**: `|f(x_k) - f(x_{k-1})| < obj_tol`
2. **Parameter tolerance**: `||x_k - x_{k-1}|| < param_tol`
3. **Maximum iterations**: `k >= max_iter`

## Output Format

The `minimize()` method returns a dictionary with:

```python
{
    'x': final_point,           # Optimal point found
    'f': final_value,           # Function value at optimal point
    'converged': bool,          # Whether algorithm converged
    'iterations': int,          # Number of iterations used
    'path': array,              # Optimization path (if tracking enabled)
    'f_values': list,           # Function values per iteration
    'method': str               # Method used ('gradient_descent' or 'newton')
}
```

## Contributing

1. All objective functions must implement exact analytic derivatives
2. Include comprehensive unit tests for new functions
3. Add visualization examples for new test problems
4. Document the mathematical derivation of your derivatives

## License

[Add your license information here]

## References

- Nocedal, J., & Wright, S. (2006). Numerical optimization (2nd ed)
- Boyd, S., & Vandenberghe, L. (2004). Convex optimization
- [Add other relevant references]