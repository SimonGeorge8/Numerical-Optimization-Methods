#!/usr/bin/env python3
"""
Complete Assignment Runner for Numerical Optimization Assignment

This script runs all tests and generates all required output for the assignment.
It produces the exact format and results needed for the report.

Run this script to generate:
1. Console output with iteration details (saved to file)
2. Contour plots with optimization paths  
3. Convergence plots showing function values vs iterations

Usage: python run_assignment.py
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path
from datetime import datetime

# Add src and tests to Python path
src_path = Path(__file__).parent / "src"
tests_path = Path(__file__).parent / "tests"
sys.path.insert(0, str(src_path))
sys.path.insert(0, str(tests_path))

# Now import our modules
try:
    from unconstrained_min import UnconstrainedMinimizer
    from examples import get_assignment_functions
    from utils import plot_contour, plot_function_values
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure you have saved the files correctly:")
    print("  src/unconstrained_min.py")
    print("  src/utils.py") 
    print("  tests/examples.py")
    print(f"Current working directory: {Path.cwd()}")
    sys.exit(1)


class OutputCapture:
    """Class to capture output and write to both console and file."""
    
    def __init__(self, filename):
        self.filename = filename
        self.file = open(filename, 'w', encoding='utf-8')
        
    def write(self, text):
        # Write to both console and file
        sys.__stdout__.write(text)
        self.file.write(text)
        self.file.flush()  # Ensure immediate writing
        
    def close(self):
        self.file.close()


def run_assignment():
    """Run the complete assignment with all required functions and parameters."""
    
    # Create results directory
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # Create output file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"results/assignment_results_{timestamp}.txt"
    
    # Set up output capture
    output_capture = OutputCapture(output_filename)
    
    def print_and_save(*args, **kwargs):
        """Custom print function that saves to file and displays on console."""
        # Convert all arguments to strings and join them
        message = ' '.join(str(arg) for arg in args)
        if kwargs.get('end', '\n') != '\n':
            message += kwargs.get('end', '')
        else:
            message += '\n'
        output_capture.write(message)
    
    try:
        print_and_save("NUMERICAL OPTIMIZATION WITH PYTHON 2025B")
        print_and_save("Programming Assignment 01 - Results")
        print_and_save(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print_and_save("=" * 80)
        
        # Assignment parameters (exactly as specified in PDF)
        obj_tol = 1e-12      # Objective function change tolerance  
        param_tol = 1e-8     # Step tolerance
        wolfe_c1 = 0.01      # Wolfe condition constant
        backtrack_factor = 0.5  # Backtracking constant
        
        print_and_save(f"Parameters:")
        print_and_save(f"  Objective tolerance: {obj_tol}")
        print_and_save(f"  Parameter tolerance: {param_tol}")
        print_and_save(f"  Wolfe constant: {wolfe_c1}")
        print_and_save(f"  Backtracking factor: {backtrack_factor}")
        print_and_save(f"  Starting points: x0 = [1,1] (except Rosenbrock: [-1,2])")
        print_and_save(f"  Function interface: func(x, hessian_needed) -> (f, g, h)")
        print_and_save("=" * 80)
        
        # Get all assignment functions
        functions = get_assignment_functions()
        
        # Process each function
        for i, (func_name, func_data) in enumerate(functions.items(), 1):
            print_and_save(f"\n{i}. {func_name.upper()}")
            print_and_save("=" * 60)
            
            func = func_data['func']
            x0 = func_data['x0']
            title = func_data['title']
            
            print_and_save(f"Function: {title}")
            print_and_save(f"Starting point: x0 = {x0}")
            
            # Verify function interface
            try:
                f_test, g_test, h_test = func(x0, hessian_needed=True)
                print_and_save(f"Function interface verified: f={f_test:.6f}, grad_shape={g_test.shape}, hess_shape={h_test.shape}")
            except Exception as e:
                print_and_save(f"ERROR: Function interface problem: {e}")
                continue
            
            # Set iteration limits per assignment requirements
            if func_name == 'Rosenbrock':
                max_iter_gd = 10000  # Special case for Rosenbrock + GD
                max_iter_newton = 100
            else:
                max_iter_gd = 100
                max_iter_newton = 100
            
            # Storage for results
            results = {}
            paths = []
            f_values = []
            method_names = []
            
            # Test both methods
            methods = [
                ('gradient_descent', 'Gradient Descent', max_iter_gd),
                ('newton', 'Newton\'s Method', max_iter_newton)
            ]
            
            for method_key, method_name, max_iter in methods:
                print_and_save(f"\n{'-'*40}")
                print_and_save(f"{method_name.upper()}")
                print_and_save(f"{'-'*40}")
                
                # Create minimizer
                minimizer = UnconstrainedMinimizer(method=method_key)
                
                # Run optimization
                result = minimizer.minimize(
                    f=func,
                    x0=x0.copy(),
                    obj_tol=obj_tol,
                    param_tol=param_tol,
                    max_iter=max_iter
                )
                
                # Store results
                results[method_name] = result
                paths.append(result['path'])
                f_values.append(result['f_values'])
                method_names.append(method_name)
                
                # Print final result (as required by assignment)
                print_and_save(f"\nFINAL ITERATION REPORT:")
                print_and_save(f"  Method: {method_name}")
                print_and_save(f"  Final location x: {result['x']}")
                print_and_save(f"  Final objective value f(x): {result['f']:.12e}")
                print_and_save(f"  Total iterations: {result['iterations']}")
                print_and_save(f"  Converged: {'SUCCESS' if result['converged'] else 'FAILURE'}")
                print_and_save(f"  Success/Failure flag: {result['converged']}")
            
            # Generate plots (as required by assignment)
            print_and_save(f"\n{'-'*40}")
            print_and_save("GENERATING REQUIRED PLOTS")
            print_and_save(f"{'-'*40}")
            
            # Determine appropriate plot limits
            plot_limits = {
                'Quadratic Circles': ((-2, 2), (-2, 2)),
                'Quadratic Axis-Aligned': ((-1.5, 1.5), (-0.5, 0.5)),
                'Quadratic Rotated': ((-1.5, 1.5), (-1.5, 1.5)),
                'Rosenbrock': ((-2, 2), (-1, 3)),
                'Linear': ((-3, 3), (-3, 3)),
                'Smoothed Corner Triangles': ((-2, 2), (-2, 2))
            }
            
            x_limits, y_limits = plot_limits.get(func_name, ((-2, 2), (-2, 2)))
            
            # Plot 1: Contour lines with iteration paths
            plt.figure(figsize=(10, 8))
            plot_contour(
                objective_func=func,
                x_limits=x_limits,
                y_limits=y_limits,
                algorithm_paths=paths,
                path_names=method_names,
                levels=20,
                title=f"{title} - Optimization Paths"
            )
            
            # Save contour plot
            contour_filename = f"results/{func_name.lower().replace(' ', '_')}_contour.png"
            plt.savefig(contour_filename, dpi=300, bbox_inches='tight')
            print_and_save(f"  Saved: {contour_filename}")
            plt.close()
            
            # Plot 2: Function values vs iteration number
            plt.figure(figsize=(10, 6))
            plot_function_values(
                iteration_data=f_values,
                method_names=method_names,
                title=f"{title} - Function Value vs Iteration",
                log_scale=True
            )
            
            # Save convergence plot
            convergence_filename = f"results/{func_name.lower().replace(' ', '_')}_convergence.png"
            plt.savefig(convergence_filename, dpi=300, bbox_inches='tight')
            print_and_save(f"  Saved: {convergence_filename}")
            plt.close()
            
            # Print comparison summary
            print_and_save(f"\n{'-'*40}")
            print_and_save("COMPARISON SUMMARY")
            print_and_save(f"{'-'*40}")
            print_and_save(f"{'Method':<20} {'Iterations':<10} {'Final Value':<15} {'Status'}")
            print_and_save(f"{'-'*55}")
            for method_name, result in results.items():
                status = "SUCCESS" if result['converged'] else "FAILURE"
                print_and_save(f"{method_name:<20} {result['iterations']:<10} {result['f']:<15.6e} {status}")
        
        # Final summary
        print_and_save(f"\n{'='*80}")
        print_and_save("ASSIGNMENT COMPLETED SUCCESSFULLY")
        print_and_save(f"{'='*80}")
        print_and_save("Generated files in 'results/' directory:")
        
        for func_name in functions.keys():
            base_name = func_name.lower().replace(' ', '_')
            print_and_save(f"  - {base_name}_contour.png (contour plot with paths)")
            print_and_save(f"  - {base_name}_convergence.png (function values vs iterations)")
        
        print_and_save(f"  - {output_filename} (complete results output)")
        
        print_and_save(f"\nFor your report, include:")
        print_and_save(f"  1. All 'FINAL ITERATION REPORT' outputs shown above")
        print_and_save(f"  2. All contour plots with optimization paths (12 plots total)")
        print_and_save(f"  3. All convergence plots (12 plots total)")
        print_and_save(f"  4. DO NOT include the entire iteration logs - only final results")
        
        print_and_save(f"\nREPORT CHECKLIST:")
        print_and_save(f"  ✓ GitHub link at beginning of report")
        print_and_save(f"  ✓ Final iteration details for each test (shown above)")
        print_and_save(f"  ✓ Contour plots with GD and NT paths for each function")
        print_and_save(f"  ✓ Function value vs iteration plots for each function")
        print_and_save(f"  ✓ Success/failure flags for each optimization")
        print_and_save(f"  ✓ Correct function interface: func(x, hessian_needed) -> (f, g, h)")
        
        # Create formatted results for easy copying
        print_and_save(f"\n{'='*80}")
        print_and_save("FORMATTED RESULTS FOR REPORT")
        print_and_save(f"{'='*80}")
        
        for func_name, func_data in functions.items():
            print_and_save(f"\n## {func_name}")
            print_and_save(f"")
            if func_name in ['Gradient Descent', 'Newton\'s Method']:
                continue
            
            # This section would need the actual results to fill in
            print_and_save(f"**Gradient Descent:**")
            print_and_save(f"- Final Location: [To be filled from results above]")
            print_and_save(f"- Final Objective value: [To be filled from results above]") 
            print_and_save(f"- Iterations: [To be filled from results above]")
            print_and_save(f"- Success: [To be filled from results above]")
            print_and_save(f"")
            print_and_save(f"**Newton's Method:**")
            print_and_save(f"- Final Location: [To be filled from results above]")
            print_and_save(f"- Final Objective value: [To be filled from results above]")
            print_and_save(f"- Iterations: [To be filled from results above]") 
            print_and_save(f"- Success: [To be filled from results above]")
            print_and_save(f"")
            print_and_save("---")
        
    finally:
        output_capture.close()
        print(f"\n✓ All output saved to: {output_filename}")


def test_imports():
    """Test that all required modules can be imported."""
    try:
        from unconstrained_min import UnconstrainedMinimizer
        from examples import get_assignment_functions
        from utils import plot_contour, plot_function_values
        print("✓ All modules imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False


def test_function_interface():
    """Test that all functions follow the correct interface."""
    try:
        functions = get_assignment_functions()
        print("✓ Testing function interfaces...")
        
        for name, func_data in functions.items():
            func = func_data['func']
            x0 = func_data['x0']
            
            # Test interface
            f, g, h = func(x0, hessian_needed=True)
            assert isinstance(f, float), f"Function {name} should return float for f"
            assert isinstance(g, np.ndarray), f"Function {name} should return ndarray for gradient"
            assert isinstance(h, np.ndarray), f"Function {name} should return ndarray for Hessian when requested"
            
            # Test without Hessian
            f2, g2, h2 = func(x0, hessian_needed=False)
            assert h2 is None, f"Function {name} should return None for Hessian when not requested"
            
            print(f"  ✓ {name}: interface correct")
        
        print("✓ All function interfaces are correct")
        return True
    except Exception as e:
        print(f"✗ Function interface error: {e}")
        return False


def main():
    """Main function."""
    print("Testing imports...")
    if not test_imports():
        print("Please ensure all files are in the correct directories:")
        print("  src/unconstrained_min.py")
        print("  src/utils.py")
        print("  tests/examples.py")
        return
    
    print("Testing function interfaces...")
    if not test_function_interface():
        print("Function interface problems detected. Please check examples.py")
        return
    
    print("Starting assignment execution...\n")
    
    try:
        run_assignment()
    except Exception as e:
        print(f"\nERROR during execution: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"\n{'='*80}")
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("Ready for report submission.")
    print("Interface verified: func(x, hessian_needed) -> (f, g, h)")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()