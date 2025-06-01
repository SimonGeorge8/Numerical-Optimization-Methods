"""
Utility functions for numerical optimization visualization and analysis.
"""
import matplotlib.pyplot as plt
import numpy as np


def plot_contour(objective_func, x_limits, y_limits, algorithm_paths=None, path_names=None, 
                 levels=20, figsize=(10, 8), title=None):
    """
    Plot contour lines of an objective function with optional algorithm paths.
    
    Parameters:
    -----------
    objective_func : callable
        The objective function to plot f(x, y)
    x_limits : tuple
        (x_min, x_max) limits for x-axis
    y_limits : tuple
        (y_min, y_max) limits for y-axis
    algorithm_paths : list of arrays, optional
        List of optimization paths, where each path is an array of shape (n_iterations, 2)
    path_names : list of str, optional
        Names for each algorithm path (for legend)
    levels : int or array-like, optional
        Number of contour levels or specific level values
    figsize : tuple, optional
        Figure size (width, height)
    title : str, optional
        Plot title
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    ax : matplotlib.axes.Axes
        The axes object
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create grid
    x = np.linspace(x_limits[0], x_limits[1], 200)
    y = np.linspace(y_limits[0], y_limits[1], 200)
    X, Y = np.meshgrid(x, y)
    
    # Evaluate function
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            try:
                if hasattr(objective_func, 'func'):
                    Z[i, j] = objective_func.func([X[i, j], Y[i, j]])
                else:
                    try:
                        f_val, _, _ = objective_func([X[i, j], Y[i, j]], hessian_needed=False)
                        Z[i, j] = f_val
                    except:
                        Z[i, j] = objective_func([X[i, j], Y[i, j]])
            except:
                Z[i, j] = np.nan
    
    # Determine contour levels
    valid_z = Z[~np.isnan(Z)]
    if len(valid_z) > 0:
        z_min, z_max = np.min(valid_z), np.max(valid_z)
        if isinstance(levels, int):
            if z_max / (z_min + 1e-10) > 100:
                # Log scale for large ranges
                levels = np.logspace(np.log10(max(z_min, 1e-10)), np.log10(z_max), 30)
            else:
                levels = np.linspace(z_min, z_max, 30)
    
    # Plot contours
    contourf = ax.contourf(X, Y, Z, levels=levels, cmap='Blues', alpha=0.3)
    contour = ax.contour(X, Y, Z, levels=levels[::3], colors='gray', alpha=0.4, linewidths=0.8)
    
    # Add colorbar
    cbar = plt.colorbar(contourf, ax=ax)
    cbar.set_label('Function Value')
    
    # Plot paths
    if algorithm_paths is not None:
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
        
        if path_names is None:
            path_names = [f'Algorithm {i+1}' for i in range(len(algorithm_paths))]
        
        for i, (path, name) in enumerate(zip(algorithm_paths, path_names)):
            if len(path) > 0:
                path = np.array(path)
                color = colors[i % len(colors)]
                
                # Plot line
                ax.plot(path[:, 0], path[:, 1], 
                       color=color, linewidth=2.5, label=name, alpha=0.8)
                
                # Plot points
                step = max(1, len(path) // 20)
                ax.plot(path[::step, 0], path[::step, 1], 
                       'o', color=color, markersize=4)
                
                # Start and end
                ax.plot(path[0, 0], path[0, 1], 
                       'o', color=color, markersize=10, 
                       markerfacecolor='white', markeredgewidth=2)
                ax.plot(path[-1, 0], path[-1, 1], 
                       '*', color=color, markersize=15)
        
        ax.legend()
    
    # Labels
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_xlim(x_limits)
    ax.set_ylim(y_limits)
    ax.grid(True, alpha=0.3)
    
    if title:
        ax.set_title(title)
    
    plt.tight_layout()
    return fig, ax


def plot_function_values(iteration_data, method_names, figsize=(10, 6), title=None, 
                        xlabel="Iteration", ylabel="Function Value", log_scale=False):
    """
    Plot function values at each iteration for multiple optimization methods.
    
    Parameters:
    -----------
    iteration_data : list of arrays
        List where each element is an array of function values per iteration for one method
    method_names : list of str
        Names of the optimization methods (for legend)
    figsize : tuple, optional
        Figure size (width, height)
    title : str, optional
        Plot title
    xlabel : str, optional
        Label for x-axis
    ylabel : str, optional
        Label for y-axis
    log_scale : bool, optional
        Whether to use logarithmic scale for y-axis
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    ax : matplotlib.axes.Axes
        The axes object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    linestyles = ['-', '--', '-.', ':', '-', '--']
    
    for i, (data, name) in enumerate(zip(iteration_data, method_names)):
        if len(data) > 0:
            iterations = range(len(data))
            color = colors[i % len(colors)]
            linestyle = linestyles[i % len(linestyles)]
            
            ax.plot(iterations, data, color=color, linestyle=linestyle, 
                   linewidth=2, marker='o', markersize=4, 
                   markevery=max(1, len(data)//20), label=name)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    
    if log_scale:
        ax.set_yscale('log')
    
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    
    if title:
        ax.set_title(title)
    
    plt.tight_layout()
    return fig, ax


def plot_convergence_comparison(results_dict, figsize=(15, 6)):
    """
    Create a comprehensive comparison plot showing both contour plots and convergence curves.
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary containing optimization results with keys:
        - 'objective_func': the objective function
        - 'x_limits': x-axis limits
        - 'y_limits': y-axis limits  
        - 'methods': dict with method names as keys and results as values
    figsize : tuple, optional
        Figure size (width, height)
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    axes : array of matplotlib.axes.Axes
        Array of axes objects
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    objective_func = results_dict['objective_func']
    x_limits = results_dict['x_limits']
    y_limits = results_dict['y_limits']
    methods = results_dict['methods']
    
    # Extract paths and function values
    paths = []
    f_values = []
    method_names = []
    
    for method_name, result in methods.items():
        if 'path' in result and len(result['path']) > 0:
            paths.append(result['path'])
            f_values.append(result['f_values'])
            method_names.append(method_name)
    
    # Plot contours with paths
    plt.sca(ax1)
    plot_contour(objective_func, x_limits, y_limits, 
                algorithm_paths=paths, path_names=method_names, 
                title='Optimization Paths')
    
    # Plot convergence curves
    plt.sca(ax2)
    plot_function_values(f_values, method_names, 
                        title='Convergence Comparison', log_scale=True)
    
    plt.tight_layout()
    return fig, (ax1, ax2)


def save_optimization_results(results, filename, format='png', dpi=150):
    """
    Save optimization results and plots to file.
    
    Parameters:
    -----------
    results : dict
        Dictionary containing optimization results
    filename : str
        Output filename (without extension)
    format : str, optional
        File format ('png', 'pdf', 'svg')
    dpi : int, optional
        Resolution for raster formats
    """
    plt.savefig(f"{filename}.{format}", dpi=dpi, bbox_inches='tight')


def print_optimization_summary(results_dict):
    """
    Print a formatted summary of optimization results.
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary containing optimization results for multiple methods
    """
    print("\nOptimization Summary")
    print("=" * 50)
    
    for method_name, result in results_dict.items():
        print(f"\n{method_name}:")
        print(f"  Final point: {result['x']}")
        print(f"  Final value: {result['f']:.8e}")
        print(f"  Iterations: {result['iterations']}")
        print(f"  Converged: {result['converged']}")


def create_test_functions():
    """
    Create a dictionary of common test functions for optimization.
    
    Returns:
    --------
    test_functions : dict
        Dictionary with function names as keys and function objects as values
    """
    try:
        from examples import get_assignment_functions
        return get_assignment_functions()
    except ImportError:
        # Return some basic test functions if examples module not available
        return {
            'quadratic': lambda x: x[0]**2 + x[1]**2,
            'rosenbrock': lambda x: (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2
        }


def validate_convergence_criteria(obj_tol, param_tol, max_iter):
    """
    Validate that convergence criteria are reasonable.
    
    Parameters:
    -----------
    obj_tol : float
        Objective function tolerance
    param_tol : float
        Parameter tolerance
    max_iter : int
        Maximum number of iterations
        
    Returns:
    --------
    is_valid : bool
        True if criteria are valid
    warnings : list
        List of warning messages if any issues found
    """
    warnings = []
    is_valid = True
    
    if obj_tol <= 0:
        warnings.append("Objective tolerance must be positive")
        is_valid = False
    
    if param_tol <= 0:
        warnings.append("Parameter tolerance must be positive")
        is_valid = False
    
    if max_iter <= 0:
        warnings.append("Maximum iterations must be positive")
        is_valid = False
    
    if obj_tol < 1e-16:
        warnings.append("Objective tolerance may be too strict (< 1e-16)")
    
    if param_tol < 1e-16:
        warnings.append("Parameter tolerance may be too strict (< 1e-16)")
    
    if max_iter > 50000:
        warnings.append("Maximum iterations may be too large (> 50000)")
    
    return is_valid, warnings


def get_optimal_plot_limits(initial_point, scale_factor=2.0):
    """
    Automatically determine good plot limits for a function based on starting point.
    
    Parameters:
    -----------
    initial_point : array-like
        Starting point for optimization
    scale_factor : float
        How much to scale the limits around the starting point
        
    Returns:
    --------
    x_limits : tuple
        (x_min, x_max) for x-axis
    y_limits : tuple  
        (y_min, y_max) for y-axis
    """
    x0 = np.array(initial_point)
    
    # Default limits based on starting point
    x_range = max(abs(x0[0]), 1.0) * scale_factor
    y_range = max(abs(x0[1]), 1.0) * scale_factor
    
    x_limits = (x0[0] - x_range, x0[0] + x_range)
    y_limits = (x0[1] - y_range, x0[1] + y_range)
    
    return x_limits, y_limits