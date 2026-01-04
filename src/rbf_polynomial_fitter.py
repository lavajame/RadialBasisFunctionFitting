"""
Radial Basis Function + Polynomial Fitter with L2 Regularization

This module provides a fitter that combines RBF kernels with polynomial basis
functions to approximate functions f: R^N -> R using both function values and
gradient information.
"""

import numpy as np
from typing import Tuple, Optional, Callable, Literal
from scipy.stats import qmc


class RBFPolynomialFitter:
    """
    Combines RBF and polynomial basis functions for function fitting.
    
    Supports fitting with both function values and gradient information using
    L2-penalized regression solved via matrix inversion.
    """
    
    # Predefined RBF kernels with shape parameters
    BUILTIN_KERNELS = {
        'multiquadric': ('Multiquadric: sqrt(1 + (epsilon*r)^2)', 0.5),
        'gaussian': ('Gaussian: exp(-(epsilon*r)^2)', 0.5),
        'inverse_multiquadric': ('Inverse Multiquadric: 1/sqrt(1 + (epsilon*r)^2)', 0.5),
        'thin_plate_spline': ('Thin Plate Spline: r^2 * log(r)', None),
        'cubic': ('Cubic: r^3', None),
        'linear': ('Linear: r', None),
    }
    
    def __init__(
        self,
        rbf_kernel: Callable = None,
        rbf_kernel_derivative: Callable = None,
        rbf_name: Optional[Literal['multiquadric', 'gaussian', 'inverse_multiquadric', 
                                    'thin_plate_spline', 'cubic', 'linear']] = None,
        rbf_shape_parameter: Optional[float] = None,
        polynomial_degree: Optional[int] = 1,
        regularization_lambda: float = 1e-6,
    ):
        """
        Initialize the RBF + Polynomial fitter.
        
        Parameters
        ----------
        rbf_kernel : callable, optional
            RBF kernel function k(r) where r is Euclidean distance.
            Ignored if rbf_name is provided. Default: Multiquadric
        rbf_kernel_derivative : callable, optional
            Derivative of RBF kernel w.r.t. r (dr/dx needed for gradients).
            Ignored if rbf_name is provided.
        rbf_name : str, optional
            Name of built-in RBF kernel. Options: 'multiquadric', 'gaussian',
            'inverse_multiquadric', 'thin_plate_spline', 'cubic', 'linear'.
            If provided, overrides rbf_kernel and rbf_kernel_derivative.
        rbf_shape_parameter : float, optional
            Shape parameter (epsilon) for RBF kernels that support it.
            Used only when rbf_name is provided. Default: 0.5.
            Ignored for kernels without shape parameter (e.g., thin plate spline).
        polynomial_degree : int or None, default=1
            Degree of polynomial basis (0=constant, 1=linear, 2=quadratic, etc.).
            If None, no polynomial basis is used (RBF-only).
        regularization_lambda : float, default=1e-6
            L2 regularization parameter (lambda in ridge regression)
        """
        self.polynomial_degree = polynomial_degree
        self.regularization_lambda = regularization_lambda
        self.rbf_name = rbf_name
        self.rbf_shape_parameter = rbf_shape_parameter
        
        # Set up RBF kernel and derivative
        if rbf_name is not None:
            self._setup_builtin_kernel(rbf_name, rbf_shape_parameter)
        else:
            # Custom kernel
            if rbf_kernel is None:
                # Default: Multiquadric RBF
                self.rbf_shape_parameter = rbf_shape_parameter or 0.5
                epsilon = self.rbf_shape_parameter
                self.rbf_kernel = lambda r: np.sqrt(1 + (epsilon * r)**2)
                self.rbf_kernel_derivative = lambda r: (epsilon**2 * r) / np.sqrt(1 + (epsilon * r)**2)
            else:
                self.rbf_kernel = rbf_kernel
                self.rbf_kernel_derivative = rbf_kernel_derivative
        
        self.centers = None
        self.coefficients = None
        self.poly_coefficients = None
        self.n_rbf_bases = None
        self.n_features = None
        self.fitted = False
    
    def _setup_builtin_kernel(self, rbf_name: str, shape_parameter: Optional[float]):
        """Set up a built-in RBF kernel."""
        if rbf_name not in self.BUILTIN_KERNELS:
            raise ValueError(
                f"Unknown RBF kernel '{rbf_name}'. Available: {list(self.BUILTIN_KERNELS.keys())}"
            )
        
        desc, default_shape = self.BUILTIN_KERNELS[rbf_name]
        
        if shape_parameter is None:
            shape_parameter = default_shape
        
        if rbf_name == 'multiquadric':
            self.rbf_shape_parameter = shape_parameter
            self.rbf_kernel = lambda r, eps=shape_parameter: np.sqrt(1 + (eps * r)**2)
            self.rbf_kernel_derivative = lambda r, eps=shape_parameter: (eps**2 * r) / np.sqrt(1 + (eps * r)**2)
        
        elif rbf_name == 'gaussian':
            self.rbf_shape_parameter = shape_parameter
            self.rbf_kernel = lambda r, eps=shape_parameter: np.exp(-(eps * r)**2)
            self.rbf_kernel_derivative = lambda r, eps=shape_parameter: -2 * eps**2 * r * np.exp(-(eps * r)**2)
        
        elif rbf_name == 'inverse_multiquadric':
            self.rbf_shape_parameter = shape_parameter
            self.rbf_kernel = lambda r, eps=shape_parameter: 1 / np.sqrt(1 + (eps * r)**2)
            self.rbf_kernel_derivative = lambda r, eps=shape_parameter: -(eps**2 * r) / (1 + (eps * r)**2)**1.5
        
        elif rbf_name == 'thin_plate_spline':
            # No shape parameter
            self.rbf_shape_parameter = None
            # r^2 * log(r), with r^2 * log(r) -> 0 as r -> 0
            def tps_kernel(r):
                result = np.zeros_like(r, dtype=float)
                mask = r > 0
                with np.errstate(divide='ignore', invalid='ignore'):
                    result[mask] = r[mask]**2 * np.log(r[mask])
                return result
            def tps_deriv(r):
                result = np.zeros_like(r, dtype=float)
                mask = r > 0
                with np.errstate(divide='ignore', invalid='ignore'):
                    result[mask] = r[mask] * (2 * np.log(r[mask]) + 1)
                return result
            self.rbf_kernel = tps_kernel
            self.rbf_kernel_derivative = tps_deriv
        
        elif rbf_name == 'cubic':
            # No shape parameter
            self.rbf_shape_parameter = None
            self.rbf_kernel = lambda r: r**3
            self.rbf_kernel_derivative = lambda r: 3 * r**2
        
        elif rbf_name == 'linear':
            # No shape parameter
            self.rbf_shape_parameter = None
            self.rbf_kernel = lambda r: r
            self.rbf_kernel_derivative = lambda r: np.ones_like(r)
    
    def fit(
        self,
        X: np.ndarray,
        f: np.ndarray,
        df: Optional[np.ndarray] = None,
        centers: Optional[np.ndarray] = None,
        n_centers: Optional[int] = None,
        use_lhs_centers: bool = False,
        lhs_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        random_state: Optional[int] = None,
    ) -> "RBFPolynomialFitter":
        """
        Fit the RBF + polynomial model to data.
        
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Input points for function evaluation
        f : np.ndarray, shape (n_samples,)
            Function values at X
        df : np.ndarray, optional, shape (n_samples, n_features)
            Gradient values at X (d f / d x_i for each dimension i)
            If provided, these are included as additional constraints
        centers : np.ndarray, optional, shape (n_centers, n_features)
            RBF center points. If None and use_lhs_centers=False, uses X as centers.
        n_centers : int, optional
            Number of centers to use. Only relevant if use_lhs_centers=True or 
            if a subset of X centers is desired.
        use_lhs_centers : bool, default=False
            If True, generate center points using Latin Hypercube Sampling instead
            of using training data points. Requires lhs_bounds to be specified.
        lhs_bounds : tuple of (lower, upper), optional
            Lower and upper bounds for LHS sampling, each shape (n_features,).
            Required if use_lhs_centers=True.
        random_state : int, optional
            Random seed for reproducibility in LHS generation
        
        Returns
        -------
        self : RBFPolynomialFitter
            Fitted model
        """
        X = np.atleast_2d(X)
        f = np.atleast_1d(f).reshape(-1)
        
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")
        if f.shape[0] != X.shape[0]:
            raise ValueError(f"f must have same number of samples as X")
        
        n_samples, n_features = X.shape
        self.n_features = n_features
        
        # Set centers
        if use_lhs_centers:
            if lhs_bounds is None:
                raise ValueError("lhs_bounds must be provided when use_lhs_centers=True")
            lower, upper = lhs_bounds
            lower = np.atleast_1d(lower)
            upper = np.atleast_1d(upper)
            if lower.shape[0] != n_features or upper.shape[0] != n_features:
                raise ValueError(
                    f"lhs_bounds must have shape (n_features,)={(n_features,)}, "
                    f"got lower={lower.shape}, upper={upper.shape}"
                )
            
            if n_centers is None:
                n_centers = n_samples
            
            self.centers = self._generate_lhs_centers(n_centers, n_features, lower, upper, random_state)
            self.n_rbf_bases = n_centers
        elif centers is not None:
            self.centers = np.atleast_2d(centers)
            self.n_rbf_bases = self.centers.shape[0]
        else:
            # Use training data as centers, optionally subset
            if n_centers is not None:
                if n_centers > n_samples:
                    raise ValueError(f"n_centers ({n_centers}) cannot exceed n_samples ({n_samples})")
                indices = np.random.choice(n_samples, n_centers, replace=False)
                self.centers = X[indices]
            else:
                self.centers = X.copy()
            self.n_rbf_bases = self.centers.shape[0]
        
        # Build design matrix
        n_poly_bases = self._count_polynomial_bases(n_features) if self.polynomial_degree is not None else 0
        n_total_bases = self.n_rbf_bases + n_poly_bases
        
        # Initialize design matrix and target vector
        if df is None:
            # Only function values
            design_matrix = self._build_rbf_matrix(X)
            target = f.copy()
        else:
            # Function values + gradients
            df = np.atleast_2d(df)
            if df.shape != (n_samples, n_features):
                raise ValueError(
                    f"df must have shape (n_samples, n_features)={X.shape}, "
                    f"got {df.shape}"
                )
            
            n_constraints = n_samples * (1 + n_features)  # f + df_i for each i
            design_matrix = np.zeros((n_constraints, n_total_bases))
            target = np.zeros(n_constraints)
            
            # Fill function value constraints
            design_matrix[:n_samples] = self._build_rbf_matrix(X)
            target[:n_samples] = f
            
            # Fill gradient constraints
            for i in range(n_features):
                row_start = n_samples + i * n_samples
                row_end = row_start + n_samples
                design_matrix[row_start:row_end] = self._build_rbf_gradient_matrix(X, i)
                target[row_start:row_end] = df[:, i]
        
        # Add L2 regularization (ridge regression)
        # Solve: (D^T D + lambda*I) c = D^T y
        # Using: c = (D^T D + lambda*I)^{-1} D^T y
        
        gram_matrix = design_matrix.T @ design_matrix
        ridge_matrix = gram_matrix + self.regularization_lambda * np.eye(n_total_bases)
        
        right_hand_side = design_matrix.T @ target
        
        try:
            self.coefficients = np.linalg.inv(ridge_matrix) @ right_hand_side
        except np.linalg.LinAlgError:
            raise ValueError("Design matrix is singular; increase regularization_lambda")
        
        self.fitted = True
        return self
    
    @staticmethod
    def _generate_lhs_centers(n_centers: int, n_features: int, 
                             lower: np.ndarray, upper: np.ndarray,
                             random_state: Optional[int] = None) -> np.ndarray:
        """
        Generate centers using Latin Hypercube Sampling.
        
        Parameters
        ----------
        n_centers : int
            Number of centers to generate
        n_features : int
            Dimension of input space
        lower : np.ndarray, shape (n_features,)
            Lower bounds
        upper : np.ndarray, shape (n_features,)
            Upper bounds
        random_state : int, optional
            Random seed
        
        Returns
        -------
        centers : np.ndarray, shape (n_centers, n_features)
            LHS-sampled center points
        """
        sampler = qmc.LatinHypercube(d=n_features, seed=random_state)
        lhs_samples = sampler.random(n_centers)  # shape (n_centers, n_features), values in [0, 1)
        
        # Scale to bounds
        centers = lower + lhs_samples * (upper - lower)
        return centers
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict function values at new points.
        
        Parameters
        ----------
        X : np.ndarray, shape (n_eval, n_features)
            Points at which to evaluate the fitted function
        
        Returns
        -------
        predictions : np.ndarray, shape (n_eval,)
            Predicted function values
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X = np.atleast_2d(X)
        basis_values = self._evaluate_basis(X)
        return basis_values @ self.coefficients
    
    def predict_gradient(self, X: np.ndarray) -> np.ndarray:
        """
        Predict gradient at new points.
        
        Parameters
        ----------
        X : np.ndarray, shape (n_eval, n_features)
            Points at which to evaluate the gradient
        
        Returns
        -------
        gradient : np.ndarray, shape (n_eval, n_features)
            Gradient predictions
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X = np.atleast_2d(X)
        n_eval = X.shape[0]
        gradient = np.zeros((n_eval, self.n_features))
        
        for i in range(self.n_features):
            basis_gradients = self._evaluate_basis_gradient(X, i)
            gradient[:, i] = basis_gradients @ self.coefficients
        
        return gradient
    
    def _count_polynomial_bases(self, n_features: int) -> int:
        """Count number of polynomial basis functions."""
        if self.polynomial_degree is None:
            return 0
        # Polynomial bases from constant to degree polynomial_degree
        # Using homogeneous polynomial counting
        count = 0
        for d in range(self.polynomial_degree + 1):
            count += self._binomial(n_features + d - 1, d)
        return count
    
    @staticmethod
    def _binomial(n: int, k: int) -> int:
        """Compute binomial coefficient."""
        if k > n or k < 0:
            return 0
        if k == 0 or k == n:
            return 1
        k = min(k, n - k)
        result = 1
        for i in range(k):
            result = result * (n - i) // (i + 1)
        return result
    
    def _build_rbf_matrix(self, X: np.ndarray) -> np.ndarray:
        """Build RBF + polynomial basis matrix for function values."""
        n_samples = X.shape[0]
        n_poly_bases = self._count_polynomial_bases(self.n_features)
        matrix = np.zeros((n_samples, self.n_rbf_bases + n_poly_bases))
        
        # RBF part: compute distances and kernel values
        distances = np.linalg.norm(X[:, np.newaxis, :] - self.centers[np.newaxis, :, :], axis=2)
        matrix[:, :self.n_rbf_bases] = self.rbf_kernel(distances)
        
        # Polynomial part
        poly_start = self.n_rbf_bases
        poly_matrix = self._build_polynomial_basis(X)
        matrix[:, poly_start:] = poly_matrix
        
        return matrix
    
    def _build_rbf_gradient_matrix(self, X: np.ndarray, dim: int) -> np.ndarray:
        """Build RBF gradient matrix for a specific dimension."""
        n_samples = X.shape[0]
        n_poly_bases = self._count_polynomial_bases(self.n_features)
        matrix = np.zeros((n_samples, self.n_rbf_bases + n_poly_bases))
        
        # RBF gradient: d/dx_dim [k(||x - c||)]
        # = k'(r) * (x_dim - c_dim) / r
        diff = X[:, np.newaxis, dim] - self.centers[np.newaxis, :, dim]
        distances = np.linalg.norm(X[:, np.newaxis, :] - self.centers[np.newaxis, :, :], axis=2)
        
        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            kernel_deriv = self.rbf_kernel_derivative(distances)
            gradient = kernel_deriv * diff / distances
            gradient = np.nan_to_num(gradient)  # 0/0 -> 0
        
        matrix[:, :self.n_rbf_bases] = gradient
        
        # Polynomial gradient part
        poly_start = self.n_rbf_bases
        poly_grad_matrix = self._build_polynomial_gradient_basis(X, dim)
        matrix[:, poly_start:] = poly_grad_matrix
        
        return matrix
    
    def _build_polynomial_basis(self, X: np.ndarray) -> np.ndarray:
        """Build polynomial basis functions."""
        n_samples, n_features = X.shape
        
        if self.polynomial_degree is None:
            return np.zeros((n_samples, 0))
        
        n_bases = self._count_polynomial_bases(n_features)
        poly_matrix = np.zeros((n_samples, n_bases))
        
        col = 0
        for degree in range(self.polynomial_degree + 1):
            for indices in self._generate_multiindices(n_features, degree):
                poly_matrix[:, col] = np.prod([X[:, i] ** indices[i] for i in range(n_features)], axis=0)
                col += 1
        
        return poly_matrix
    
    def _build_polynomial_gradient_basis(self, X: np.ndarray, dim: int) -> np.ndarray:
        """Build polynomial gradient basis functions."""
        n_samples, n_features = X.shape
        
        if self.polynomial_degree is None:
            return np.zeros((n_samples, 0))
        
        n_bases = self._count_polynomial_bases(n_features)
        poly_grad_matrix = np.zeros((n_samples, n_bases))
        
        col = 0
        for degree in range(self.polynomial_degree + 1):
            for indices in self._generate_multiindices(n_features, degree):
                if indices[dim] == 0:
                    poly_grad_matrix[:, col] = 0
                else:
                    # d/dx_dim [x_0^i_0 * ... * x_n^i_n] = i_dim * x_0^i_0 * ... * x_dim^(i_dim-1) * ... * x_n^i_n
                    new_indices = list(indices)
                    new_indices[dim] -= 1
                    poly_grad_matrix[:, col] = indices[dim] * np.prod(
                        [X[:, i] ** new_indices[i] for i in range(n_features)], axis=0
                    )
                col += 1
        
        return poly_grad_matrix
    
    @staticmethod
    def _generate_multiindices(n_features: int, degree: int):
        """Generate all multi-indices of given degree."""
        if n_features == 1:
            yield [degree]
        else:
            for i in range(degree + 1):
                for rest in RBFPolynomialFitter._generate_multiindices(n_features - 1, degree - i):
                    yield [i] + rest
    
    def _evaluate_basis(self, X: np.ndarray) -> np.ndarray:
        """Evaluate basis functions at X."""
        n_samples = X.shape[0]
        n_poly_bases = self._count_polynomial_bases(self.n_features)
        basis = np.zeros((n_samples, self.n_rbf_bases + n_poly_bases))
        
        # RBF part
        distances = np.linalg.norm(X[:, np.newaxis, :] - self.centers[np.newaxis, :, :], axis=2)
        basis[:, :self.n_rbf_bases] = self.rbf_kernel(distances)
        
        # Polynomial part
        poly_start = self.n_rbf_bases
        poly_matrix = self._build_polynomial_basis(X)
        basis[:, poly_start:] = poly_matrix
        
        return basis
    
    def _evaluate_basis_gradient(self, X: np.ndarray, dim: int) -> np.ndarray:
        """Evaluate basis function gradients at X."""
        n_samples = X.shape[0]
        n_poly_bases = self._count_polynomial_bases(self.n_features)
        basis_grad = np.zeros((n_samples, self.n_rbf_bases + n_poly_bases))
        
        # RBF gradient part
        diff = X[:, np.newaxis, dim] - self.centers[np.newaxis, :, dim]
        distances = np.linalg.norm(X[:, np.newaxis, :] - self.centers[np.newaxis, :, :], axis=2)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            kernel_deriv = self.rbf_kernel_derivative(distances)
            gradient = kernel_deriv * diff / distances
            gradient = np.nan_to_num(gradient)
        
        basis_grad[:, :self.n_rbf_bases] = gradient
        
        # Polynomial gradient part
        poly_start = self.n_rbf_bases
        poly_grad_matrix = self._build_polynomial_gradient_basis(X, dim)
        basis_grad[:, poly_start:] = poly_grad_matrix
        
        return basis_grad
