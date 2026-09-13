# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Orders of Magnitude LLC <orders@ofmagnitude.com>
"""
Bella GPAW Adapter: CheFSI eigensolver for GPAW LCAO calculations.

Implements the diagonalizer interface that GPAW expects:
    diagonalize(H_MM, C_nM, eps_n, S_MM, is_already_decomposed)

Solves generalized eigenvalue problem H C = S C eps via:
    1. Cholesky: S = L L^H
    2. Transform: H_tilde = L^{-1} H L^{-H} (via LinearOperator)
    3. CheFSI: H_tilde y = λ y
    4. Back-transform: C = L^{-H} y
"""

import numpy as np
import time
from scipy.linalg import cholesky, solve_triangular, eigh
from scipy.sparse.linalg import LinearOperator
from bella_chefsi_v2 import chefsi


class BellaDiagonalizer:
    """
    CheFSI-based diagonalizer for GPAW LCAO calculations.
    
    Computes only the lowest k eigenpairs (occupied bands + buffer)
    instead of all N eigenpairs, providing significant speedup for large systems.
    """
    
    accepts_decomposed_overlap_matrix = False
    
    def __repr__(self):
        return 'Bella (CheFSI block-sparse eigensolver)'
    
    def diagonalize(self, H_MM, C_nM, eps_n, S_MM, is_already_decomposed):
        """
        Solve generalized eigenvalue problem: H C = S C eps
        for k = C_nM.shape[0] lowest eigenpairs.
        
        Parameters:
        -----------
        H_MM : numpy.ndarray (N, N)
            Hamiltonian matrix
        C_nM : numpy.ndarray (k, N)
            Eigenvector output (rows = eigenvectors, GPAW convention)
        eps_n : numpy.ndarray (k,)
            Eigenvalue output
        S_MM : numpy.ndarray (N, N)
            Overlap matrix
        is_already_decomposed : bool
            If True, S_MM is already Cholesky factor L
        """
        N = H_MM.shape[0]
        k = C_nM.shape[0]
        
        # Step 1: Cholesky decompose S = L L^H (lower triangular)
        if is_already_decomposed:
            L = S_MM
        else:
            L = cholesky(S_MM, lower=True)
        
        # Step 2: Create LinearOperator for H_tilde = L^{-1} H L^{-H}
        # H_tilde @ V = L^{-1} @ H @ L^{-H} @ V (implicit solves, no dense L_inv)
        def matmat(V):
            # Solve L^H @ temp = V  (temp = L^{-H} @ V)
            temp = solve_triangular(L.conj().T, V, lower=False)
            # Compute H @ temp
            H_temp = H_MM @ temp
            # Solve L @ result = H_temp  (result = L^{-1} @ H @ L^{-H} @ V)
            result = solve_triangular(L, H_temp, lower=True)
            return result
        
        def matvec(v):
            return matmat(v[:, None])[:, 0]
        
        H_op = LinearOperator((N, N), matvec=matvec, matmat=matmat, dtype=H_MM.dtype)
        
        # Step 4: Warm start from previous eigenvectors if available
        V_init = None
        if not np.all(C_nM == 0):
            V_init = C_nM.T.copy()   # (N, k): rows of C_nM are eigenvectors
        
        # Step 5: Run CheFSI v2 on standard problem (via LinearOperator)
        vals, Y, n_iters = chefsi(H_op, N, k)
        
        # Step 6: Back-transform eigenvectors
        # C = L^{-H} Y
        C_raw = solve_triangular(L.conj().T, Y, lower=False)  # (N, k)
        
        # Step 7: Write into GPAW output arrays (in-place)
        eps_n[:] = vals
        C_nM[:] = C_raw.T  # GPAW wants (k, N)


def run_gpaw_benchmark(atoms, basis='dzp', h=0.2):
    """
    Benchmark Bella CheFSI on a synthetic sparse Hamiltonian
    that mimics real GPAW Hamiltonian structure.
    
    Parameters:
    -----------
    atoms : ase.Atoms
        Atomic structure (used to determine matrix size)
    basis : str
        Basis set (unused, kept for compatibility)
    h : float
        Grid spacing (unused, kept for compatibility)
    
    Returns:
    --------
    dict
        Benchmark results including times, speedup, residual norm, sparsity
    """
    from scipy.sparse import random, eye as sparse_eye
    
    print(f"\n{'='*70}")
    print(f"Benchmark: {atoms.get_chemical_formula()}, N={len(atoms)} atoms")
    print(f"Using synthetic sparse Hamiltonian (GPAW-like structure)")
    print(f"{'='*70}\n")
    
    # Determine matrix size based on number of atoms
    # Approximate: ~50 basis functions per atom for dzp basis
    n_atoms = len(atoms)
    N = n_atoms * 50  # Approximate basis size
    n_occ = n_atoms * 4  # Approximate occupied states (4 valence e- per Si)
    k = n_occ + 10  # Buffer of 10 unoccupied states
    
    print(f"Matrix size: N={N}")
    print(f"Number of occupied bands: {n_occ}")
    print(f"Target eigenpairs: k={k}")
    
    # Generate sparse Hamiltonian (mimics real-space FD Hamiltonian)
    # Real GPAW Hamiltonians are ~95-99% sparse
    density = 0.05  # 5% density = 95% sparse
    H_sparse = random(N, N, density=density, format='csr', random_state=42)
    H_sparse = H_sparse + H_sparse.T  # Make symmetric
    # Add diagonal dominance for numerical stability
    H_sparse = H_sparse + N * 2 * sparse_eye(N, format='csr')
    
    # Generate sparse overlap matrix (LCAO basis overlap is sparse)
    S_sparse = random(N, N, density=density*2, format='csr', random_state=43)
    S_sparse = S_sparse + S_sparse.T
    S_sparse = S_sparse + N * sparse_eye(N, format='csr')  # Make SPD
    
    # Compute sparsity
    nnz = H_sparse.nnz
    sparsity = 100.0 * (1.0 - nnz / (N * N))
    print(f"Hamiltonian sparsity: {sparsity:.2f}%\n")
    
    # --- Bella CheFSI benchmark ---
    print("Running Bella CheFSI...")
    t_bella_start = time.perf_counter()
    
    # Cholesky decompose S (need dense for Cholesky)
    S_dense = S_sparse.toarray()
    L = cholesky(S_dense, lower=True)
    
    # Create LinearOperator for H_tilde using sparse matvec
    def matmat(V):
        # L^{-H} @ V via triangular solve
        temp = solve_triangular(L.conj().T, V, lower=False)
        # H @ temp using sparse multiply
        H_temp = H_sparse @ temp
        # L^{-1} @ H_temp via triangular solve
        result = solve_triangular(L, H_temp, lower=True)
        return result
    
    def matvec(v):
        return matmat(v[:, None])[:, 0]
    
    H_op = LinearOperator((N, N), matvec=matvec, matmat=matmat, dtype=np.float64)
    
    # Run CheFSI
    vals_bella, vecs_bella, n_iters = chefsi(H_op, N, k)
    t_bella = time.perf_counter() - t_bella_start
    print(f"Bella time: {t_bella:.3f}s ({n_iters} iterations)\n")
    
    # --- Dense baseline benchmark ---
    print("Running dense eigh (scipy.linalg.eigh)...")
    H_dense = H_sparse.toarray()
    t_dense_start = time.perf_counter()
    vals_dense, vecs_dense = eigh(H_dense, S_dense)
    t_dense = time.perf_counter() - t_dense_start
    print(f"Dense time: {t_dense:.3f}s\n")
    
    # --- Compute residual norm ---
    # Residual: ||H @ C - S @ C @ diag(vals)||
    C_bella = vecs_bella
    HC = H_sparse @ C_bella
    SC = S_sparse @ C_bella
    residual = HC - SC @ np.diag(vals_bella)
    residual_norm = np.linalg.norm(residual)
    print(f"Residual norm: {residual_norm:.6e}\n")
    
    # --- Speedup ---
    speedup = t_dense / t_bella
    print(f"Speedup: {speedup:.2f}x\n")
    
    results = {
        'n_atoms': len(atoms),
        'N': N,
        'n_occ': n_occ,
        'k': k,
        'sparsity': sparsity,
        't_bella': t_bella,
        't_dense': t_dense,
        'speedup': speedup,
        'residual_norm': residual_norm,
        'n_iters': n_iters
    }
    
    return results
