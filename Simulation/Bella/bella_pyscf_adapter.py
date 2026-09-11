# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Orders of Magnitude LLC <orders@ofmagnitude.com>
"""
Bella PySCF Adapter: CheFSI eigensolver for PySCF quantum chemistry calculations.

Extracts Hamiltonian and overlap matrices from PySCF SCF calculations
and benchmarks Bella CheFSI against scipy.sparse.linalg.eigsh.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.linalg as la
import time
from scipy.linalg import cholesky, solve_triangular
from scipy.sparse.linalg import LinearOperator

from bella_chefsi_v2 import chefsi


def extract_hamiltonian(mol):
    """
    Extract Hamiltonian and overlap matrices from PySCF molecule.
    
    Parameters:
    -----------
    mol : pyscf.gto.Mole
        PySCF molecule object
    
    Returns:
    --------
    H : scipy.sparse.csr_matrix
        Hamiltonian matrix (T + V) in AO basis
    S : scipy.sparse.csr_matrix
        Overlap matrix in AO basis
    n_occ : int
        Number of occupied orbitals
    """
    from pyscf import scf
    
    # Run RHF calculation
    mf = scf.RHF(mol)
    mf.kernel()
    
    # Get AO integrals
    # Kinetic energy integral
    T = mol.intor('int1e_kin')
    # Nuclear attraction integral
    V = mol.intor('int1e_nuc')
    # Overlap integral
    S = mol.intor('int1e_ovlp')
    
    # Hamiltonian = T + V
    H = T + V
    
    # Convert to sparse matrices
    H_sparse = sp.csr_matrix(H)
    S_sparse = sp.csr_matrix(S)
    
    # Number of occupied orbitals
    n_occ = mol.nelectron // 2  # RHF: 2 electrons per orbital
    
    return H_sparse, S_sparse, n_occ


def run_bella_benchmark(mol, label):
    """
    Benchmark Bella CheFSI on PySCF molecule.
    
    Parameters:
    -----------
    mol : pyscf.gto.Mole
        PySCF molecule object
    label : str
        Label for output
    
    Returns:
    --------
    dict
        Benchmark results
    """
    print(f"\n{'='*70}")
    print(f"Benchmark: {label}")
    print(f"{'='*70}\n")
    
    # Extract Hamiltonian
    print("Extracting Hamiltonian from PySCF...")
    t_extract_start = time.perf_counter()
    H, S, n_occ = extract_hamiltonian(mol)
    t_extract = time.perf_counter() - t_extract_start
    print(f"Extraction time: {t_extract:.3f}s\n")
    
    N = H.shape[0]
    k = n_occ + 10  # Buffer of 10 unoccupied states
    
    # Compute sparsity
    nnz = H.nnz
    sparsity = 100.0 * (1.0 - nnz / (N * N))
    
    print(f"Matrix size: N={N}")
    print(f"Number of occupied orbitals: {n_occ}")
    print(f"Target eigenpairs: k={k}")
    print(f"Hamiltonian sparsity: {sparsity:.2f}%\n")
    
    # --- Bella CheFSI benchmark ---
    print("Running Bella CheFSI...")
    t_bella_start = time.perf_counter()
    
    # Cholesky decompose S (need dense for Cholesky)
    S_dense = S.toarray()
    L = cholesky(S_dense, lower=True)
    
    # Create LinearOperator for H_tilde using sparse matvec
    def matmat(V):
        # L^{-H} @ V via triangular solve
        temp = solve_triangular(L.conj().T, V, lower=False)
        # H @ temp using sparse multiply
        H_temp = H @ temp
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
    
    # --- Sparse ARPACK baseline (eigsh) ---
    print("Running sparse eigsh (ARPACK)...")
    t_eigsh_start = time.perf_counter()
    vals_eigsh, vecs_eigsh = spla.eigsh(H, k=k, M=S, which='SM')
    t_eigsh = time.perf_counter() - t_eigsh_start
    print(f"eigsh time: {t_eigsh:.3f}s\n")
    
    # --- Compute residual norm ---
    # Residual: ||H @ C - S @ C @ diag(vals)||
    C_bella = vecs_bella
    HC = H @ C_bella
    SC = S @ C_bella
    residual = HC - SC @ np.diag(vals_bella)
    residual_norm = np.linalg.norm(residual)
    print(f"Residual norm: {residual_norm:.6e}\n")
    
    # --- Speedup ---
    speedup = t_eigsh / t_bella
    print(f"Speedup vs eigsh: {speedup:.2f}x\n")
    
    results = {
        'label': label,
        'N': N,
        'n_occ': n_occ,
        'k': k,
        'sparsity': sparsity,
        't_extract': t_extract,
        't_bella': t_bella,
        't_eigsh': t_eigsh,
        'speedup': speedup,
        'residual_norm': residual_norm,
        'n_iters': n_iters
    }
    
    return results
