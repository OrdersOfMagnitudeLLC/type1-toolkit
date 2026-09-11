// Copyright (c) 2026 Orders of Magnitude LLC
// Licensed under the OOM Commercial License v1.0
// See LICENSE.md in the repository root or ofmagnitude.com

// BellaCheFSI_shim.cpp - GPL v3 shim to integrate BellaCheFSI with SPARC
// 
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <cblas.h>

// SPARC headers (C functions - need extern "C" linkage)
extern "C" {
#include "../sparc-engine/src/include/isddft.h"
#include "../sparc-engine/src/include/eigenSolver.h"
#include "../sparc-engine/src/include/hamiltonianVecRoutines.h"
}

// BellaCheFSI header (proprietary, read-only)
#include "../core/BellaCheFSI.h"

// Matvec context structure - holds all parameters needed for Hamiltonian_vectors_mult
typedef struct {
    SPARC_OBJ *pSPARC;
    int DMnd;
    int *DMVertices;
    double *Veff_loc;
    ATOM_NLOC_INFLUENCE_OBJ *Atom_Influence_nloc;
    NLOC_PROJ_OBJ *nlocProj;
    int spin;
    MPI_Comm comm;
} BellaMatvecContext;

// Matvec wrapper function - matches BellaCheFSI's expected signature
// Computes y = H*x for a single vector
static void bella_matvec(const double* x, double* y, int N, void* ctx) {
    BellaMatvecContext *context = (BellaMatvecContext*)ctx;
    
    // Call SPARC's Hamiltonian_vectors_mult with c=0.0 (pure H*x, no shift)
    // ncol=1 for single vector, ldi=ldo=N for contiguous storage
    Hamiltonian_vectors_mult(
        context->pSPARC,
        context->DMnd,
        context->DMVertices,
        context->Veff_loc,
        context->Atom_Influence_nloc,
        context->nlocProj,
        1,  // ncol = 1 (single vector)
        0.0,  // c = 0.0 (no diagonal shift)
        (double*)x,  // input vector (cast away const for SPARC API)
        N,  // ldi = N
        y,  // output vector
        N,  // ldo = N
        context->spin,
        context->comm
    );
}

// Main shim function - replaces SPARC's CheFSI with BellaCheFSI
extern "C" void BellaCheFSI_shim(SPARC_OBJ *pSPARC, int spn_i) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    // Extract parameters from SPARC_OBJ
    int N = pSPARC->Nd_d_dmcomm;  // Local dimension on this process
    int k = pSPARC->Nstates;      // Total number of eigenpairs
    int k_local = pSPARC->Nband_bandcomm;  // Local number of eigenpairs (band parallelization)
    double tol = pSPARC->TOL_SCF; // SCF tolerance
    
    // Calculate spin offset for multi-spin calculations
    int spin_offset = spn_i * pSPARC->Nd_d_dmcomm;
    
    // Populate matvec context
    BellaMatvecContext ctx;
    ctx.pSPARC = pSPARC;
    ctx.DMnd = N;
    ctx.DMVertices = pSPARC->DMVertices_dmcomm;
    ctx.Veff_loc = pSPARC->Veff_loc_dmcomm + spin_offset;
    ctx.Atom_Influence_nloc = pSPARC->Atom_Influence_nloc;
    ctx.nlocProj = pSPARC->nlocProj;
    ctx.spin = spn_i;
    ctx.comm = pSPARC->dmcomm;
    
    // Get initial guess from SPARC's orbitals
    double *Xorb = pSPARC->Xorb + spn_i * N * k_local;  // Local bands only
    
    // Calculate orbital offset for this spin/channel block
    int orb_offset = spn_i * N * k_local;  // Each spin owns N * k_local entries
    
    // Create lambda wrapper for matvec that matches BellaCheFSI's expected signature
    // BellaCheFSI expects: void(const Scalar* x, Scalar* y, int N)
    auto matvec = [&](const double* x, double* y, int N_local) {
        bella_matvec(x, y, N_local, &ctx);
    };
    
    // Call BellaCheFSI with local band count
    auto result = BellaCheFSI<double>::solve(
        matvec,
        N,
        k_local,  // Use local band count, not total k
        tol,
        50,  // max_iter (increased from 30)
        200,  // degree_cap (increased from 60)
        pSPARC->Xorb + orb_offset  // warm start from previous SCF orbitals
    );
    
    // Debug: print first 5 eigenvalues
    if (rank == 0) {
        printf("BellaCheFSI_shim: first 5 eigenvalues (k_local=%d):\n", k_local);
        for(int i=0;i<std::min(5,k_local);i++) printf("  lambda[%d] = %.10f\n", i, result.eigenvalues[i]);
    }
    
    // Write eigenvalues back to SPARC
    // Need to account for band_start_indx for global indexing
    int band_start = pSPARC->band_start_indx;
    int lambda_offset = spn_i * k + band_start;
    for (int i = 0; i < k_local; i++) {
        pSPARC->lambda[lambda_offset + i] = result.eigenvalues[i];
    }
    
    // Write eigenvectors back to SPARC in band-major (column) layout.
    // pSPARC->Xorb stores local bands contiguously: Xorb[grid + band*N]
    // result.eigenvectors is returned in the same band-major ordering.
    if (rank == 0) {
        printf("BellaCheFSI_shim: writing eigenvectors, N=%d k_local=%d orb_offset=%d total_size=%d\n",
               N, k_local, orb_offset, N * k_local);
    }
    
    for (int band = 0; band < k_local; ++band) {
        for (int grid = 0; grid < N; ++grid) {
            pSPARC->Xorb[orb_offset + grid + band * N] = result.eigenvectors[grid + band * N];
        }
    }
    
    // Re-orthogonalize Xorb via modified Gram-Schmidt 
    // to guard against degenerate state collapse
    double *X = pSPARC->Xorb + orb_offset;
    for (int j = 0; j < k_local; j++) {
        // Subtract projections onto previous vectors
        for (int i = 0; i < j; i++) {
            double dot = cblas_ddot(N, X + i*N, 1, X + j*N, 1);
            cblas_daxpy(N, -dot, X + i*N, 1, X + j*N, 1);
        }
        // Normalize
        double norm = cblas_dnrm2(N, X + j*N, 1);
        if (norm > 1e-14) cblas_dscal(N, 1.0/norm, X + j*N, 1);
    }
    
    if (rank == 0) {
        printf("BellaCheFSI_shim: converged=%s iters=%d k=%d N=%d\n",
               result.converged ? "true" : "false",
               result.iterations,
               k,
               N);
    }
}
