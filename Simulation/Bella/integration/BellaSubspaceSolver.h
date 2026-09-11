// Copyright (c) 2026 Orders of Magnitude LLC
// Licensed under the OOM Commercial License v1.0
// See LICENSE.md in the repository root or ofmagnitude.com

// LGPL v2.1 — integration shim only, no OOM algorithm logic

#ifndef BELLA_SUBSPACE_SOLVER_H
#define BELLA_SUBSPACE_SOLVER_H

#include <vector>
#include <mpi.h>
#include <cstring>
#include "../core/BellaCheFSI.h"

// Forward declarations for DFT-FE types
namespace dftfe {
  namespace utils {
    enum class MemorySpace;
  }
  namespace linearAlgebra {
    template<dftfe::utils::MemorySpace> class BLASWrapper;
  }
}

template<typename MemorySpace>
class operatorDFTClass;

class elpaScalaManager;

class BellaSubspaceSolver {
public:
  BellaSubspaceSolver(int num_eigenpairs, double tol, int max_iter, int degree_cap = 60)
    : d_numEigenpairs(num_eigenpairs), d_tol(tol), d_maxIter(max_iter), d_degreeCap(degree_cap) {}

  void solve(
    operatorDFTClass<dftfe::utils::MemorySpace::HOST> &operatorMatrix,
    std::shared_ptr<dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>> &BLASWrapperPtr,
    elpaScalaManager &elpaScala,
    double *eigenVectorsFlattenedArray,
    const unsigned int totalNumberWaveFunctions,
    const unsigned int localSize,
    std::vector<double> &eigenValues,
    std::vector<double> &residualNorms,
    const MPI_Comm &interBandGroupComm,
    const MPI_Comm &mpiCommDomain,
    const bool isFirstFilteringCall,
    const bool computeResidual,
    const bool useMixedPrec,
    const bool isFirstScf) {
    // Get scratch multivectors (pre-allocated, no overhead)
    auto &src_mv = operatorMatrix.getScratchFEMultivector(1, 0);
    auto &dst_mv = operatorMatrix.getScratchFEMultivector(1, 1);

    // Wrap HX() as single-vector matvec for BellaCheFSI
    auto matvec = [&](const double* x, double* y, int N) {
      // Copy raw pointer into src multivector
      std::memcpy(src_mv.data(), x, N * sizeof(double));
      
      // Apply H: dst = 1.0 * H * src + 0.0 * dst
      operatorMatrix.HX(src_mv, 1.0, 0.0, 1.0, dst_mv);
      
      // Extract result back to raw pointer
      std::memcpy(y, dst_mv.data(), N * sizeof(double));
    };

    // Call BellaCheFSI
    auto result = BellaCheFSI<double>::solve(
      matvec,
      (int)localSize,
      (int)totalNumberWaveFunctions,
      d_tol,
      d_maxIter,
      d_degreeCap);

    // Copy eigenvalues back
    std::copy(result.eigenvalues.begin(),
              result.eigenvalues.end(),
              eigenValues.begin());

    // Copy eigenvectors back to DFT-FE flattened array
    std::memcpy(eigenVectorsFlattenedArray,
                result.eigenvectors.data(),
                localSize * totalNumberWaveFunctions * sizeof(double));

    printf("BellaCheFSI: converged=%s iters=%d n_eigs=%d\n",
      result.converged ? "true" : "false",
      result.iterations,
      (int)totalNumberWaveFunctions);
  }

private:
  int d_numEigenpairs;
  double d_tol;
  int d_maxIter;
  int d_degreeCap;
};

#endif // BELLA_SUBSPACE_SOLVER_H
