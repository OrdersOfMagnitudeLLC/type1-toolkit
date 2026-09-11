// OOM Proprietary — All Rights Reserved, Orders of Magnitude LLC

#ifndef BELLA_CHEFSI_H
#define BELLA_CHEFSI_H

#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <cblas.h>
#include <lapacke.h>

template<typename Scalar = double>
class BellaCheFSI {
public:
    struct Result {
        std::vector<Scalar> eigenvalues;   // size k
        std::vector<Scalar> eigenvectors;  // size N*k, column-major
        int iterations;
        bool converged;
    };

    // MatVec: callable, signature void(const Scalar* x, Scalar* y, int N)
    //         computes y = H*x for one vector
    template<typename MatVec>
    static Result solve(
        MatVec matvec,       // H*x operator
        int N,               // matrix dimension
        int k,               // number of eigenpairs
        Scalar tol = 1e-10,
        int max_iter = 30,
        int degree_cap = 60,
        const Scalar* X0 = nullptr) {
        
        Result result;
        result.eigenvalues.resize(k);
        result.eigenvectors.resize(N * k);
        result.iterations = 0;
        result.converged = false;
        
        // Fixed parameters
        const Scalar tol_residual = Scalar(1e-6);
        
        // Scale degree with sqrt(k)
        int degree = std::max(20, int(30 * std::sqrt(std::max(k, 25) / 25.0)));
        degree = std::min(degree, degree_cap);
        
        const int min_iter_floor = std::max(5, k / 50);
        
        // --- Phase 0: spectral bounds via Lanczos ---
        const int n_lanczos = 20;
        std::mt19937 rng(42);
        std::normal_distribution<Scalar> dist(0.0, 1.0);
        
        std::vector<Scalar> v(N);
        for (int i = 0; i < N; ++i) v[i] = dist(rng);
        
        // Normalize v
        Scalar v_norm = 0;
        for (int i = 0; i < N; ++i) v_norm += v[i] * v[i];
        v_norm = std::sqrt(v_norm);
        for (int i = 0; i < N; ++i) v[i] /= v_norm;
        
        // Lanczos tridiagonal matrix
        std::vector<Scalar> alpha(n_lanczos, 0);
        std::vector<Scalar> beta(n_lanczos - 1, 0);  // off-diagonal has size n-1
        std::vector<Scalar> w(N);
        
        for (int j = 0; j < n_lanczos; ++j) {
            matvec(v.data(), w.data(), N);
            
            if (j > 0) {
                // w -= beta[j-1] * v_prev
                for (int i = 0; i < N; ++i) w[i] -= beta[j-1] * v[i];
            }
            
            // alpha[j] = v^T * w
            alpha[j] = 0;
            for (int i = 0; i < N; ++i) alpha[j] += v[i] * w[i];
            
            // w -= alpha[j] * v
            for (int i = 0; i < N; ++i) w[i] -= alpha[j] * v[i];
            
            // beta[j] = ||w|| (only for j < n_lanczos-1)
            if (j < n_lanczos - 1) {
                beta[j] = 0;
                for (int i = 0; i < N; ++i) beta[j] += w[i] * w[i];
                beta[j] = std::sqrt(beta[j]);
                
                if (beta[j] > 1e-14) {
                    // v = w / beta[j]
                    for (int i = 0; i < N; ++i) v[i] = w[i] / beta[j];
                }
            }
        }
        
        // Compute Ritz values from tridiagonal matrix using LAPACK
        // dstevd overwrites alpha with eigenvalues
        std::vector<Scalar> work(2 * n_lanczos);
        int info = LAPACKE_dstevd(LAPACK_COL_MAJOR, 'N', n_lanczos, alpha.data(), 
                                   beta.data(), alpha.data(), n_lanczos);
        
        Scalar E_min = alpha[0] * Scalar(0.98);
        Scalar E_max = alpha[n_lanczos - 1] * Scalar(1.02);
        
        // E_fermi initial
        Scalar E_fermi = E_min + (Scalar(k) / Scalar(N)) * (E_max - E_min);
        
        // Adaptive buffer
        const int buffer_init = std::max(k / 2, 20);
        const int buffer_main = std::max(k / 4, 10);
        int k_buf = k + buffer_init;
        
        // Initialize V with random vectors
        std::vector<Scalar> V(N * k_buf);
        if (X0 != nullptr) {
            std::copy(X0, X0 + N * k, V.data());
            // zero-pad the buffer extension
            std::fill(V.data() + N * k, V.data() + N * k_buf, Scalar(0));
        } else {
            for (int i = 0; i < N * k_buf; ++i) V[i] = dist(rng);
        }
        
        // QR decomposition of V using LAPACK
        std::vector<Scalar> tau(k_buf);
        std::vector<Scalar> work_qr(std::max(1, 3 * k_buf));
        info = LAPACKE_dgeqrf(LAPACK_COL_MAJOR, N, k_buf, V.data(), N, tau.data());
        info = LAPACKE_dorgqr(LAPACK_COL_MAJOR, N, k_buf, k_buf, V.data(), N, tau.data());
        
        // Locking storage
        std::vector<Scalar> locked_vecs;  // N × n_locked
        std::vector<Scalar> locked_vals;  // n_locked
        std::vector<Scalar> theta(k_buf);  // Current Ritz values (initialized, moved outside loop)
        
        // --- Main filter loop ---
        for (int iteration = 0; iteration < max_iter; ++iteration) {
            result.iterations = iteration + 1;
            
            // Shrink buffer after iteration 1
            if (iteration == 2) {
                k_buf = k + buffer_main;
                V.resize(N * k_buf);
                theta.resize(k_buf);  // Resize theta when k_buf changes
                // Re-orthogonalize
                tau.resize(k_buf);
                work_qr.resize(std::max(1, 3 * k_buf));
                info = LAPACKE_dgeqrf(LAPACK_COL_MAJOR, N, k_buf, V.data(), N, tau.data());
                info = LAPACKE_dorgqr(LAPACK_COL_MAJOR, N, k_buf, k_buf, V.data(), N, tau.data());
            }
            
            // Chebyshev filter
            Scalar center = (E_max + E_fermi) / Scalar(2.0);
            Scalar half = std::max((E_max - E_fermi) / Scalar(2.0), Scalar(1e-10));
            
            std::vector<Scalar> p0 = V;
            std::vector<Scalar> p1(N * k_buf);
            
            // p1 = (H*V - center*V) / half
            const int chunk_size = 64;
            for (int col = 0; col < k_buf; col += chunk_size) {
                int cols_this = std::min(chunk_size, k_buf - col);
                for (int c = 0; c < cols_this; ++c) {
                    int col_idx = col + c;
                    Scalar* v_col = V.data() + col_idx * N;
                    Scalar* p1_col = p1.data() + col_idx * N;
                    matvec(v_col, p1_col, N);
                    for (int i = 0; i < N; ++i) {
                        p1_col[i] = (p1_col[i] - center * v_col[i]) / half;
                    }
                }
            }
            
            // 3-term recurrence for remaining degree steps
            for (int d = 1; d < degree; ++d) {
                std::vector<Scalar> p2(N * k_buf);
                for (int col = 0; col < k_buf; col += chunk_size) {
                    int cols_this = std::min(chunk_size, k_buf - col);
                    for (int c = 0; c < cols_this; ++c) {
                        int col_idx = col + c;
                        Scalar* p1_col = p1.data() + col_idx * N;
                        Scalar* p0_col = p0.data() + col_idx * N;
                        Scalar* p2_col = p2.data() + col_idx * N;
                        matvec(p1_col, p2_col, N);
                        for (int i = 0; i < N; ++i) {
                            p2_col[i] = Scalar(2.0) * (p2_col[i] - center * p1_col[i]) / half - p0_col[i];
                        }
                    }
                }
                p0 = p1;
                p1 = p2;
            }
            V = p1;
            
            // QR decomposition
            tau.resize(k_buf);
            work_qr.resize(std::max(1, 3 * k_buf));
            info = LAPACKE_dgeqrf(LAPACK_COL_MAJOR, N, k_buf, V.data(), N, tau.data());
            info = LAPACKE_dorgqr(LAPACK_COL_MAJOR, N, k_buf, k_buf, V.data(), N, tau.data());
            
            // Rayleigh-Ritz
            std::vector<Scalar> HV(N * k_buf);
            for (int col = 0; col < k_buf; ++col) {
                matvec(V.data() + col * N, HV.data() + col * N, N);
            }
            
            // H_small = V^T * HV (k_buf × k_buf)
            std::vector<Scalar> H_small(k_buf * k_buf, 0);
            cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, k_buf, k_buf, N,
                       1.0, V.data(), N, HV.data(), N, 0.0, H_small.data(), k_buf);
            
            // Symmetrize
            for (int i = 0; i < k_buf; ++i) {
                for (int j = i + 1; j < k_buf; ++j) {
                    Scalar avg = (H_small[i + j * k_buf] + H_small[j + i * k_buf]) / Scalar(2.0);
                    H_small[i + j * k_buf] = avg;
                    H_small[j + i * k_buf] = avg;
                }
            }
            
            // Eigen decomposition of H_small
            std::vector<Scalar> theta(k_buf);
            work_qr.resize(std::max(1, 3 * k_buf));
            info = LAPACKE_dsyev(LAPACK_COL_MAJOR, 'V', 'U', k_buf, H_small.data(), k_buf, theta.data());
            
            // V = V * Y
            std::vector<Scalar> V_temp(N * k_buf);
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, N, k_buf, k_buf,
                       1.0, V.data(), N, H_small.data(), k_buf, 0.0, V_temp.data(), N);
            V = V_temp;
            
            // Locking
            int n_locked = locked_vals.size();
            int k_active = k - n_locked;
            
            // Compute residuals
            std::vector<Scalar> residuals(k);
            for (int i = 0; i < k; ++i) {
                std::vector<Scalar> vi(N);
                std::vector<Scalar> Hvi(N);
                for (int j = 0; j < N; ++j) vi[j] = V[j + i * N];
                matvec(vi.data(), Hvi.data(), N);
                
                Scalar res_norm = 0;
                for (int j = 0; j < N; ++j) {
                    Scalar diff = Hvi[j] - theta[i] * vi[j];
                    res_norm += diff * diff;
                }
                residuals[i] = std::sqrt(res_norm);
            }
            
            // Lock converged pairs
            for (int i = 0; i < k; ++i) {
                if (residuals[i] < tol_residual && n_locked < k) {
                    // Add to locked storage
                    for (int j = 0; j < N; ++j) {
                        locked_vecs.push_back(V[j + i * N]);
                    }
                    locked_vals.push_back(theta[i]);
                    n_locked++;
                }
            }
            
            // Project out locked vectors
            if (n_locked > 0 && k_active > 0) {
                // V_active = V[:, :k_active]
                // proj = locked_vecs * (locked_vecs^T * V_active)
                // V_active -= proj
                
                std::vector<Scalar> proj(N * k_active, 0);
                cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, n_locked, k_active, N,
                           1.0, locked_vecs.data(), N, V.data(), N, 0.0, H_small.data(), n_locked);
                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, N, k_active, n_locked,
                           1.0, locked_vecs.data(), N, H_small.data(), n_locked, 0.0, proj.data(), N);
                
                for (int i = 0; i < N * k_active; ++i) {
                    V[i] -= proj[i];
                }
                
                // Re-orthogonalize active subspace
                tau.resize(k_active);
                work_qr.resize(std::max(1, 3 * k_active));
                info = LAPACKE_dgeqrf(LAPACK_COL_MAJOR, N, k_active, V.data(), N, tau.data());
                info = LAPACKE_dorgqr(LAPACK_COL_MAJOR, N, k_active, k_active, V.data(), N, tau.data());
            }
            
            // E_fermi self-update
            if (k < k_buf - 1) {
                Scalar E_fermi_new = (theta[k-1] + theta[k]) / Scalar(2.0);
                Scalar alpha = (iteration < 3) ? Scalar(0.3) : Scalar(0.8);
                E_fermi = (Scalar(1.0) - alpha) * E_fermi + alpha * E_fermi_new;
            }
            
            // Convergence check
            if (n_locked == k) {
                result.converged = true;
                break;
            }
        }
        
        // Return locked eigenpairs
        if (locked_vals.size() == size_t(k)) {
            // All pairs locked - return sorted
            std::vector<int> idx(k);
            for (int i = 0; i < k; ++i) idx[i] = i;
            std::sort(idx.begin(), idx.end(), [&](int a, int b) {
                return locked_vals[a] < locked_vals[b];
            });
            
            for (int i = 0; i < k; ++i) {
                result.eigenvalues[i] = locked_vals[idx[i]];
                for (int j = 0; j < N; ++j) {
                    result.eigenvectors[j + i * N] = locked_vecs[j + idx[i] * N];
                }
            }
        } else {
            // Combine locked and active
            int n_locked = locked_vals.size();
            int n_active = k - n_locked;
            
            // Sort locked indices
            std::vector<int> idx_locked(n_locked);
            for (int i = 0; i < n_locked; ++i) idx_locked[i] = i;
            std::sort(idx_locked.begin(), idx_locked.end(), [&](int a, int b) {
                return locked_vals[a] < locked_vals[b];
            });
            
            // Copy locked values
            for (int i = 0; i < n_locked; ++i) {
                result.eigenvalues[i] = locked_vals[idx_locked[i]];
                for (int j = 0; j < N; ++j) {
                    result.eigenvectors[j + i * N] = locked_vecs[j + idx_locked[i] * N];
                }
            }
            
            // Copy active values
            for (int i = 0; i < n_active; ++i) {
                result.eigenvalues[n_locked + i] = theta[i];
                for (int j = 0; j < N; ++j) {
                    result.eigenvectors[j + (n_locked + i) * N] = V[j + i * N];
                }
            }
        }
        
        return result;
    }
};

#endif // BELLA_CHEFSI_H
