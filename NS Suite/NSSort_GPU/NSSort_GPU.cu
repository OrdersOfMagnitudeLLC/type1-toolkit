// Copyright (c) 2026 Orders of Magnitude LLC
// Licensed under the OOM Commercial License v1.0
// See LICENSE.md in the repository root or ofmagnitude.com

/*
 * NSSort GPU: two-level MSD radix sort with five-way input routing.
 * Ported from the NSSort CPU engine (NS/NSSort/NSSort.hpp): PATH_SORTED,
 * PATH_REVERSE, PATH_NEARLY_SORTED, PATH_COUNTING and PATH_GENERAL, with a
 * size-based REFINE split (single-block fused vs. multi-block pipeline).
 * Keys are 64-bit (int64_t) to match the CPU engine's element type.
 */

#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/merge.h>
#include <thrust/execution_policy.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <climits>
#include <chrono>
#include <random>
#include <cmath>
#include <algorithm>
#include <vector>

// ---------------------------------------------------------------------------
// Routing / sizing constants
// ---------------------------------------------------------------------------
constexpr int      SAMPLE_N            = 256;
constexpr int      COARSE_BITS         = 8;
constexpr int      COARSE_BINS         = 1 << COARSE_BITS;   // 256
constexpr float    COUNTING_RANGE_FRAC = 0.10f;
constexpr int      WINDOW              = 2048;   // NS repair window / dirty-bitmap granularity
constexpr int      NS_THRESH           = SAMPLE_N / 16;
constexpr int      NS_BAIL_DIVISOR     = 32;
constexpr uint64_t RESCALE_THRESHOLD   = (1ull << 28);

// Two-level MSD split on a 64-bit key: top byte selects the coarse sector,
// the next byte selects the REFINE sub-bucket. The remaining 48 bits are
// resolved by DeviceSegmentedSort within each leaf bucket.
constexpr int COARSE_SHIFT = 56;
constexpr int REFINE_SHIFT = 48;

// Multi-block REFINE: sectors are split into REFINE_CHUNK_SIZE-element
// blocks. REFINE_MAX_TOTAL_BLOCKS bounds the sum of blocks across all
// 256 sectors at 100M elements.
constexpr int REFINE_CHUNK_SIZE       = 1 << 20;
constexpr int REFINE_MAX_TOTAL_BLOCKS = 8192;

// Sectors with count >= SECTOR_MB_THRESHOLD use the multi-block REFINE path;
// smaller sectors use the single-block fused path.
constexpr uint32_t SECTOR_MB_THRESHOLD = 2000000;

// ---------------------------------------------------------------------------
// Detection: O(256 samples) routing signal (Section 1.1 fast classification)
// ---------------------------------------------------------------------------
struct ViolationResult {
    int      desc;      // count of descending violations (arr[i] > arr[i+1])
    int      asc;       // count of ascending violations (arr[i] < arr[i+1])
    uint64_t smin;      // sample min value (as uint64 after sign-flip)
    uint64_t smax;      // sample max value (as uint64 after sign-flip)
    int      disp_est;  // displacement estimate: max forward scan distance for a violation
};

// 256 threads, one block. Each thread samples one element at stride n/256,
// tracks sample min/max, adjacent-pair violation counts, and a displacement
// estimate (forward scan distance within the sample for each violation).
__global__ void detection_kernel(const int64_t* __restrict__ input, ViolationResult* result, int n) {
    int tid = threadIdx.x;
    int stride = n / 256;
    if (stride < 1) stride = 1;
    int idx = tid * stride;

    __shared__ int      local_desc;
    __shared__ int      local_asc;
    __shared__ uint64_t local_smin;
    __shared__ uint64_t local_smax;
    __shared__ int      local_disp;
    __shared__ uint64_t s_samples[256];  // sampled values for disp_est scan

    if (tid == 0) {
        local_desc = 0;
        local_asc  = 0;
        local_smin = 0xFFFFFFFFFFFFFFFFull;
        local_smax = 0ull;
        local_disp = 0;
    }
    __syncthreads();

    // Load sample (sign-flipped for uint ordering)
    uint64_t val = (idx < n) ? ((uint64_t)input[idx] ^ 0x8000000000000000ull) : 0xFFFFFFFFFFFFFFFFull;
    s_samples[tid] = val;
    __syncthreads();

    // Update min/max (cast to unsigned long long: uint64_t is `unsigned long`
    // on LP64 targets, but CUDA's 64-bit atomics are defined on `unsigned
    // long long`; the cast keeps the call unambiguous.)
    atomicMin((unsigned long long*)&local_smin, (unsigned long long)val);
    atomicMax((unsigned long long*)&local_smax, (unsigned long long)val);

    // Violation check against next sample
    if (tid < 255) {
        uint64_t cur  = s_samples[tid];
        uint64_t nxt  = s_samples[tid + 1];
        if (cur > nxt) atomicAdd(&local_desc, 1);
        if (cur < nxt) atomicAdd(&local_asc,  1);
    }
    __syncthreads();

    // Displacement estimate: for each desc violation (cur > nxt),
    // scan forward to find where cur would fit; distance * stride.
    if (tid < 255) {
        uint64_t cur = s_samples[tid];
        uint64_t nxt = s_samples[tid + 1];
        if (cur > nxt) {
            int dist = 1;
            for (int k = tid + 2; k < 256 && s_samples[k] < cur; ++k) ++dist;
            atomicMax(&local_disp, dist * stride);
        }
    }
    __syncthreads();

    if (tid == 0) {
        result->desc     = local_desc;
        result->asc      = local_asc;
        result->smin     = local_smin;
        result->smax     = local_smax;
        result->disp_est = local_disp;
    }
}

// ---------------------------------------------------------------------------
// Verification: one grid-stride pass; produces violation count, sorted flag,
// rev flag, and a dirty bitmap (one bit per WINDOW-sized window) used by the
// nearly-sorted path to locate the regions that need repair.
// ---------------------------------------------------------------------------
constexpr int DIRTY_WORDS = 8192;  // covers n up to WINDOW * 32 * DIRTY_WORDS

struct VerifyResult {
    int  violations;    // total adjacent violations (arr[i] > arr[i+1])
    int  sorted_flag;   // 1 if violations == 0
    int  rev_flag;      // 1 if zero ascending pairs (all desc)
    int  viol_count;    // alias: total violation count (same as violations)
};

// d_dirty: DIRTY_WORDS uint32 words, each bit covers one WINDOW-element window.
__global__ void verify_kernel(const int64_t* __restrict__ arr, int n,
                               VerifyResult* __restrict__ out,
                               uint32_t* __restrict__ d_dirty) {
    __shared__ int s_viols;
    __shared__ int s_asc;
    if (threadIdx.x == 0) { s_viols = 0; s_asc = 0; }
    __syncthreads();

    int viols = 0, asc = 0;
    for (int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
         i < n - 1;
         i += (int)(gridDim.x * blockDim.x)) {
        if (arr[i] > arr[i + 1]) {
            ++viols;
            // Mark both windows that touch this violation boundary.
            // If the pair straddles a window edge (i == w*WINDOW-1, i+1 == w*WINDOW),
            // the right-element's window must also be dirtied so Regime B extracts it.
            int win_l = i / WINDOW;
            int win_r = (i + 1) / WINDOW;
            atomicOr(&d_dirty[win_l >> 5], 1u << (win_l & 31));
            if (win_r != win_l)
                atomicOr(&d_dirty[win_r >> 5], 1u << (win_r & 31));
        }
        else if (arr[i] < arr[i + 1]) ++asc;
    }
    if (viols) atomicAdd(&s_viols, viols);
    if (asc)   atomicAdd(&s_asc,   asc);
    __syncthreads();
    if (threadIdx.x == 0) {
        if (s_viols) atomicAdd(&out->violations, s_viols);
        if (s_asc)   atomicAdd(&out->rev_flag,   s_asc);
    }
}

// verify_finalize: called once after verify_kernel.
// rev_flag == 0 means zero ascending pairs globally -> truly reverse-sorted.
// sorted_flag == 1 means zero violation pairs globally -> truly sorted.
__global__ void verify_finalize_kernel(VerifyResult* out) {
    if (threadIdx.x == 0) {
        out->sorted_flag = (out->violations == 0) ? 1 : 0;
        // rev_flag currently holds global asc count; convert to boolean
        out->rev_flag    = (out->rev_flag == 0 && out->violations > 0) ? 1 : 0;
    }
}

// ---------------------------------------------------------------------------
// Coarse histogram: top-byte (bits 56-63) bin of the sign-flipped 64-bit key
// ---------------------------------------------------------------------------
__device__ __forceinline__ int coarse_bin(uint64_t x) {
    return (int)(x >> COARSE_SHIFT);
}

__global__ void hist_kernel(const uint64_t* arr, size_t n, uint32_t* g_hist) {
    __shared__ uint32_t s_hist[COARSE_BINS];
    for (int i = threadIdx.x; i < COARSE_BINS; i += blockDim.x) s_hist[i] = 0;
    __syncthreads();
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += (size_t)gridDim.x * blockDim.x)
        atomicAdd(&s_hist[coarse_bin(arr[i])], 1u);
    __syncthreads();
    for (int i = threadIdx.x; i < COARSE_BINS; i += blockDim.x)
        atomicAdd(&g_hist[i], s_hist[i]);
}

// ---------------------------------------------------------------------------
// FAST PATH: DO NOT MODIFY
// REVERSE kernel: trivial parallel reversal
// Each thread: output[i] = input[n-1-i]
// ---------------------------------------------------------------------------
__global__ void reverse_kernel(const int64_t* __restrict__ input, int64_t* __restrict__ output, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        output[i] = input[n - 1 - i];
    }
}

// ---------------------------------------------------------------------------
// FAST PATH: DO NOT MODIFY
// PATH_COUNTING: parallel counting sort for small-range data.
// Kernel 1: block-local shared-mem histogram over grid-stride input, values
// pre-shifted to [0, range) by the caller so they can index directly.
// ---------------------------------------------------------------------------
__global__ void count_kernel(const int64_t* __restrict__ input, int* __restrict__ counts, int n, int range) {
    extern __shared__ int local_counts[];

    int tid = threadIdx.x;
    bool smem_active = (range <= 12288);

    if (smem_active) {
        for (int i = tid; i < range; i += blockDim.x)
            local_counts[i] = 0;
        __syncthreads();
    }

    if (smem_active) {
        for (int i = blockIdx.x * blockDim.x + tid;
             i < n;
             i += gridDim.x * blockDim.x)
            atomicAdd(&local_counts[(int)input[i]], 1);
        __syncthreads();
        for (int i = tid; i < range; i += blockDim.x)
            if (local_counts[i] > 0)
                atomicAdd(&counts[i], local_counts[i]);
    } else {
        for (int i = blockIdx.x * blockDim.x + tid;
             i < n;
             i += gridDim.x * blockDim.x)
            atomicAdd(&counts[(int)input[i]], 1);
    }
}

// Kernel 2: fill contiguous ranges (no atomics). One block per unique value v.
__global__ void fill_kernel(int64_t* __restrict__ output, const int* __restrict__ offsets,
                             const int* __restrict__ counts, int num_unique, int64_t value_offset = 0) {
    int v = blockIdx.x;
    if (v < num_unique && counts[v] > 0) {
        int start = offsets[v];
        int count = counts[v];
        for (int i = threadIdx.x; i < count; i += blockDim.x) {
            output[start + i] = (int64_t)v + value_offset;
        }
    }
}

// ---------------------------------------------------------------------------
// CUDA error checking
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

// ---------------------------------------------------------------------------
// GPU memory manager: allocate once, reuse across distributions
// ---------------------------------------------------------------------------
class GPUMemory {
public:
    int64_t* d_input;
    int64_t* d_output;
    int*     d_counts;   // counting sort: per-value counts   [0, max_range)
    int*     d_offsets;  // counting sort: exclusive-prefix-sum offsets
    size_t   max_n;
    size_t   max_range;

    void*  d_scan_storage;      // CUB DeviceScan::ExclusiveSum temp
    size_t scan_storage_bytes;
    void*  d_sort_storage;      // CUB DeviceRadixSort::SortKeys temp
    size_t sort_storage_bytes;
    void*  d_reduce_storage;    // CUB DeviceReduce::Min/Max temp
    size_t reduce_storage_bytes;

    ViolationResult* d_viol_result;
    int64_t* d_min_val;
    int64_t* d_max_val;
    int*     d_inversion_count;  // large-displacement flag for ns_max_disp_kernel
    int*     d_all_equal;       // zero-sampling check flag

    // Verification: violation/sorted/reverse flags + dirty bitmap
    VerifyResult* d_verify_result;
    uint32_t*     d_dirty;         // DIRTY_WORDS uint32 words

    // Per-sector min/max (256 sectors, uint64 keys)
    uint64_t* d_sec_minmax;   // [2*s]=min, [2*s+1]=max for sector s

    // Nearly-sorted (Regime B) buffers
    int64_t*  d_ns_tmp;               // n-element extraction/merge buffer
    uint8_t*  d_ns_flags;             // n-element flag array for DeviceSelect
    void*     d_ns_select_storage;    // CUB DeviceSelect::Flagged temp
    size_t    ns_select_storage_bytes;
    int*      d_ns_num_selected;

    // General sort: coarse histogram, base/cursor, scatter + refine buffers
    uint32_t* d_g_hist;       // 256 coarse histogram
    uint32_t* d_g_base;       // 256 exclusive-prefix-sum of g_hist
    uint32_t* d_g_cursor;     // 256 atomic write cursors (copy of g_base)
    uint64_t* d_scatter_out;  // n-element coarse scatter output
    uint64_t* d_refine_tmp;   // n-element REFINE scatter temp
    // Fused REFINE segment table (256 sectors * 256 sub-bins = 65536 max)
    int*      d_fused_seg_starts;
    int*      d_fused_seg_ends;
    int*      d_fused_num_segs;
    void*     d_segsort_storage;   // CUB DeviceSegmentedSort::SortKeys temp
    size_t    segsort_storage_bytes;

    // Multi-block REFINE arrays (large-sector path)
    uint64_t* d_g_sec_minmax_mb;  // 256*2: per-sector min/max
    uint32_t* d_g_sub_hist;       // 256*256: per-sector sub-histograms
    uint32_t* d_g_sub_scan;       // 256*256: per-sector sub-scans
    uint32_t* d_g_sub_cursor;     // 256*256: per-sector mutable scatter cursors
    int*      d_blk_sector;       // REFINE_MAX_TOTAL_BLOCKS: block->sector
    int*      d_blk_chunk_off;    // REFINE_MAX_TOTAL_BLOCKS: block->chunk start
    int*      d_blk_chunk_len;    // REFINE_MAX_TOTAL_BLOCKS: block->chunk length
    // Single-block REFINE: occupied small-sector indices
    int*      d_small_sector_list;  // up to COARSE_BINS entries

    explicit GPUMemory(size_t n) : max_n(n), max_range(1000000) {
        CUDA_CHECK(cudaMalloc(&d_input,   n * sizeof(int64_t)));
        CUDA_CHECK(cudaMalloc(&d_output,  n * sizeof(int64_t)));
        CUDA_CHECK(cudaMalloc(&d_counts,  max_range * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_offsets, max_range * sizeof(int)));

        scan_storage_bytes = 0;
        cub::DeviceScan::ExclusiveSum(
            nullptr, scan_storage_bytes,
            d_counts, d_offsets, max((int)max_range, 65536));
        CUDA_CHECK(cudaMalloc(&d_scan_storage, scan_storage_bytes));

        sort_storage_bytes = 0;
        cub::DeviceRadixSort::SortKeys(
            nullptr, sort_storage_bytes,
            d_input, d_output, (int)n);
        CUDA_CHECK(cudaMalloc(&d_sort_storage, sort_storage_bytes));

        reduce_storage_bytes = 0;
        cub::DeviceReduce::Min(nullptr, reduce_storage_bytes, d_input, d_input, (int)n);
        CUDA_CHECK(cudaMalloc(&d_reduce_storage, reduce_storage_bytes));

        CUDA_CHECK(cudaMalloc(&d_viol_result, sizeof(ViolationResult)));
        CUDA_CHECK(cudaMalloc(&d_min_val, sizeof(int64_t)));
        CUDA_CHECK(cudaMalloc(&d_max_val, sizeof(int64_t)));
        CUDA_CHECK(cudaMalloc(&d_inversion_count, sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_all_equal, sizeof(int)));

        CUDA_CHECK(cudaMalloc(&d_verify_result, sizeof(VerifyResult)));
        CUDA_CHECK(cudaMalloc((void**)&d_dirty, DIRTY_WORDS * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc((void**)&d_sec_minmax, 512 * sizeof(uint64_t)));

        CUDA_CHECK(cudaMalloc((void**)&d_ns_tmp,          n * sizeof(int64_t)));
        CUDA_CHECK(cudaMalloc((void**)&d_ns_flags,        n * sizeof(uint8_t)));
        CUDA_CHECK(cudaMalloc((void**)&d_ns_num_selected, sizeof(int)));
        ns_select_storage_bytes = 0;
        cub::DeviceSelect::Flagged(nullptr, ns_select_storage_bytes,
            d_input, d_ns_flags, d_ns_tmp, d_ns_num_selected, (int)n);
        CUDA_CHECK(cudaMalloc(&d_ns_select_storage, ns_select_storage_bytes));

        CUDA_CHECK(cudaMalloc((void**)&d_g_hist,      COARSE_BINS * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc((void**)&d_g_base,      COARSE_BINS * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc((void**)&d_g_cursor,    COARSE_BINS * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc((void**)&d_scatter_out, n           * sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc((void**)&d_refine_tmp,  n           * sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc((void**)&d_fused_seg_starts, 65536 * sizeof(int)));
        CUDA_CHECK(cudaMalloc((void**)&d_fused_seg_ends,   65536 * sizeof(int)));
        CUDA_CHECK(cudaMalloc((void**)&d_fused_num_segs,   sizeof(int)));

        segsort_storage_bytes = 0;
        cub::DeviceSegmentedSort::SortKeys(
            nullptr, segsort_storage_bytes,
            (uint64_t*)nullptr, (uint64_t*)nullptr, (int)n,
            65536, (const int*)nullptr, (const int*)nullptr);
        CUDA_CHECK(cudaMalloc(&d_segsort_storage, segsort_storage_bytes));

        CUDA_CHECK(cudaMalloc((void**)&d_g_sec_minmax_mb, 256 * 2 * sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc((void**)&d_g_sub_hist,      256 * 256 * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc((void**)&d_g_sub_scan,      256 * 256 * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc((void**)&d_g_sub_cursor,    256 * 256 * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc((void**)&d_blk_sector,    REFINE_MAX_TOTAL_BLOCKS * sizeof(int)));
        CUDA_CHECK(cudaMalloc((void**)&d_blk_chunk_off, REFINE_MAX_TOTAL_BLOCKS * sizeof(int)));
        CUDA_CHECK(cudaMalloc((void**)&d_blk_chunk_len, REFINE_MAX_TOTAL_BLOCKS * sizeof(int)));
        CUDA_CHECK(cudaMalloc((void**)&d_small_sector_list, COARSE_BINS * sizeof(int)));
    }

    ~GPUMemory() {
        CUDA_CHECK(cudaFree(d_input));
        CUDA_CHECK(cudaFree(d_output));
        CUDA_CHECK(cudaFree(d_counts));
        CUDA_CHECK(cudaFree(d_offsets));
        CUDA_CHECK(cudaFree(d_scan_storage));
        CUDA_CHECK(cudaFree(d_sort_storage));
        CUDA_CHECK(cudaFree(d_reduce_storage));
        CUDA_CHECK(cudaFree(d_viol_result));
        CUDA_CHECK(cudaFree(d_min_val));
        CUDA_CHECK(cudaFree(d_max_val));
        CUDA_CHECK(cudaFree(d_inversion_count));
        CUDA_CHECK(cudaFree(d_all_equal));
        CUDA_CHECK(cudaFree(d_verify_result));
        CUDA_CHECK(cudaFree(d_dirty));
        CUDA_CHECK(cudaFree(d_sec_minmax));
        CUDA_CHECK(cudaFree(d_ns_tmp));
        CUDA_CHECK(cudaFree(d_ns_flags));
        CUDA_CHECK(cudaFree(d_ns_num_selected));
        CUDA_CHECK(cudaFree(d_ns_select_storage));
        CUDA_CHECK(cudaFree(d_g_hist));
        CUDA_CHECK(cudaFree(d_g_base));
        CUDA_CHECK(cudaFree(d_g_cursor));
        CUDA_CHECK(cudaFree(d_scatter_out));
        CUDA_CHECK(cudaFree(d_refine_tmp));
        CUDA_CHECK(cudaFree(d_fused_seg_starts));
        CUDA_CHECK(cudaFree(d_fused_seg_ends));
        CUDA_CHECK(cudaFree(d_fused_num_segs));
        CUDA_CHECK(cudaFree(d_segsort_storage));
        CUDA_CHECK(cudaFree(d_g_sec_minmax_mb));
        CUDA_CHECK(cudaFree(d_g_sub_hist));
        CUDA_CHECK(cudaFree(d_g_sub_scan));
        CUDA_CHECK(cudaFree(d_g_sub_cursor));
        CUDA_CHECK(cudaFree(d_blk_sector));
        CUDA_CHECK(cudaFree(d_blk_chunk_off));
        CUDA_CHECK(cudaFree(d_blk_chunk_len));
        CUDA_CHECK(cudaFree(d_small_sector_list));
    }
};

// ---------------------------------------------------------------------------
// FAST PATH: DO NOT MODIFY
// PATH_COUNTING: parallel counting sort. Assumes values pre-shifted to
// [0, num_unique). 3-kernel approach: count (shared mem) -> CUB scan -> fill.
// ---------------------------------------------------------------------------
float gpu_counting_sort(int64_t* h_arr, int n, GPUMemory& mem, int num_unique,
                         int output_offset = 0, int64_t value_offset = 0) {
    if (h_arr != nullptr) {
        CUDA_CHECK(cudaMemcpy(mem.d_input, h_arr, n * sizeof(int64_t), cudaMemcpyHostToDevice));
    }

    CUDA_CHECK(cudaMemset(mem.d_counts, 0, num_unique * sizeof(int)));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    int threads = 256;
    int sm_count = 0;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);
    int max_blocks = (n + threads - 1) / threads;
    int blocks = min(max_blocks, sm_count * 4);

    CUDA_CHECK(cudaEventRecord(start));

    size_t shared_mem_size = (num_unique <= 12288) ? (size_t)num_unique * sizeof(int) : 0;
    count_kernel<<<blocks, threads, shared_mem_size>>>(mem.d_input, mem.d_counts, n, num_unique);
    CUDA_CHECK(cudaGetLastError());

    cub::DeviceScan::ExclusiveSum(mem.d_scan_storage, mem.scan_storage_bytes, mem.d_counts, mem.d_offsets, num_unique);

    fill_kernel<<<num_unique, threads>>>(mem.d_output + output_offset, mem.d_offsets, mem.d_counts, num_unique, value_offset);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    if (h_arr != nullptr) {
        CUDA_CHECK(cudaMemcpy(h_arr, mem.d_output + output_offset, n * sizeof(int64_t), cudaMemcpyDeviceToHost));
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return ms;
}

// ---------------------------------------------------------------------------
// Zero sampling kernel: quick check for all-equal arrays
// ---------------------------------------------------------------------------
__global__ void zero_sample_kernel(const int64_t* __restrict__ d_input, int* __restrict__ d_all_equal, int n) {
    int tid = threadIdx.x;
    if (tid >= 32) return;
    
    // Read 32 spread indices across the array
    int64_t val = d_input[tid * (n / 32)];
    __shared__ int64_t s_vals[32];
    s_vals[tid] = val;
    __syncthreads();
    
    // Check if all values are equal
    if (tid == 0) {
        int64_t first = s_vals[0];
        int all_eq = 1;
        for (int i = 1; i < 32; i++) {
            if (s_vals[i] != first) {
                all_eq = 0;
                break;
            }
        }
        *d_all_equal = all_eq;
    }
}

// ---------------------------------------------------------------------------
// Sign-flip: XOR the sign bit so natural unsigned ordering of the bit
// pattern matches signed ordering (standard radix-sort-on-signed-ints trick).
// ---------------------------------------------------------------------------
__global__ void flip_sign_bit_kernel(int64_t* d_data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        d_data[i] ^= INT64_MIN;
    }
}

// ---------------------------------------------------------------------------
// Fused REFINE kernel: one block per coarse sector (gridDim.x==COARSE_BINS,
// blockDim.x==256). Empty sectors (g_hist[s]==0) exit immediately.
//   1. Sub-histogram over the sector's own value range (rescaled to 256 bins).
//   2. cub::BlockScan exclusive-sum -> relative offsets within the sector.
//   3. Non-empty sub-bins emit a segment into d_fused_seg_starts/ends.
//   4. Sub-scatter into refine_tmp using the scanned offsets as cursors.
// ---------------------------------------------------------------------------

// Forward declaration (defined below)
float gpu_nssort_general(int64_t* h_arr, int n, int64_t h_min, int64_t h_max, GPUMemory& mem);

// File-scope functor: inverts a uint8 flag (0->1, 1->0) for CUB TransformInputIterator
struct InvertFlag {
    __host__ __device__ uint8_t operator()(uint8_t f) const { return f ^ 1u; }
};

// ---------------------------------------------------------------------------
// RESCALE variant: activated when (smax - smin) < RESCALE_THRESHOLD, i.e. the
// full array's value range is compressed. bin = (x-smin)*COARSE_BINS/range.
// ---------------------------------------------------------------------------
__device__ __forceinline__ int coarse_bin_rescale(uint64_t x,
                                                   uint64_t smin, uint64_t range) {
    return (int)((x - smin) * COARSE_BINS / range);
}

__global__ void hist_kernel_rescale(const uint64_t* arr, size_t n, uint32_t* g_hist,
                                     uint64_t smin, uint64_t range) {
    __shared__ uint32_t s_hist[COARSE_BINS];
    for (int i = threadIdx.x; i < COARSE_BINS; i += blockDim.x) s_hist[i] = 0;
    __syncthreads();
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += (size_t)gridDim.x * blockDim.x)
        atomicAdd(&s_hist[coarse_bin_rescale(arr[i], smin, range)], 1u);
    __syncthreads();
    for (int i = threadIdx.x; i < COARSE_BINS; i += blockDim.x)
        atomicAdd(&g_hist[i], s_hist[i]);
}

// ---------------------------------------------------------------------------
// CHUNK scatter: 256-thread block, epoch-barrier drain.
// Per epoch (one element per thread): each thread computes its bin, stages
// into s_stage[bin][slot] (or writes directly to global on overflow), then
// after a barrier thread t drains bin t's staged elements to global. A
// second barrier ensures the reset is visible before the next epoch.
// BYTE_SHIFT selects the coarse bin's byte (COARSE_SHIFT = top byte).
// CHNK_SLOTS=16 (vs. 32 for the 32-bit key version): s_stage now holds
// 8-byte keys, so 16 slots keeps static SMEM at 256*16*8 + 256*4 = 33792
// bytes, under the 48KB default budget. The overflow path (direct global
// write when a bin's per-epoch count exceeds CHNK_SLOTS) is correct for
// any slot count, so this only trades off overflow frequency.
// ---------------------------------------------------------------------------
constexpr int CHNK_SLOTS = 16;

template<int BYTE_SHIFT>
__global__ void chunk_scatter_kernel(const uint64_t* __restrict__ arr, size_t n,
                                      uint64_t* __restrict__ out,
                                      uint32_t* __restrict__ g_cursor) {
    __shared__ uint64_t s_stage[COARSE_BINS][CHNK_SLOTS];
    __shared__ uint32_t s_count[COARSE_BINS];

    const int tid = threadIdx.x;   // 0..255, one-to-one with bins in drain phase

    s_count[tid] = 0;
    __syncthreads();

    for (size_t base = (size_t)blockIdx.x * blockDim.x;
         base < n;
         base += (size_t)gridDim.x * blockDim.x) {

        size_t idx = base + tid;
        if (idx < n) {
            uint64_t x = arr[idx];
            int b = (int)((x >> BYTE_SHIFT) & 0xFFu);
            uint32_t slot = atomicAdd(&s_count[b], 1u);
            if (slot < (uint32_t)CHNK_SLOTS) {
                s_stage[b][slot] = x;
            } else {
                atomicSub(&s_count[b], 1u);
                uint32_t dst = atomicAdd(&g_cursor[b], 1u);
                out[dst] = x;
            }
        }

        __syncthreads();

        {
            uint32_t tail = s_count[tid];
            if (tail > 0) {
                if (tail > (uint32_t)CHNK_SLOTS) tail = (uint32_t)CHNK_SLOTS;
                uint32_t dst = atomicAdd(&g_cursor[tid], tail);
                for (uint32_t k = 0; k < tail; ++k)
                    out[dst + k] = s_stage[tid][k];
                s_count[tid] = 0;
            }
        }

        __syncthreads();
    }
}

// Rescale variant: identical logic, uses coarse_bin_rescale() for bin.
template<int DUMMY>
__global__ void chunk_scatter_kernel_rescale(const uint64_t* __restrict__ arr, size_t n,
                                              uint64_t* __restrict__ out,
                                              uint32_t* __restrict__ g_cursor,
                                              uint64_t smin, uint64_t range) {
    __shared__ uint64_t s_stage[COARSE_BINS][CHNK_SLOTS];
    __shared__ uint32_t s_count[COARSE_BINS];

    const int tid = threadIdx.x;

    s_count[tid] = 0;
    __syncthreads();

    for (size_t base = (size_t)blockIdx.x * blockDim.x;
         base < n;
         base += (size_t)gridDim.x * blockDim.x) {

        size_t idx = base + tid;
        if (idx < n) {
            uint64_t x = arr[idx];
            int b = coarse_bin_rescale(x, smin, range);
            uint32_t slot = atomicAdd(&s_count[b], 1u);
            if (slot < (uint32_t)CHNK_SLOTS) {
                s_stage[b][slot] = x;
            } else {
                atomicSub(&s_count[b], 1u);
                uint32_t dst = atomicAdd(&g_cursor[b], 1u);
                out[dst] = x;
            }
        }

        __syncthreads();

        {
            uint32_t tail = s_count[tid];
            if (tail > 0) {
                if (tail > (uint32_t)CHNK_SLOTS) tail = (uint32_t)CHNK_SLOTS;
                uint32_t dst = atomicAdd(&g_cursor[tid], tail);
                for (uint32_t k = 0; k < tail; ++k)
                    out[dst + k] = s_stage[tid][k];
                s_count[tid] = 0;
            }
        }

        __syncthreads();
    }
}

// Per-sector min/max reduction: 256 blocks (one per sector).
// Reads scatter_out[g_base[s]..g_base[s]+g_hist[s]), writes d_sec_minmax[2s]/[2s+1].

// ===========================================================================
// Single-block REFINE: indirection wrappers for small-sector path
//
// These wrap sec_minmax_kernel / refine_fused_rescale_kernel logic but use
// a sector_list[] indirection so the grid can contain only occupied small
// sectors (blockIdx.x indexes into sector_list, not the absolute sector).
// They write to the same d_sec_minmax / refine_tmp / seg_starts/ends buffers
// as the original kernels, so the downstream DeviceSegmentedSort is unchanged.
// ===========================================================================

// sec_minmax_sb_kernel: like sec_minmax_kernel but blockIdx.x -> sector_list[blockIdx.x].
__global__ void sec_minmax_sb_kernel(
    const uint64_t* __restrict__ scatter_out,
    const uint32_t* __restrict__ g_hist,
    const uint32_t* __restrict__ g_base,
    const int*      __restrict__ sector_list,   // [num_small_occupied]
    uint64_t*       __restrict__ d_sec_minmax)
{
    const int s   = sector_list[blockIdx.x];
    const int tid = threadIdx.x;
    uint32_t sec_n   = g_hist[s];
    uint32_t sec_off = g_base[s];
    if (sec_n == 0) {
        if (tid == 0) { d_sec_minmax[2*s] = 0xFFFFFFFFFFFFFFFFull; d_sec_minmax[2*s+1] = 0ull; }
        return;
    }
    __shared__ uint64_t s_min, s_max;
    if (tid == 0) { s_min = 0xFFFFFFFFFFFFFFFFull; s_max = 0ull; }
    __syncthreads();
    const uint64_t* sec_data = scatter_out + sec_off;
    uint64_t lmin = 0xFFFFFFFFFFFFFFFFull, lmax = 0ull;
    for (uint32_t i = tid; i < sec_n; i += blockDim.x) {
        uint64_t x = sec_data[i];
        if (x < lmin) lmin = x;
        if (x > lmax) lmax = x;
    }
    atomicMin((unsigned long long*)&s_min, (unsigned long long)lmin);
    atomicMax((unsigned long long*)&s_max, (unsigned long long)lmax);
    __syncthreads();
    if (tid == 0) { d_sec_minmax[2*s] = s_min; d_sec_minmax[2*s+1] = s_max; }
}

// refine_fused_sb_kernel: like refine_fused_rescale_kernel but blockIdx.x -> sector_list[blockIdx.x].
__global__ void refine_fused_sb_kernel(
    const uint64_t* __restrict__ scatter_out,
    uint64_t*       __restrict__ refine_tmp,
    const uint32_t* __restrict__ g_hist,
    const uint32_t* __restrict__ g_base,
    const uint64_t* __restrict__ d_sec_minmax,
    const int*      __restrict__ sector_list,   // [num_small_occupied]
    int*            __restrict__ d_seg_starts,
    int*            __restrict__ d_seg_ends,
    int*            __restrict__ d_num_segs)
{
    const int s   = sector_list[blockIdx.x];
    const int tid = threadIdx.x;
    uint32_t sec_n   = g_hist[s];
    uint32_t sec_off = g_base[s];
    if (sec_n == 0) return;

    uint64_t sec_min   = d_sec_minmax[2*s];
    uint64_t sec_max   = d_sec_minmax[2*s+1];
    uint64_t sec_range = sec_max - sec_min + 1;

    __shared__ uint32_t s_sh[256];
    __shared__ uint32_t s_sc[256];
    __shared__ uint32_t s_cursor[256];

    s_sh[tid] = 0;
    __syncthreads();

    const uint64_t* sec_data = scatter_out + sec_off;
    for (uint32_t i = tid; i < sec_n; i += 256) {
        uint64_t x = sec_data[i];
        int b = (int)((x - sec_min) * 256 / sec_range);
        atomicAdd(&s_sh[b], 1u);
    }
    __syncthreads();

    typedef cub::BlockScan<uint32_t, 256> BlockScan;
    __shared__ typename BlockScan::TempStorage scan_tmp;
    uint32_t my_count  = s_sh[tid];
    uint32_t my_offset;
    BlockScan(scan_tmp).ExclusiveSum(my_count, my_offset);
    s_sc[tid] = my_offset;
    __syncthreads();

    if (my_count > 0) {
        int slot = atomicAdd(d_num_segs, 1);
        d_seg_starts[slot] = (int)(sec_off + my_offset);
        d_seg_ends  [slot] = (int)(sec_off + my_offset + my_count);
    }
    __syncthreads();

    s_cursor[tid] = s_sc[tid];
    __syncthreads();

    uint64_t* sec_dst = refine_tmp + sec_off;
    for (uint32_t i = tid; i < sec_n; i += 256) {
        uint64_t x = sec_data[i];
        int b = (int)((x - sec_min) * 256 / sec_range);
        uint32_t dst = atomicAdd(&s_cursor[b], 1u);
        sec_dst[dst] = x;
    }
}

// ===========================================================================
// Multi-block REFINE: four-kernel replacement for sec_minmax + refine_fused
// ===========================================================================

// ---------------------------------------------------------------------------
// Kernel MB-1: refine_minmax_mb_kernel
// Computes per-sector min/max using multiple blocks per sector.
// Each block reduces over its assigned chunk [chunk_off .. chunk_off+chunk_len).
// Atomically updates d_g_sec_minmax_mb[s*2+0] (min) and [s*2+1] (max).
//
// d_blk_sector[b]    : coarse sector index for block b
// d_blk_chunk_off[b] : start element index within the sector's data
// d_blk_chunk_len[b] : number of elements this block processes
// d_g_sec_minmax_mb  : 256*2 uint32 (initialised to 0xFFFFFFFF/0 by host)
// ---------------------------------------------------------------------------
__global__ void refine_minmax_mb_kernel(
    const uint64_t* __restrict__ scatter_out,
    const uint32_t* __restrict__ g_base,
    const int*      __restrict__ d_blk_sector,
    const int*      __restrict__ d_blk_chunk_off,
    const int*      __restrict__ d_blk_chunk_len,
    uint64_t*       __restrict__ d_g_sec_minmax_mb)
{
    const int b   = blockIdx.x;
    const int s   = d_blk_sector[b];
    const int off = d_blk_chunk_off[b];
    const int len = d_blk_chunk_len[b];
    if (len == 0) return;

    const uint64_t* src = scatter_out + g_base[s] + off;

    __shared__ uint64_t s_min, s_max;
    if (threadIdx.x == 0) { s_min = 0xFFFFFFFFFFFFFFFFull; s_max = 0ull; }
    __syncthreads();

    uint64_t lmin = 0xFFFFFFFFFFFFFFFFull, lmax = 0ull;
    for (int i = threadIdx.x; i < len; i += blockDim.x) {
        uint64_t x = src[i];
        if (x < lmin) lmin = x;
        if (x > lmax) lmax = x;
    }
    atomicMin((unsigned long long*)&s_min, (unsigned long long)lmin);
    atomicMax((unsigned long long*)&s_max, (unsigned long long)lmax);
    __syncthreads();

    if (threadIdx.x == 0) {
        atomicMin((unsigned long long*)&d_g_sec_minmax_mb[s * 2 + 0], (unsigned long long)s_min);
        atomicMax((unsigned long long*)&d_g_sec_minmax_mb[s * 2 + 1], (unsigned long long)s_max);
    }
}

// ---------------------------------------------------------------------------
// Kernel MB-2: refine_subhist_mb_kernel
// Computes per-sector sub-histogram using multiple blocks per sector.
// Each block builds a local 256-bin SMEM histogram over its chunk, using
// the per-sector rescale bin:  sub_bin = (x - sec_min) * 256 / sec_range
// then atomicAdd's into g_sub_hist[s*256 + b].
//
// d_g_sec_minmax_mb must be fully populated (kernel MB-1 completed).
// g_sub_hist is zeroed by host before launch.
// ---------------------------------------------------------------------------
__global__ void refine_subhist_mb_kernel(
    const uint64_t* __restrict__ scatter_out,
    const uint32_t* __restrict__ g_base,
    const int*      __restrict__ d_blk_sector,
    const int*      __restrict__ d_blk_chunk_off,
    const int*      __restrict__ d_blk_chunk_len,
    const uint64_t* __restrict__ d_g_sec_minmax_mb,
    uint32_t*       __restrict__ g_sub_hist)          // [256 * 256]
{
    const int b   = blockIdx.x;
    const int s   = d_blk_sector[b];
    const int off = d_blk_chunk_off[b];
    const int len = d_blk_chunk_len[b];
    if (len == 0) return;

    uint64_t sec_min   = d_g_sec_minmax_mb[s * 2 + 0];
    uint64_t sec_max   = d_g_sec_minmax_mb[s * 2 + 1];
    uint64_t sec_range = sec_max - sec_min + 1;

    const uint64_t* src = scatter_out + g_base[s] + off;

    __shared__ uint32_t s_h[256];
    for (int i = threadIdx.x; i < 256; i += blockDim.x) s_h[i] = 0;
    __syncthreads();

    for (int i = threadIdx.x; i < len; i += blockDim.x) {
        uint64_t x = src[i];
        int bin = (int)((x - sec_min) * 256 / sec_range);
        atomicAdd(&s_h[bin], 1u);
    }
    __syncthreads();

    uint32_t* dst = g_sub_hist + (size_t)s * 256;
    for (int i = threadIdx.x; i < 256; i += blockDim.x)
        if (s_h[i]) atomicAdd(&dst[i], s_h[i]);
}

// ---------------------------------------------------------------------------
// Kernel MB-3: refine_scan_mb_kernel
// One block per sector (256 blocks total).
// Reads g_sub_hist[s*256 .. s*256+255], runs BlockScan, writes:
//   g_sub_scan[s*256 + bin]   = exclusive prefix sum (relative offset in sector)
//   g_sub_cursor[s*256 + bin] = same (mutable copy for scatter)
// Emits segment table entries for non-empty sub-bins (same as refine_fused_rescale_kernel).
// ---------------------------------------------------------------------------
__global__ void refine_scan_mb_kernel(
    const uint32_t* __restrict__ g_hist,          // coarse sector counts [256]
    const uint32_t* __restrict__ g_base,          // coarse sector offsets [256]
    const uint32_t* __restrict__ g_sub_hist,      // [256*256]
    uint32_t*       __restrict__ g_sub_scan,      // [256*256] out: exclusive prefix sums
    uint32_t*       __restrict__ g_sub_cursor,    // [256*256] out: mutable scatter cursors
    int*            __restrict__ d_seg_starts,
    int*            __restrict__ d_seg_ends,
    int*            __restrict__ d_num_segs)
{
    const int s   = blockIdx.x;   // sector index 0..255
    const int tid = threadIdx.x;  // 0..255

    if (g_hist[s] == 0) return;

    uint32_t sec_off = g_base[s];

    const uint32_t* sh  = g_sub_hist   + (size_t)s * 256;
    uint32_t*       sc  = g_sub_scan   + (size_t)s * 256;
    uint32_t*       cur = g_sub_cursor + (size_t)s * 256;

    typedef cub::BlockScan<uint32_t, 256> BlockScan;
    __shared__ typename BlockScan::TempStorage scan_tmp;

    uint32_t my_count  = sh[tid];
    uint32_t my_offset;
    BlockScan(scan_tmp).ExclusiveSum(my_count, my_offset);
    __syncthreads();

    sc[tid]  = my_offset;
    cur[tid] = my_offset;
    __syncthreads();

    if (my_count > 0) {
        int slot = atomicAdd(d_num_segs, 1);
        d_seg_starts[slot] = (int)(sec_off + my_offset);
        d_seg_ends  [slot] = (int)(sec_off + my_offset + my_count);
    }
}

// ---------------------------------------------------------------------------
// Kernel MB-4: refine_scatter_mb_kernel
// Multi-block scatter: each block processes its chunk.
// For each element, computes sub_bin via per-sector rescale, then
// atomicAdd on g_sub_cursor[s*256 + sub_bin] to claim slot,
// writes to refine_tmp[sec_off + slot].
//
// g_sub_cursor must be initialised to exclusive prefix sums (kernel MB-3).
// ---------------------------------------------------------------------------
__global__ void refine_scatter_mb_kernel(
    const uint64_t* __restrict__ scatter_out,
    uint64_t*       __restrict__ refine_tmp,
    const uint32_t* __restrict__ g_base,
    const int*      __restrict__ d_blk_sector,
    const int*      __restrict__ d_blk_chunk_off,
    const int*      __restrict__ d_blk_chunk_len,
    const uint64_t* __restrict__ d_g_sec_minmax_mb,
    uint32_t*       __restrict__ g_sub_cursor)      // [256*256]
{
    const int b   = blockIdx.x;
    const int s   = d_blk_sector[b];
    const int off = d_blk_chunk_off[b];
    const int len = d_blk_chunk_len[b];
    if (len == 0) return;

    uint64_t sec_min   = d_g_sec_minmax_mb[s * 2 + 0];
    uint64_t sec_max   = d_g_sec_minmax_mb[s * 2 + 1];
    uint64_t sec_range = sec_max - sec_min + 1;
    uint32_t sec_off   = g_base[s];

    const uint64_t* src = scatter_out + sec_off + off;
    uint32_t*       cur = g_sub_cursor + (size_t)s * 256;

    for (int i = threadIdx.x; i < len; i += blockDim.x) {
        uint64_t x   = src[i];
        int      bin = (int)((x - sec_min) * 256 / sec_range);
        uint32_t dst = atomicAdd(&cur[bin], 1u);
        refine_tmp[sec_off + dst] = x;
    }
}


// ---------------------------------------------------------------------------
// Section 4: NEARLY-SORTED PATH
// ---------------------------------------------------------------------------

// ns_max_disp_kernel: displacement check for fake-sorted routing.
// For each dirty window, checks whether any element's estimated sorted
// position (linear interpolation over [smin,smax]) deviates from its
// current index by more than WINDOW. Uses double (not float) since keys
// span the full int64_t range and float's 24-bit mantissa is insufficient;
// this only feeds a routing heuristic (Regime A vs B), not correctness.
__global__ void ns_max_disp_kernel(const int64_t* __restrict__ arr, int n,
                                    const int* __restrict__ d_dirty_wins, int num_dirty,
                                    int64_t smin, int64_t smax,
                                    int* __restrict__ d_large_disp) {
    int bi = blockIdx.x;
    if (bi >= num_dirty) return;
    int w     = d_dirty_wins[bi];
    int start = w * WINDOW;
    int end   = start + WINDOW;
    if (end > n) end = n;

    int tid = threadIdx.x;
    double range = (double)((uint64_t)smax - (uint64_t)smin);
    if (range < 1.0) range = 1.0;
    double inv_range = (double)(n - 1) / range;

    for (int i = start + tid; i < end; i += 256) {
        int64_t v = arr[i];
        double est = (double)((uint64_t)v - (uint64_t)smin) * inv_range;
        double diff = est - (double)i;
        if (diff < 0.0) diff = -diff;
        if (diff > (double)WINDOW) {
            d_large_disp[0] = 1;
            return;
        }
    }
}

constexpr int BLOCK_NS  = 256;
constexpr int ELEMS_PER = (2 * WINDOW) / BLOCK_NS;  // = 16 elements per thread

// ns_repair_kernel: Regime A local repair. Each block loads arr[w*WINDOW ..
// w*WINDOW+2*WINDOW) into SMEM, sorts with cub::BlockRadixSort<int64_t,256,16>,
// writes back the full sorted 2W region. Two passes (even/odd windows)
// prevent boundary races between adjacent dirty windows.
//
// With 64-bit keys, the data buffer (2*WINDOW*8 = 32768 bytes) plus the
// BlockRadixSort TempStorage exceed the 48KB static-SMEM limit, so this
// kernel uses dynamic shared memory with the large-SMEM opt-in (see
// cudaFuncSetAttribute call in gpu_ns_sort).
struct NsRepairSmem {
    int64_t data[2 * WINDOW];
    typename cub::BlockRadixSort<int64_t, BLOCK_NS, ELEMS_PER>::TempStorage sort_tmp;
};

__global__ void ns_repair_kernel(int64_t* __restrict__ arr, int n,
                                  const int* __restrict__ dirty_wins,
                                  int num_dirty, int pass) {
    extern __shared__ char ns_smem_raw[];
    NsRepairSmem& smem = *reinterpret_cast<NsRepairSmem*>(ns_smem_raw);

    int bi = blockIdx.x;
    if (bi >= num_dirty) return;
    int w = dirty_wins[bi];
    if ((w & 1) != pass) return;

    int start = w * WINDOW;
    int end2  = min(start + 2 * WINDOW, n);
    int len   = end2 - start;
    if (len <= 0) return;

    int tid = threadIdx.x;
    for (int i = tid; i < 2 * WINDOW; i += BLOCK_NS)
        smem.data[i] = (start + i < end2) ? arr[start + i] : INT64_MAX;
    __syncthreads();

    typedef cub::BlockRadixSort<int64_t, BLOCK_NS, ELEMS_PER> BlockSort;

    int64_t keys[ELEMS_PER];
    for (int k = 0; k < ELEMS_PER; ++k)
        keys[k] = smem.data[tid * ELEMS_PER + k];

    BlockSort(smem.sort_tmp).Sort(keys);

    for (int k = 0; k < ELEMS_PER; ++k)
        smem.data[tid * ELEMS_PER + k] = keys[k];
    __syncthreads();

    for (int i = tid; i < 2 * WINDOW && (start + i) < n; i += BLOCK_NS)
        arr[start + i] = smem.data[i];
}

// ns_window_flag_kernel: Regime B extraction: flag every element inside a
// dirty window (one block per window). Extracting whole windows (rather
// than just violation-adjacent elements) guarantees no displaced element
// inside a dirty window is silently left in the clean set.
__global__ void ns_window_flag_kernel(int n,
                                       const int* __restrict__ d_dirty_wins,
                                       int num_dirty,
                                       uint8_t* __restrict__ flags) {
    int bi = blockIdx.x;
    if (bi >= num_dirty) return;
    int start = d_dirty_wins[bi] * WINDOW;
    int end   = start + WINDOW;
    if (end > n) end = n;
    for (int i = start + threadIdx.x; i < end; i += 256)
        flags[i] = 1u;
}

// merge_kernel: parallel merge of two sorted arrays A[0..na) and B[0..nb)
// into out[0..na+nb) via merge-path (binary search per output element).
__global__ void merge_kernel(const int64_t* __restrict__ A, int na,
                              const int64_t* __restrict__ B, int nb,
                              int64_t* __restrict__ out) {
    int total = na + nb;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < total;
         i += gridDim.x * blockDim.x) {
        // Handle edge cases where one array is empty
        if (na == 0) {
            out[i] = B[i];
            continue;
        }
        if (nb == 0) {
            out[i] = A[i];
            continue;
        }
        int lo = (i > nb) ? (i - nb) : 0;
        int hi = (i < na) ? i : na;
        while (lo < hi) {
            int mid = lo + (hi - lo) / 2;
            if (A[mid] <= B[i - mid - 1]) lo = mid + 1;
            else                          hi = mid;
        }
        int ia = lo, ib = i - lo;
        int64_t val;
        if      (ia >= na)       val = B[ib];
        else if (ib >= nb)       val = A[ia];
        else if (A[ia] <= B[ib]) val = A[ia];
        else                     val = B[ib];
        out[i] = val;
    }
}

// gpu_ns_sort: full nearly-sorted dispatch (Section 4).
// Called after verify_kernel has populated d_dirty and h_ver.violations.
// d_input holds SIGNED int64_t data (not yet sign-flipped).
float gpu_ns_sort(int64_t* h_arr, int n, int viol_count, int disp_est,
                  GPUMemory& mem, cudaEvent_t ev_start) {

    if (viol_count > n / NS_BAIL_DIVISOR) {
        printf("[NSSort] path=PATH_NS->PATH_GENERAL (bail viol=%d>n/%d) n=%d\n",
               viol_count, NS_BAIL_DIVISOR, n);
        CUDA_CHECK(cudaMemcpy(h_arr, mem.d_input, n * sizeof(int64_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaEventDestroy(ev_start));
        return gpu_nssort_general(h_arr, n, 0, 0, mem);
    }

    if (disp_est <= WINDOW) {
        // ---- Regime A: local window repair ----------------------------------

        // Build host list of dirty window indices from d_dirty bitmap
        int n_words = ((n + WINDOW - 1) / WINDOW + 31) / 32 + 1;
        if (n_words > DIRTY_WORDS) n_words = DIRTY_WORDS;
        std::vector<uint32_t> h_dirty(n_words);
        CUDA_CHECK(cudaMemcpy(h_dirty.data(), mem.d_dirty,
                              n_words * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        int max_win = (n + WINDOW - 1) / WINDOW;
        std::vector<int> dirty_wins;
        for (int word = 0; word < n_words; ++word) {
            uint32_t bits = h_dirty[word];
            while (bits) {
                int bit = __builtin_ctz(bits);
                int win = word * 32 + bit;
                if (win < max_win) dirty_wins.push_back(win);
                bits &= bits - 1;
            }
        }

        if (!dirty_wins.empty()) {
            // Estimate dirty elements: each dirty window has at most WINDOW elements
            int ne_est = dirty_wins.size() * WINDOW;
            int nc_est = n - ne_est;
            
            // Opt in to >48KB shared memory (sizeof(NsRepairSmem) with 64-bit
            // keys exceeds the static default) for the dynamic SMEM launch.
            static const size_t ns_smem_bytes = sizeof(NsRepairSmem);
            CUDA_CHECK(cudaFuncSetAttribute(
                ns_repair_kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize, (int)ns_smem_bytes));

            int* d_dirty_wins;
            CUDA_CHECK(cudaMalloc(&d_dirty_wins, dirty_wins.size() * sizeof(int)));
            CUDA_CHECK(cudaMemcpy(d_dirty_wins, dirty_wins.data(),
                                  dirty_wins.size() * sizeof(int), cudaMemcpyHostToDevice));
            int nd = (int)dirty_wins.size();
            for (int pass = 0; pass < 2; ++pass) {
                ns_repair_kernel<<<nd, BLOCK_NS, ns_smem_bytes>>>(mem.d_input, n, d_dirty_wins, nd, pass);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaDeviceSynchronize());
            }
            CUDA_CHECK(cudaFree(d_dirty_wins));
        }

        cudaEvent_t stop;
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, stop));
        CUDA_CHECK(cudaEventDestroy(ev_start));
        CUDA_CHECK(cudaEventDestroy(stop));
        CUDA_CHECK(cudaMemcpy(h_arr, mem.d_input, n * sizeof(int64_t), cudaMemcpyDeviceToHost));
        return ms;

    } else {
        // ---- Regime B: dirty-window extract -> sort -> merge back -----------
        // Extraction uses the dirty bitmap (already computed by verify_kernel),
        // NOT ns_flag_kernel. Extracting entire dirty windows closes the
        // stealth-displacement gap: a window is marked dirty when any boundary
        // violation touches it, so no displaced element inside can escape.

        // --- Step 1: decode dirty bitmap -> window list (same as Regime A) ---
        int n_words_b = ((n + WINDOW - 1) / WINDOW + 31) / 32 + 1;
        if (n_words_b > DIRTY_WORDS) n_words_b = DIRTY_WORDS;
        std::vector<uint32_t> h_dirty_b(n_words_b);
        CUDA_CHECK(cudaMemcpy(h_dirty_b.data(), mem.d_dirty,
                              n_words_b * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        int max_win_b = (n + WINDOW - 1) / WINDOW;
        std::vector<int> dirty_wins_b;
        for (int word = 0; word < n_words_b; ++word) {
            uint32_t bits = h_dirty_b[word];
            while (bits) {
                int bit = __builtin_ctz(bits);
                int win = word * 32 + bit;
                if (win < max_win_b) dirty_wins_b.push_back(win);
                bits &= bits - 1;
            }
        }
        int num_dirty_b = (int)dirty_wins_b.size();

        // --- Step 2: build flag buffer from dirty windows --------------------
        // Zero entire flag buffer first (elements outside dirty windows = clean).
        CUDA_CHECK(cudaMemset(mem.d_ns_flags, 0, n * sizeof(uint8_t)));
        if (num_dirty_b > 0) {
            int* d_dirty_wins_b;
            CUDA_CHECK(cudaMalloc(&d_dirty_wins_b, num_dirty_b * sizeof(int)));
            CUDA_CHECK(cudaMemcpy(d_dirty_wins_b, dirty_wins_b.data(),
                                  num_dirty_b * sizeof(int), cudaMemcpyHostToDevice));
            ns_window_flag_kernel<<<num_dirty_b, 256>>>(
                n, d_dirty_wins_b, num_dirty_b, mem.d_ns_flags);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaFree(d_dirty_wins_b));
        }

        // --- Step 3: extract flagged (window) elements -> d_ns_tmp -----------
        {
            int zero = 0;
            CUDA_CHECK(cudaMemcpy(mem.d_ns_num_selected, &zero, sizeof(int),
                                  cudaMemcpyHostToDevice));
            cub::DeviceSelect::Flagged(
                mem.d_ns_select_storage, mem.ns_select_storage_bytes,
                mem.d_input, mem.d_ns_flags, mem.d_ns_tmp,
                mem.d_ns_num_selected, n);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        int ne;
        CUDA_CHECK(cudaMemcpy(&ne, mem.d_ns_num_selected, sizeof(int),
                              cudaMemcpyDeviceToHost));

        // Report ne (window-based) vs ne_old (violation-adjacency) at representative sizes.
        // ne_old = 2 * viol_count (each violation flags 2 elements, roughly).
        // Printed for n <= 1100000 so the sweep output is visible without spamming 100M.

        // Bail to general path if too many elements are dirty (inefficient to sort everything)
        if (ne > n / 2) {
            printf("[NSSort] path=PATH_NEARLY_SORTED regime=B -> PATH_GENERAL (ne=%d > n/2=%d) n=%d\n",
                   ne, n / 2, n);
            CUDA_CHECK(cudaMemcpy(h_arr, mem.d_input, n * sizeof(int64_t), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaEventDestroy(ev_start));
            return gpu_nssort_general(h_arr, n, 0, 0, mem);
        }

        // --- Step 4: extract clean (unflagged) elements -> d_output ----------
        {
            auto inv = cub::TransformInputIterator<uint8_t, InvertFlag, uint8_t*>(
                mem.d_ns_flags, InvertFlag());
            int zero = 0;
            CUDA_CHECK(cudaMemcpy(mem.d_ns_num_selected, &zero, sizeof(int),
                                  cudaMemcpyHostToDevice));
            cub::DeviceSelect::Flagged(
                mem.d_ns_select_storage, mem.ns_select_storage_bytes,
                mem.d_input, inv, mem.d_output,
                mem.d_ns_num_selected, n);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        int nc = n - ne;

        // --- Step 5: sort window-extracted offenders -------------------------
        // Use thrust::sort which handles signed int64_t correctly
        thrust::sort(thrust::device, mem.d_ns_tmp, mem.d_ns_tmp + ne);
        CUDA_CHECK(cudaDeviceSynchronize());

        // --- Step 5.5: sort clean elements as well (they may not be fully sorted) ---
        thrust::sort(thrust::device, mem.d_output, mem.d_output + nc);
        CUDA_CHECK(cudaDeviceSynchronize());

        // --- Step 6: merge clean + sorted offenders -> d_ns_tmp --------------
        // If all elements were dirty (nc == 0), skip merge and use sorted result directly
        if (nc == 0) {
            // d_ns_tmp already contains the sorted result
        } else {
            // Copy sorted result to d_scatter_out for merge
            CUDA_CHECK(cudaMemcpy(mem.d_scatter_out, mem.d_ns_tmp, ne * sizeof(int64_t), cudaMemcpyDeviceToDevice));
            int total = nc + ne;
            int merge_blk = (total + 255) / 256;
            merge_kernel<<<merge_blk, 256>>>(
                mem.d_output, nc,
                (const int64_t*)mem.d_scatter_out, ne,
                mem.d_ns_tmp);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        cudaEvent_t stop;
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, stop));
        CUDA_CHECK(cudaEventDestroy(ev_start));
        CUDA_CHECK(cudaEventDestroy(stop));
        CUDA_CHECK(cudaMemcpy(h_arr, mem.d_ns_tmp, n * sizeof(int64_t), cudaMemcpyDeviceToHost));
        return ms;
    }
}

// ---------------------------------------------------------------------------
// PATH_GENERAL: hist_kernel -> ExclusiveSum into g_base -> copy g_base to
// g_cursor -> chunk_scatter (coarse) -> REFINE per occupied sector ->
// one DeviceSegmentedSort over all leaf segments.
// Input: d_input (signed int64_t, already on device). Output: d_input's
// buffer chain is overwritten with the sorted result; h_arr is written
// back at the end.
// ---------------------------------------------------------------------------
float gpu_nssort_general(int64_t* h_arr, int n, int64_t h_min, int64_t h_max, GPUMemory& mem) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    const int threads = 256;
    const int blocks  = (n + threads - 1) / threads;

    // ---- 3.1 Detect compressed range (rescale variant) -----------------------
    // sign-flip both min and max to get uint64 ordering, then check range.
    uint64_t smin_u = (uint64_t)h_min ^ 0x8000000000000000ull;
    uint64_t smax_u = (uint64_t)h_max ^ 0x8000000000000000ull;
    uint64_t urange = smax_u - smin_u + 1;
    bool do_rescale = (h_min != 0 || h_max != 0) &&
                      (urange < RESCALE_THRESHOLD);
    uint64_t rescale_range = do_rescale ? urange : 1ull;

    // ---- 3.2 Histogram (1 DRAM read) ----------------------------------------
    // Flip sign bit so natural uint64 order matches signed order.
    flip_sign_bit_kernel<<<blocks, threads>>>(mem.d_input, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemset(mem.d_g_hist, 0, COARSE_BINS * sizeof(uint32_t)));
    if (do_rescale) {
        hist_kernel_rescale<<<blocks, threads>>>(
            (const uint64_t*)mem.d_input, (size_t)n, mem.d_g_hist,
            smin_u, rescale_range);
    } else {
        hist_kernel<<<blocks, threads>>>(
            (const uint64_t*)mem.d_input, (size_t)n, mem.d_g_hist);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // ---- 3.3 Prefix sum: g_hist -> g_base ------------------------------------
    cub::DeviceScan::ExclusiveSum(
        mem.d_scan_storage, mem.scan_storage_bytes,
        mem.d_g_hist, mem.d_g_base, COARSE_BINS);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy g_base into g_cursor (cursors advance during scatter)
    CUDA_CHECK(cudaMemcpy(mem.d_g_cursor, mem.d_g_base,
                          COARSE_BINS * sizeof(uint32_t), cudaMemcpyDeviceToDevice));

    // ---- 3.4 CHUNK scatter (256-thread block, epoch-barrier drain) ----------
    {
        const int blocks_256 = (n + 255) / 256;
        int sm_count = 0;
        cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);
        int chunk_blocks = min(blocks_256, sm_count * 16);
        if (do_rescale) {
            chunk_scatter_kernel_rescale<0><<<chunk_blocks, 256>>>(
                (const uint64_t*)mem.d_input, (size_t)n,
                mem.d_scatter_out, mem.d_g_cursor, smin_u, rescale_range);
        } else {
            chunk_scatter_kernel<COARSE_SHIFT><<<chunk_blocks, 256>>>(
                (const uint64_t*)mem.d_input, (size_t)n,
                mem.d_scatter_out, mem.d_g_cursor);
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // ---- 3.5 REFINE: size-based routing ------------------------------------
    //
    // Threshold: SECTOR_MB_THRESHOLD = 2,000,000 elements.
    //   Small sectors (count < threshold): single-block fused path
    //     (sec_minmax_sb_kernel + refine_fused_sb_kernel via sector_list indirection).
    //   Large sectors (count >= threshold): multi-block path
    //     (MB-1 minmax + MB-2 subhist + MB-3 scan + MB-4 scatter).
    //
    // Both paths write segment entries into the same d_fused_seg_starts/ends table
    // and increment the same d_fused_num_segs counter, so the downstream
    // DeviceSegmentedSort consumes both paths' output transparently.
    {
        // --- Host-side: fetch g_hist, partition sectors ---
        uint32_t h_hist_r[COARSE_BINS];
        CUDA_CHECK(cudaMemcpy(h_hist_r, mem.d_g_hist,
                              COARSE_BINS * sizeof(uint32_t), cudaMemcpyDeviceToHost));

        // Partition into small / large lists
        static int   h_small_list  [COARSE_BINS];
        static int   h_blk_sector   [REFINE_MAX_TOTAL_BLOCKS];
        static int   h_blk_chunk_off[REFINE_MAX_TOTAL_BLOCKS];
        static int   h_blk_chunk_len[REFINE_MAX_TOTAL_BLOCKS];

        int num_small  = 0;
        int total_mb_blocks = 0;
        int blocks_per_sector_report[COARSE_BINS] = {};

        for (int s = 0; s < COARSE_BINS; ++s) {
            uint32_t sec_n = h_hist_r[s];
            if (sec_n == 0) continue;
            if (sec_n < SECTOR_MB_THRESHOLD) {
                // Small: single-block path
                h_small_list[num_small++] = s;
            } else {
                // Large: multi-block path
                int bps = (int)(((uint64_t)sec_n + REFINE_CHUNK_SIZE - 1) / REFINE_CHUNK_SIZE);
                if (bps < 1) bps = 1;
                blocks_per_sector_report[s] = bps;
                for (int bi = 0; bi < bps; ++bi) {
                    if (total_mb_blocks >= REFINE_MAX_TOTAL_BLOCKS) {
                        fprintf(stderr, "NSSort_GPU error %s:%d: REFINE_MAX_TOTAL_BLOCKS (%d) exceeded, "
                                        "input too large for multi-block refine path\n",
                                __FILE__, __LINE__, REFINE_MAX_TOTAL_BLOCKS);
                        exit(1);
                    }
                    int chunk_off = bi * REFINE_CHUNK_SIZE;
                    int chunk_len = (int)sec_n - chunk_off;
                    if (chunk_len > REFINE_CHUNK_SIZE) chunk_len = REFINE_CHUNK_SIZE;
                    h_blk_sector   [total_mb_blocks] = s;
                    h_blk_chunk_off[total_mb_blocks] = chunk_off;
                    h_blk_chunk_len[total_mb_blocks] = chunk_len;
                    ++total_mb_blocks;
                }
            }
        }

        // Zero the segment counter once: both paths atomicAdd into it
        int zero = 0;
        CUDA_CHECK(cudaMemcpy(mem.d_fused_num_segs, &zero, sizeof(int), cudaMemcpyHostToDevice));

        // ---- SMALL-SECTOR PATH (single-block fused) ----
        if (num_small > 0) {
            CUDA_CHECK(cudaMemcpy(mem.d_small_sector_list, h_small_list,
                                  num_small * sizeof(int), cudaMemcpyHostToDevice));

            sec_minmax_sb_kernel<<<num_small, 256>>>(
                (const uint64_t*)mem.d_scatter_out,
                (const uint32_t*)mem.d_g_hist,
                (const uint32_t*)mem.d_g_base,
                mem.d_small_sector_list,
                mem.d_sec_minmax);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            refine_fused_sb_kernel<<<num_small, 256>>>(
                (const uint64_t*)mem.d_scatter_out,
                mem.d_refine_tmp,
                (const uint32_t*)mem.d_g_hist,
                (const uint32_t*)mem.d_g_base,
                (const uint64_t*)mem.d_sec_minmax,
                mem.d_small_sector_list,
                mem.d_fused_seg_starts,
                mem.d_fused_seg_ends,
                mem.d_fused_num_segs);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // ---- LARGE-SECTOR PATH (multi-block) ----
        if (total_mb_blocks > 0) {
            CUDA_CHECK(cudaMemcpy(mem.d_blk_sector,    h_blk_sector,
                                  total_mb_blocks * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(mem.d_blk_chunk_off, h_blk_chunk_off,
                                  total_mb_blocks * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(mem.d_blk_chunk_len, h_blk_chunk_len,
                                  total_mb_blocks * sizeof(int), cudaMemcpyHostToDevice));

            // MB-1: per-sector min/max (init only large sectors' slots)
            {
                uint64_t h_init[COARSE_BINS * 2];
                for (int s = 0; s < COARSE_BINS; ++s) {
                    h_init[s * 2 + 0] = 0xFFFFFFFFFFFFFFFFull;
                    h_init[s * 2 + 1] = 0ull;
                }
                CUDA_CHECK(cudaMemcpy(mem.d_g_sec_minmax_mb, h_init,
                                      COARSE_BINS * 2 * sizeof(uint64_t), cudaMemcpyHostToDevice));
            }
            refine_minmax_mb_kernel<<<total_mb_blocks, 256>>>(
                (const uint64_t*)mem.d_scatter_out,
                (const uint32_t*)mem.d_g_base,
                mem.d_blk_sector,
                mem.d_blk_chunk_off,
                mem.d_blk_chunk_len,
                mem.d_g_sec_minmax_mb);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            // MB-2: per-sector sub-histogram
            CUDA_CHECK(cudaMemset(mem.d_g_sub_hist, 0, COARSE_BINS * 256 * sizeof(uint32_t)));
            refine_subhist_mb_kernel<<<total_mb_blocks, 256>>>(
                (const uint64_t*)mem.d_scatter_out,
                (const uint32_t*)mem.d_g_base,
                mem.d_blk_sector,
                mem.d_blk_chunk_off,
                mem.d_blk_chunk_len,
                (const uint64_t*)mem.d_g_sec_minmax_mb,
                mem.d_g_sub_hist);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            // MB-3: scan + segment table (256 blocks; empty large sectors skip via g_hist check).
            // Small sectors have zero in g_sub_hist (memset above), so BlockScan yields
            // my_count==0 for every bin and no segment is emitted for them here.
            refine_scan_mb_kernel<<<COARSE_BINS, 256>>>(
                (const uint32_t*)mem.d_g_hist,
                (const uint32_t*)mem.d_g_base,
                (const uint32_t*)mem.d_g_sub_hist,
                mem.d_g_sub_scan,
                mem.d_g_sub_cursor,
                mem.d_fused_seg_starts,
                mem.d_fused_seg_ends,
                mem.d_fused_num_segs);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            // MB-4: scatter
            refine_scatter_mb_kernel<<<total_mb_blocks, 256>>>(
                (const uint64_t*)mem.d_scatter_out,
                mem.d_refine_tmp,
                (const uint32_t*)mem.d_g_base,
                mem.d_blk_sector,
                mem.d_blk_chunk_off,
                mem.d_blk_chunk_len,
                (const uint64_t*)mem.d_g_sec_minmax_mb,
                mem.d_g_sub_cursor);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }

    // d) ONE DeviceSegmentedSort over all leaf segments
    int num_segs = 0;
    CUDA_CHECK(cudaMemcpy(&num_segs, mem.d_fused_num_segs, sizeof(int), cudaMemcpyDeviceToHost));
    if (num_segs > 0) {
        // Segment offset arrays already live on device: no upload needed.
        cub::DoubleBuffer<uint64_t> d_keys(mem.d_refine_tmp, mem.d_scatter_out);
        cub::DeviceSegmentedSort::SortKeys(
            mem.d_segsort_storage, mem.segsort_storage_bytes,
            d_keys, n, num_segs,
            mem.d_fused_seg_starts, mem.d_fused_seg_ends);
        CUDA_CHECK(cudaDeviceSynchronize());

        // If result landed in the alternate buffer, copy back to d_refine_tmp
        if (d_keys.Current() != mem.d_refine_tmp) {
            CUDA_CHECK(cudaMemcpy(mem.d_refine_tmp, d_keys.Current(),
                                  n * sizeof(uint64_t), cudaMemcpyDeviceToDevice));
        }
    }

    // Flip sign bit back on the final buffer (d_refine_tmp)
    flip_sign_bit_kernel<<<blocks, threads>>>((int64_t*)mem.d_refine_tmp, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    // Copy sorted result back to host
    CUDA_CHECK(cudaMemcpy(h_arr, mem.d_refine_tmp, n * sizeof(int64_t), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms;
}

// ---------------------------------------------------------------------------
// Main NSSort entry point: Section 1.1 routing (exact spec chain)
// ---------------------------------------------------------------------------
float gpu_nssort(int64_t* h_arr, int n, GPUMemory& mem) {
    // Single H2D copy upfront
    CUDA_CHECK(cudaMemcpy(mem.d_input, h_arr, n * sizeof(int64_t), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    // ---- O(1) sample detection (256 threads, 1 block) -----------------------
    ViolationResult h_viol = {0, 0, 0xFFFFFFFFFFFFFFFFull, 0ull, 0};
    CUDA_CHECK(cudaMemcpy(mem.d_viol_result, &h_viol,
                          sizeof(ViolationResult), cudaMemcpyHostToDevice));
    detection_kernel<<<1, SAMPLE_N>>>(mem.d_input, mem.d_viol_result, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&h_viol, mem.d_viol_result,
                          sizeof(ViolationResult), cudaMemcpyDeviceToHost));

    auto finish = [&]() -> float {
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        return ms;
    };

    // ---- Zero-sampling pre-check (32 spread indices) -----------------------
    // Quick check for all-equal arrays to skip expensive verify_kernel
    {
        int h_all_equal = 0;
        CUDA_CHECK(cudaMemcpy(mem.d_all_equal, &h_all_equal, sizeof(int), cudaMemcpyHostToDevice));
        zero_sample_kernel<<<1, 32>>>(mem.d_input, mem.d_all_equal, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(&h_all_equal, mem.d_all_equal, sizeof(int), cudaMemcpyDeviceToHost));
        
        if (h_all_equal) {
            // All sampled values equal - skip verify, just copy
            printf("[NSSort] path=PATH_SORTED (zero-sampled) n=%d\n", n);
            float ms = finish();
            CUDA_CHECK(cudaMemcpy(h_arr, mem.d_input, n * sizeof(int64_t), cudaMemcpyDeviceToHost));
            return ms;
        }
    }

    // ---- Section 1.2 verification read (full O(n) pass) ---------------------
    // Clears and repopulates d_verify_result; sorted/rev flags authoritative.
    {
        VerifyResult zero = {0, 0, 0, 0};
        CUDA_CHECK(cudaMemcpy(mem.d_verify_result, &zero,
                              sizeof(VerifyResult), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(mem.d_dirty, 0, DIRTY_WORDS * sizeof(uint32_t)));
        int vblocks = (n + 255) / 256;
        verify_kernel<<<vblocks, 256>>>(mem.d_input, n, mem.d_verify_result, mem.d_dirty);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        verify_finalize_kernel<<<1, 1>>>(mem.d_verify_result);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    VerifyResult h_ver;
    CUDA_CHECK(cudaMemcpy(&h_ver, mem.d_verify_result,
                          sizeof(VerifyResult), cudaMemcpyDeviceToHost));

    // =========================================================================
    // Section 1.1 ROUTING TABLE: exact spec chain, in this order
    // =========================================================================

    // --- Branch 1: PATH_SORTED (desc_violations == 0 in sample) -------------
    // desc==0: sample saw zero descending pairs -> candidate for sorted.
    // verify confirms with sorted_flag (zero violations over full array).
    // If verify contradicts (fake-sorted), fall to PATH_GENERAL.
    if (h_viol.desc == 0) {
        if (h_ver.sorted_flag) {
            // =====================================================
            // FAST PATH: DO NOT MODIFY
            // =====================================================
            printf("[NSSort] path=PATH_SORTED n=%d\n", n);
            float ms = finish();
            CUDA_CHECK(cudaMemcpy(h_arr, mem.d_input, n * sizeof(int64_t), cudaMemcpyDeviceToHost));
            return ms;
        }
        // Fake-sorted: sample saw no desc violations but array isn't sorted.
        // If violation count is small (nearly sorted below sample resolution),
        // route to gpu_ns_sort. Otherwise fall to PATH_GENERAL.
        if (h_ver.violations <= n / NS_BAIL_DIVISOR) {
            // Distinguish NS-A (local swaps, disp < WINDOW) from NS-B/NS-C
            // (teleports / reversed blocks, disp >> WINDOW) using the dirty
            // bitmap already computed by verify_kernel plus a displacement check.
            //
            // Step 1: decode dirty bitmap into a host-side list of dirty window indices.
            int n_words_fs = ((n + WINDOW - 1) / WINDOW + 31) / 32 + 1;
            if (n_words_fs > DIRTY_WORDS) n_words_fs = DIRTY_WORDS;
            std::vector<uint32_t> h_dirty_fs(n_words_fs);
            CUDA_CHECK(cudaMemcpy(h_dirty_fs.data(), mem.d_dirty,
                                  n_words_fs * sizeof(uint32_t), cudaMemcpyDeviceToHost));
            int max_win_fs = (n + WINDOW - 1) / WINDOW;
            std::vector<int> dirty_wins_fs;
            for (int word = 0; word < n_words_fs; ++word) {
                uint32_t bits = h_dirty_fs[word];
                while (bits) {
                    int bit = __builtin_ctz(bits);
                    int win = word * 32 + bit;
                    if (win < max_win_fs) dirty_wins_fs.push_back(win);
                    bits &= bits - 1;
                }
            }
            int dirty_win_count = (int)dirty_wins_fs.size();

            // Step 2: run ns_max_disp_kernel on dirty windows only.
            // Uses d_inversion_count (1 int) as the large-disp flag.
            int disp_from_bitmap;
            if (dirty_win_count == 0) {
                disp_from_bitmap = 0;  // no dirty windows -> Regime A (trivially)
            } else {
                int* d_dirty_wins_fs;
                CUDA_CHECK(cudaMalloc(&d_dirty_wins_fs,
                                      dirty_win_count * sizeof(int)));
                CUDA_CHECK(cudaMemcpy(d_dirty_wins_fs, dirty_wins_fs.data(),
                                      dirty_win_count * sizeof(int),
                                      cudaMemcpyHostToDevice));
                int zero = 0;
                CUDA_CHECK(cudaMemcpy(mem.d_inversion_count, &zero, sizeof(int),
                                      cudaMemcpyHostToDevice));
                // Use exact global min/max (not the inaccurate 256-sample estimates).
                // Sample smax can be significantly below true max, causing false
                // large-displacement readings in the linear-interpolation check.
                cub::DeviceReduce::Min(mem.d_reduce_storage, mem.reduce_storage_bytes,
                                       mem.d_input, mem.d_min_val, n);
                cub::DeviceReduce::Max(mem.d_reduce_storage, mem.reduce_storage_bytes,
                                       mem.d_input, mem.d_max_val, n);
                CUDA_CHECK(cudaDeviceSynchronize());
                int64_t smin_signed, smax_signed;
                CUDA_CHECK(cudaMemcpy(&smin_signed, mem.d_min_val, sizeof(int64_t),
                                      cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(&smax_signed, mem.d_max_val, sizeof(int64_t),
                                      cudaMemcpyDeviceToHost));
                ns_max_disp_kernel<<<dirty_win_count, 256>>>(
                    mem.d_input, n, d_dirty_wins_fs, dirty_win_count,
                    smin_signed, smax_signed, mem.d_inversion_count);
                CUDA_CHECK(cudaDeviceSynchronize());
                CUDA_CHECK(cudaFree(d_dirty_wins_fs));
                int large_disp = 0;
                CUDA_CHECK(cudaMemcpy(&large_disp, mem.d_inversion_count, sizeof(int),
                                      cudaMemcpyDeviceToHost));
                disp_from_bitmap = large_disp ? (WINDOW + 1) : 0;
            }
            printf("[NSSort] path=PATH_NEARLY_SORTED (fake-sorted->NS) viol=%d dirty_wins=%d disp=%d n=%d\n",
                   h_ver.violations, dirty_win_count, disp_from_bitmap, n);
            CUDA_CHECK(cudaEventDestroy(stop));
            return gpu_ns_sort(h_arr, n, h_ver.violations, disp_from_bitmap, mem, start);
        }
        printf("[NSSort] path=PATH_GENERAL (fake-sorted reroute) n=%d\n", n);
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        return gpu_nssort_general(h_arr, n, 0, 0, mem);
    }

    // --- Branch 2: PATH_REVERSE (asc_violations == 0 in sample) -------------
    // asc==0: sample saw zero ascending pairs -> candidate for reverse-sorted.
    // verify confirms with rev_flag (zero ascending pairs over full array).
    // If verify contradicts (fake-reverse), fall to PATH_GENERAL.
    if (h_viol.asc == 0) {
        if (h_ver.rev_flag) {
            // =====================================================
            // FAST PATH: DO NOT MODIFY
            // =====================================================
            printf("[NSSort] path=PATH_REVERSE n=%d\n", n);
            int blocks = (n + 255) / 256;
            reverse_kernel<<<blocks, 256>>>(mem.d_input, mem.d_output, n);
            CUDA_CHECK(cudaDeviceSynchronize());
            float ms = finish();
            CUDA_CHECK(cudaMemcpy(h_arr, mem.d_output, n * sizeof(int64_t), cudaMemcpyDeviceToHost));
            return ms;
        }
        // Fake-reverse: sample saw no asc violations but array isn't reverse-sorted
        printf("[NSSort] path=PATH_GENERAL (fake-reverse reroute) n=%d\n", n);
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        return gpu_nssort_general(h_arr, n, 0, 0, mem);
    }

    // --- Branch 3: PATH_COUNTING (range < n * COUNTING_RANGE_FRAC) ----------
    // =====================================================
    // FAST PATH: DO NOT MODIFY
    // =====================================================
    cub::DeviceReduce::Min(mem.d_reduce_storage, mem.reduce_storage_bytes,
                           mem.d_input, mem.d_min_val, n);
    cub::DeviceReduce::Max(mem.d_reduce_storage, mem.reduce_storage_bytes,
                           mem.d_input, mem.d_max_val, n);
    int64_t h_min, h_max;
    CUDA_CHECK(cudaMemcpy(&h_min, mem.d_min_val, sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_max, mem.d_max_val, sizeof(int64_t), cudaMemcpyDeviceToHost));
    uint64_t range = (uint64_t)h_max - (uint64_t)h_min + 1ull;

    if ((double)range < (double)n * COUNTING_RANGE_FRAC &&
        range <= (uint64_t)mem.max_range) {
        printf("[NSSort] path=PATH_COUNTING range=%llu n=%d\n", (unsigned long long)range, n);
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        int64_t* h_shifted = new int64_t[n];
        for (int i = 0; i < n; i++) h_shifted[i] = h_arr[i] - h_min;
        float ms = gpu_counting_sort(h_shifted, n, mem, (int)range);
        for (int i = 0; i < n; i++) h_arr[i] = h_shifted[i] + h_min;
        delete[] h_shifted;
        return ms;
    }

    // --- Branch 4: PATH_ADAPTIVE_RADIX (narrowed radix sort for small bit ranges) ---
    uint64_t adaptive_range = (uint64_t)h_max - (uint64_t)h_min;
    int bits_needed = (adaptive_range == 0) ? 1 : (64 - __builtin_clzll(adaptive_range));
    if (bits_needed <= 48) {
        void* d_radix_temp = nullptr;
        size_t radix_temp_bytes = 0;
        cub::DeviceRadixSort::SortKeys(d_radix_temp, radix_temp_bytes,
                                      mem.d_input, mem.d_output, n);
        CUDA_CHECK(cudaMalloc(&d_radix_temp, radix_temp_bytes));
        cub::DeviceRadixSort::SortKeys(d_radix_temp, radix_temp_bytes,
                                      mem.d_input, mem.d_output, n,
                                      0, bits_needed);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaFree(d_radix_temp));
        
        float ms = finish();
        CUDA_CHECK(cudaMemcpy(h_arr, mem.d_output, n * sizeof(int64_t), cudaMemcpyDeviceToHost));
        return ms;
    }

    // --- Branch 5: PATH_NEARLY_SORTED (desc_violations <= NS_THRESH in sample) ---
    // Few descending pairs in sample = nearly sorted (small disorder).
    if (h_viol.desc <= NS_THRESH) {
        CUDA_CHECK(cudaEventDestroy(stop));
        return gpu_ns_sort(h_arr, n, h_ver.violations, h_viol.disp_est, mem, start);
    }

    // --- Branch 5: PATH_GENERAL (with rescale if compressed range) -----------
    printf("[NSSort] path=PATH_GENERAL range=%llu n=%d\n", (unsigned long long)range, n);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return gpu_nssort_general(h_arr, n, h_min, h_max, mem);
}

// ---------------------------------------------------------------------------
// CUB Radix Sort (OneSweep-style single call): the benchmark opponent.
// ---------------------------------------------------------------------------
float gpu_cub_radix_sort_onesweep(int64_t* h_arr, int n, GPUMemory& mem) {
    CUDA_CHECK(cudaMemcpy(mem.d_input, h_arr, n * sizeof(int64_t), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    cub::DeviceRadixSort::SortKeys(
        mem.d_sort_storage, mem.sort_storage_bytes,
        mem.d_input, mem.d_output, n,
        0, 64); // begin_bit, end_bit (full 64-bit range)
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaMemcpy(h_arr, mem.d_output, n * sizeof(int64_t), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return ms;
}


// ---------------------------------------------------------------------------
// Data generators: exact port of NS/NSSort/nssort_bench2.cpp distributions
// (same seeds, same algorithms) so the GPU and CPU benchmarks sort
// bit-identical inputs.
// ---------------------------------------------------------------------------
void generate_zero(int64_t* data, size_t n) {
    for (size_t i = 0; i < n; i++) data[i] = 0;
}

void generate_sorted(int64_t* data, size_t n) {
    for (size_t i = 0; i < n; i++) data[i] = (int64_t)i;
}

void generate_reverse_sorted(int64_t* data, size_t n) {
    for (size_t i = 0; i < n; i++) data[i] = (int64_t)(n - 1 - i);
}

void generate_almost_sorted(int64_t* data, size_t n) {
    for (size_t i = 0; i < n; i++) data[i] = (int64_t)i;
    std::mt19937_64 rng(42);
    std::uniform_int_distribution<size_t> dist(0, n - 1);
    size_t swaps = n / 100;
    for (size_t i = 0; i < swaps; i++) {
        size_t a = dist(rng);
        size_t b = dist(rng);
        std::swap(data[a], data[b]);
    }
}

void generate_root_dup(int64_t* data, size_t n) {
    int64_t root = (int64_t)floor(sqrt((double)n));
    if (root < 1) root = 1;
    for (size_t i = 0; i < n; i++) data[i] = (int64_t)i % root;
}

void generate_two_dup(int64_t* data, size_t n) {
    std::mt19937_64 rng(42);
    std::uniform_int_distribution<int64_t> dist(0, 1);
    for (size_t i = 0; i < n; i++) data[i] = dist(rng);
}

void generate_eight_dup(int64_t* data, size_t n) {
    std::mt19937_64 rng(42);
    std::uniform_int_distribution<int64_t> dist(0, 7);
    for (size_t i = 0; i < n; i++) data[i] = dist(rng);
}

void generate_zipf(int64_t* data, size_t n, double s, int max_unique) {
    std::mt19937_64 rng(42);
    std::vector<double> probs(max_unique);
    double sum = 0.0;
    for (int i = 1; i <= max_unique; i++) {
        probs[i - 1] = 1.0 / pow((double)i, s);
        sum += probs[i - 1];
    }
    std::vector<double> cdf(max_unique);
    cdf[0] = probs[0] / sum;
    for (int i = 1; i < max_unique; i++) cdf[i] = cdf[i - 1] + probs[i] / sum;
    std::uniform_real_distribution<double> udist(0.0, 1.0);
    for (size_t i = 0; i < n; i++) {
        double r = udist(rng);
        int64_t val = max_unique - 1;
        for (int j = 0; j < max_unique; j++) {
            if (r <= cdf[j]) { val = j; break; }
        }
        data[i] = val;
    }
}

static void generate_zipf_default(int64_t* data, size_t n) {
    generate_zipf(data, n, 1.0, 1000);
}

void generate_exponential(int64_t* data, size_t n) {
    std::mt19937_64 rng(42);
    std::exponential_distribution<double> dist(1.0);
    for (size_t i = 0; i < n; i++) data[i] = (int64_t)floor(dist(rng) * (double)n);
}

void generate_uniform(int64_t* data, size_t n) {
    std::mt19937_64 rng(42);
    for (size_t i = 0; i < n; i++) data[i] = (int64_t)rng();
}

// ---------------------------------------------------------------------------
// Correctness: thrust::sort reference + abort-on-mismatch check
// ---------------------------------------------------------------------------
static void thrust_sort(int64_t* h_arr, size_t n) {
    int64_t* d_arr;
    CUDA_CHECK(cudaMalloc(&d_arr, n * sizeof(int64_t)));
    CUDA_CHECK(cudaMemcpy(d_arr, h_arr, n * sizeof(int64_t), cudaMemcpyHostToDevice));
    thrust::sort(thrust::device_ptr<int64_t>(d_arr),
                 thrust::device_ptr<int64_t>(d_arr + n));
    CUDA_CHECK(cudaMemcpy(h_arr, d_arr, n * sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_arr));
}

// Compares `arr` against a thrust::sort reference; aborts the process on
// the first mismatch (used once per distribution, on the first timed run).
static void check_or_abort(const char* algo, const char* dist_name,
                            const int64_t* arr, const int64_t* thrust_ref, size_t n) {
    for (size_t i = 0; i < n; i++) {
        if (arr[i] != thrust_ref[i]) {
            fprintf(stderr,
                "[FATAL] %s output mismatches thrust::sort on distribution '%s' "
                "at index %zu (got %lld, expected %lld)\n",
                algo, dist_name, i, (long long)arr[i], (long long)thrust_ref[i]);
            exit(1);
        }
    }
}

static double median3(float a, float b, float c) {
    float v[3] = {a, b, c};
    std::sort(v, v + 3);
    return v[1];
}

struct Distribution {
    const char* name;
    void (*gen)(int64_t*, size_t);
};

int main() {
    constexpr size_t N = 100'000'000;  // 100M int64_t, matches NSSort CPU benchmark

    printf("NSSort GPU vs cub::DeviceRadixSort — int64_t, N=%zu\n", N);
    printf("=========================================================\n\n");

    const Distribution distributions[] = {
        {"zero",           generate_zero},
        {"sorted",         generate_sorted},
        {"reverse_sorted", generate_reverse_sorted},
        {"almost_sorted",  generate_almost_sorted},
        {"root_dup",       generate_root_dup},
        {"two_dup",        generate_two_dup},
        {"eight_dup",      generate_eight_dup},
        {"zipf",           generate_zipf_default},
        {"exponential",    generate_exponential},
        {"uniform",        generate_uniform},
    };

    GPUMemory mem(N);

    std::vector<int64_t> data(N);
    std::vector<int64_t> buf(N);

    printf("%-16s %12s %12s %8s   %s\n", "distribution", "ns_sort(ms)", "cub(ms)", "ratio", "result");
    printf("-----------------------------------------------------------------\n");

    for (const auto& d : distributions) {
        d.gen(data.data(), N);

        // Warmup: one untimed run per algorithm before the timed passes.
        std::copy(data.begin(), data.end(), buf.begin());
        gpu_nssort(buf.data(), (int)N, mem);
        std::copy(data.begin(), data.end(), buf.begin());
        gpu_cub_radix_sort_onesweep(buf.data(), (int)N, mem);

        // Correctness reference, computed once per distribution.
        std::vector<int64_t> thrust_ref(data);
        thrust_sort(thrust_ref.data(), N);

        float ns_times[3], cub_times[3];
        for (int run = 0; run < 3; run++) {
            std::copy(data.begin(), data.end(), buf.begin());
            ns_times[run] = gpu_nssort(buf.data(), (int)N, mem);
            if (run == 0) check_or_abort("ns_sort", d.name, buf.data(), thrust_ref.data(), N);

            std::copy(data.begin(), data.end(), buf.begin());
            cub_times[run] = gpu_cub_radix_sort_onesweep(buf.data(), (int)N, mem);
            if (run == 0) check_or_abort("cub_radix_sort", d.name, buf.data(), thrust_ref.data(), N);
        }

        double ns_ms  = median3(ns_times[0], ns_times[1], ns_times[2]);
        double cub_ms = median3(cub_times[0], cub_times[1], cub_times[2]);
        double ratio  = (ns_ms < cub_ms) ? (cub_ms / ns_ms) : (ns_ms / cub_ms);
        bool   win    = ns_ms < cub_ms;

        printf("%-16s %12.3f %12.3f %7.2fx   %s\n",
               d.name, ns_ms, cub_ms, ratio, win ? "WIN" : "LOSS");
    }

    printf("\nDone.\n");
    return 0;
}
