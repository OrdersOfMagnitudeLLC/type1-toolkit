// Copyright (c) 2026 Orders of Magnitude LLC
// Licensed under the OOM Commercial License v1.0
// See LICENSE.md in the repository root or ofmagnitude.com

#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <array>
#include <cstring>
#include <immintrin.h>
#include <embree4/rtcore.h>

struct AABB3 {
    float min[3], max[3];
};

struct BVHNode {
    AABB3 bounds;
    int left, right;
    int obj_idx;
    int obj_count;  // For SIMD: number of objects in leaf (1-8)
    int obj_indices[8];  // For SIMD: up to 8 object indices
};

struct alignas(32) BVH8Node {
    // SOA layout for cache efficiency — 8 children bounds
    alignas(32) float min_x[8];
    alignas(32) float min_y[8];
    alignas(32) float min_z[8];
    alignas(32) float max_x[8];
    alignas(32) float max_y[8];
    alignas(32) float max_z[8];
    int children[8];
    int n_children;
    bool is_leaf;
    int obj_indices[8];
    int n_objs;
};

struct Ray {
    float orig[3], dir[3], inv_dir[3];
};

// Grid configuration
const int GX = 25, GY = 20, GZ = 1;
const float CELL_W = 100.0f, CELL_H = 100.0f, CELL_D = 20.0f;

struct GridCell {
    bool occupied;
    std::vector<int> obj_indices;
    std::vector<BVH8Node> local_bvh;
    int local_bvh_root;
};

// AABB-AABB merge
AABB3 merge(const AABB3& a, const AABB3& b) {
    AABB3 result;
    for (int i = 0; i < 3; i++) {
        result.min[i] = std::min(a.min[i], b.min[i]);
        result.max[i] = std::max(a.max[i], b.max[i]);
    }
    return result;
}

// AABB center
void center(const AABB3& box, float c[3]) {
    for (int i = 0; i < 3; i++) {
        c[i] = (box.min[i] + box.max[i]) * 0.5f;
    }
}

// Slab method ray-AABB intersection
bool ray_aabb(const Ray& r, const AABB3& box) {
    float tmin = -INFINITY, tmax = INFINITY;
    for (int i = 0; i < 3; i++) {
        float t1 = (box.min[i] - r.orig[i]) * r.inv_dir[i];
        float t2 = (box.max[i] - r.orig[i]) * r.inv_dir[i];
        tmin = std::max(tmin, std::min(t1, t2));
        tmax = std::min(tmax, std::max(t1, t2));
        if (tmax < tmin) return false;
    }
    return tmax >= 0;
}

// Compute node bounds from object indices
AABB3 compute_bounds(const std::vector<AABB3>& objects, const std::vector<int>& indices) {
    if (indices.empty()) return {{0,0,0},{0,0,0}};
    AABB3 bounds = objects[indices[0]];
    for (size_t i = 1; i < indices.size(); i++) {
        bounds = merge(bounds, objects[indices[i]]);
    }
    return bounds;
}

// Standard SAH BVH2 builder
int build_sah_bvh2(std::vector<BVHNode>& bvh, const std::vector<AABB3>& objects, 
                   std::vector<int> indices) {
    AABB3 bounds = compute_bounds(objects, indices);
    
    // SIMD leaf: stop at 8 objects or fewer
    if (indices.size() <= 8) {
        int node_idx = bvh.size();
        BVHNode node;
        node.bounds = bounds;
        node.left = -1;
        node.right = -1;
        node.obj_idx = indices[0];
        node.obj_count = indices.size();
        for (size_t i = 0; i < indices.size(); i++) {
            node.obj_indices[i] = indices[i];
        }
        for (size_t i = indices.size(); i < 8; i++) {
            node.obj_indices[i] = indices.empty() ? 0 : indices[0];
        }
        bvh.push_back(node);
        return node_idx;
    }
    
    // Find longest axis
    float axis_len[3];
    for (int i = 0; i < 3; i++) {
        axis_len[i] = bounds.max[i] - bounds.min[i];
    }
    int split_axis = 0;
    if (axis_len[1] > axis_len[0]) split_axis = 1;
    if (axis_len[2] > axis_len[split_axis]) split_axis = 2;
    
    // Split at midpoint
    float split_pos = (bounds.min[split_axis] + bounds.max[split_axis]) * 0.5f;
    
    std::vector<int> left, right;
    for (int idx : indices) {
        float c[3];
        center(objects[idx], c);
        if (c[split_axis] < split_pos) {
            left.push_back(idx);
        } else {
            right.push_back(idx);
        }
    }
    
    // Handle degenerate case
    if (left.empty() || right.empty()) {
        int mid = indices.size() / 2;
        left.assign(indices.begin(), indices.begin() + mid);
        right.assign(indices.begin() + mid, indices.end());
    }
    
    int left_child = build_sah_bvh2(bvh, objects, left);
    int right_child = build_sah_bvh2(bvh, objects, right);
    
    int node_idx = bvh.size();
    BVHNode node;
    node.bounds = bounds;
    node.left = left_child;
    node.right = right_child;
    node.obj_idx = -1;
    node.obj_count = 0;
    for (int i = 0; i < 8; i++) node.obj_indices[i] = 0;
    bvh.push_back(node);
    return node_idx;
}

// BVH8 builder - collapses 3 levels of BVH2 into one BVH8 node
int build_bvh8(std::vector<BVH8Node>& bvh, const std::vector<AABB3>& objects, 
               std::vector<int> indices) {
    AABB3 bounds = compute_bounds(objects, indices);
    
    // SIMD leaf: stop at 8 objects or fewer
    if (indices.size() <= 8) {
        int node_idx = bvh.size();
        BVH8Node node;
        // Initialize bounds to leaf bounds
        for (int i = 0; i < 8; i++) {
            node.min_x[i] = bounds.min[0];
            node.min_y[i] = bounds.min[1];
            node.min_z[i] = bounds.min[2];
            node.max_x[i] = bounds.max[0];
            node.max_y[i] = bounds.max[1];
            node.max_z[i] = bounds.max[2];
            node.children[i] = -1;
        }
        node.n_children = 0;
        node.is_leaf = true;
        node.n_objs = indices.size();
        for (size_t i = 0; i < indices.size(); i++) {
            node.obj_indices[i] = indices[i];
        }
        for (size_t i = indices.size(); i < 8; i++) {
            node.obj_indices[i] = indices.empty() ? 0 : indices[0];
        }
        bvh.push_back(node);
        return node_idx;
    }
    
    // Split into 8 partitions using SAH
    // Find longest axis
    float axis_len[3];
    for (int i = 0; i < 3; i++) {
        axis_len[i] = bounds.max[i] - bounds.min[i];
    }
    int split_axis = 0;
    if (axis_len[1] > axis_len[0]) split_axis = 1;
    if (axis_len[2] > axis_len[split_axis]) split_axis = 2;
    
    // Split into 8 equal parts along the longest axis
    std::vector<std::vector<int>> partitions(8);
    float step = axis_len[split_axis] / 8.0f;
    
    for (int idx : indices) {
        float c[3];
        center(objects[idx], c);
        float rel_pos = c[split_axis] - bounds.min[split_axis];
        int part = std::min(7, (int)(rel_pos / step));
        partitions[part].push_back(idx);
    }
    
    // Handle empty partitions - redistribute
    for (int i = 0; i < 8; i++) {
        if (partitions[i].empty()) {
            // Find a non-empty partition and split it
            for (int j = 0; j < 8; j++) {
                if (partitions[j].size() > 1) {
                    int mid = partitions[j].size() / 2;
                    partitions[i].assign(partitions[j].begin(), partitions[j].begin() + mid);
                    partitions[j].erase(partitions[j].begin(), partitions[j].begin() + mid);
                    break;
                }
            }
        }
    }
    
    int children[8];
    for (int i = 0; i < 8; i++) {
        children[i] = build_bvh8(bvh, objects, partitions[i]);
    }
    
    int node_idx = bvh.size();
    BVH8Node node;
    for (int i = 0; i < 8; i++) {
        AABB3 child_bounds = compute_bounds(objects, partitions[i]);
        node.min_x[i] = child_bounds.min[0];
        node.min_y[i] = child_bounds.min[1];
        node.min_z[i] = child_bounds.min[2];
        node.max_x[i] = child_bounds.max[0];
        node.max_y[i] = child_bounds.max[1];
        node.max_z[i] = child_bounds.max[2];
        node.children[i] = children[i];
    }
    node.n_children = 8;
    node.is_leaf = false;
    node.n_objs = 0;
    for (int i = 0; i < 8; i++) node.obj_indices[i] = 0;
    bvh.push_back(node);
    return node_idx;
}

// Ray traversal for BVH2 (iterative to avoid stack overflow)
int traverse_bvh2(const std::vector<BVHNode>& bvh, const Ray& r, int node_idx, int& visit_count, 
                  const std::vector<AABB3>& objects) {
    std::vector<int> stack;
    stack.push_back(node_idx);
    
    while (!stack.empty()) {
        int idx = stack.back();
        stack.pop_back();
        
        if (idx < 0 || idx >= (int)bvh.size()) continue;
        
        const BVHNode& node = bvh[idx];
        visit_count++;
        
        if (!ray_aabb(r, node.bounds)) {
            continue;
        }
        
        // Leaf node
        if (node.left == -1 && node.right == -1) {
            // SIMD leaf testing with AVX2
            if (node.obj_count > 1) {
#ifdef __AVX2__
                // Pack 8 AABB min/max into SOA layout
                __m256 min_x = _mm256_setr_ps(
                    objects[node.obj_indices[0]].min[0], objects[node.obj_indices[1]].min[0],
                    objects[node.obj_indices[2]].min[0], objects[node.obj_indices[3]].min[0],
                    objects[node.obj_indices[4]].min[0], objects[node.obj_indices[5]].min[0],
                    objects[node.obj_indices[6]].min[0], objects[node.obj_indices[7]].min[0]);
                __m256 max_x = _mm256_setr_ps(
                    objects[node.obj_indices[0]].max[0], objects[node.obj_indices[1]].max[0],
                    objects[node.obj_indices[2]].max[0], objects[node.obj_indices[3]].max[0],
                    objects[node.obj_indices[4]].max[0], objects[node.obj_indices[5]].max[0],
                    objects[node.obj_indices[6]].max[0], objects[node.obj_indices[7]].max[0]);
                __m256 min_y = _mm256_setr_ps(
                    objects[node.obj_indices[0]].min[1], objects[node.obj_indices[1]].min[1],
                    objects[node.obj_indices[2]].min[1], objects[node.obj_indices[3]].min[1],
                    objects[node.obj_indices[4]].min[1], objects[node.obj_indices[5]].min[1],
                    objects[node.obj_indices[6]].min[1], objects[node.obj_indices[7]].min[1]);
                __m256 max_y = _mm256_setr_ps(
                    objects[node.obj_indices[0]].max[1], objects[node.obj_indices[1]].max[1],
                    objects[node.obj_indices[2]].max[1], objects[node.obj_indices[3]].max[1],
                    objects[node.obj_indices[4]].max[1], objects[node.obj_indices[5]].max[1],
                    objects[node.obj_indices[6]].max[1], objects[node.obj_indices[7]].max[1]);
                __m256 min_z = _mm256_setr_ps(
                    objects[node.obj_indices[0]].min[2], objects[node.obj_indices[1]].min[2],
                    objects[node.obj_indices[2]].min[2], objects[node.obj_indices[3]].min[2],
                    objects[node.obj_indices[4]].min[2], objects[node.obj_indices[5]].min[2],
                    objects[node.obj_indices[6]].min[2], objects[node.obj_indices[7]].min[2]);
                __m256 max_z = _mm256_setr_ps(
                    objects[node.obj_indices[0]].max[2], objects[node.obj_indices[1]].max[2],
                    objects[node.obj_indices[2]].max[2], objects[node.obj_indices[3]].max[2],
                    objects[node.obj_indices[4]].max[2], objects[node.obj_indices[5]].max[2],
                    objects[node.obj_indices[6]].max[2], objects[node.obj_indices[7]].max[2]);
                
                __m256 r_orig_x = _mm256_set1_ps(r.orig[0]);
                __m256 r_orig_y = _mm256_set1_ps(r.orig[1]);
                __m256 r_orig_z = _mm256_set1_ps(r.orig[2]);
                __m256 r_inv_x = _mm256_set1_ps(r.inv_dir[0]);
                __m256 r_inv_y = _mm256_set1_ps(r.inv_dir[1]);
                __m256 r_inv_z = _mm256_set1_ps(r.inv_dir[2]);
                
                // Slab test all 8 at once
                __m256 t1_x = _mm256_mul_ps(_mm256_sub_ps(min_x, r_orig_x), r_inv_x);
                __m256 t2_x = _mm256_mul_ps(_mm256_sub_ps(max_x, r_orig_x), r_inv_x);
                __m256 tmin_x = _mm256_min_ps(t1_x, t2_x);
                __m256 tmax_x = _mm256_max_ps(t1_x, t2_x);
                
                __m256 t1_y = _mm256_mul_ps(_mm256_sub_ps(min_y, r_orig_y), r_inv_y);
                __m256 t2_y = _mm256_mul_ps(_mm256_sub_ps(max_y, r_orig_y), r_inv_y);
                __m256 tmin_y = _mm256_min_ps(t1_y, t2_y);
                __m256 tmax_y = _mm256_max_ps(t1_y, t2_y);
                
                __m256 t1_z = _mm256_mul_ps(_mm256_sub_ps(min_z, r_orig_z), r_inv_z);
                __m256 t2_z = _mm256_mul_ps(_mm256_sub_ps(max_z, r_orig_z), r_inv_z);
                __m256 tmin_z = _mm256_min_ps(t1_z, t2_z);
                __m256 tmax_z = _mm256_max_ps(t1_z, t2_z);
                
                __m256 tmin = _mm256_max_ps(_mm256_max_ps(tmin_x, tmin_y), tmin_z);
                __m256 tmax = _mm256_min_ps(_mm256_min_ps(tmax_x, tmax_y), tmax_z);
                
                __m256 tmax_ge_tmin = _mm256_cmp_ps(tmax, tmin, _CMP_GE_OQ);
                __m256 tmax_ge_zero = _mm256_cmp_ps(tmax, _mm256_setzero_ps(), _CMP_GE_OQ);
                __m256 hit_mask = _mm256_and_ps(tmax_ge_tmin, tmax_ge_zero);
                
                int mask = _mm256_movemask_ps(hit_mask);
                
                // Return first hit object
                for (int i = 0; i < node.obj_count; i++) {
                    if (mask & (1 << i)) {
                        return node.obj_indices[i];
                    }
                }
                continue;
#else
                // Fallback: scalar test
                for (int i = 0; i < node.obj_count; i++) {
                    if (ray_aabb(r, objects[node.obj_indices[i]])) {
                        return node.obj_indices[i];
                    }
                }
                continue;
#endif
            }
            return node.obj_idx;
        }
        
        // Internal node: test both children
        if (node.right != -1 && ray_aabb(r, bvh[node.right].bounds)) {
            stack.push_back(node.right);
        }
        if (node.left != -1 && ray_aabb(r, bvh[node.left].bounds)) {
            stack.push_back(node.left);
        }
    }
    return -1;
}

static int avx2_leaf_count = 0;
static int scalar_leaf_count = 0;
static int avx2_internal_count = 0;
static int scalar_internal_count = 0;

// Ray traversal for BVH8 with AVX2 (iterative to avoid stack overflow)
int traverse_bvh8(const std::vector<BVH8Node>& bvh, const Ray& r, int node_idx, int& visit_count, 
                  const std::vector<AABB3>& objects) {
    int fixed_stack[64];
    int stack_top = 0;
    fixed_stack[stack_top++] = node_idx;
    
    while (stack_top > 0) {
        int idx = fixed_stack[--stack_top];
        
        if (idx < 0 || idx >= (int)bvh.size()) continue;
        
        const BVH8Node& node = bvh[idx];
        visit_count++;
        
        // Leaf node
        if (node.is_leaf) {
            if (node.n_objs > 1) {
#ifdef __AVX2__
                avx2_leaf_count++;
                // Pack 8 AABB min/max into SOA layout
                __m256 min_x = _mm256_setr_ps(
                    objects[node.obj_indices[0]].min[0], objects[node.obj_indices[1]].min[0],
                    objects[node.obj_indices[2]].min[0], objects[node.obj_indices[3]].min[0],
                    objects[node.obj_indices[4]].min[0], objects[node.obj_indices[5]].min[0],
                    objects[node.obj_indices[6]].min[0], objects[node.obj_indices[7]].min[0]);
                __m256 max_x = _mm256_setr_ps(
                    objects[node.obj_indices[0]].max[0], objects[node.obj_indices[1]].max[0],
                    objects[node.obj_indices[2]].max[0], objects[node.obj_indices[3]].max[0],
                    objects[node.obj_indices[4]].max[0], objects[node.obj_indices[5]].max[0],
                    objects[node.obj_indices[6]].max[0], objects[node.obj_indices[7]].max[0]);
                __m256 min_y = _mm256_setr_ps(
                    objects[node.obj_indices[0]].min[1], objects[node.obj_indices[1]].min[1],
                    objects[node.obj_indices[2]].min[1], objects[node.obj_indices[3]].min[1],
                    objects[node.obj_indices[4]].min[1], objects[node.obj_indices[5]].min[1],
                    objects[node.obj_indices[6]].min[1], objects[node.obj_indices[7]].min[1]);
                __m256 max_y = _mm256_setr_ps(
                    objects[node.obj_indices[0]].max[1], objects[node.obj_indices[1]].max[1],
                    objects[node.obj_indices[2]].max[1], objects[node.obj_indices[3]].max[1],
                    objects[node.obj_indices[4]].max[1], objects[node.obj_indices[5]].max[1],
                    objects[node.obj_indices[6]].max[1], objects[node.obj_indices[7]].max[1]);
                __m256 min_z = _mm256_setr_ps(
                    objects[node.obj_indices[0]].min[2], objects[node.obj_indices[1]].min[2],
                    objects[node.obj_indices[2]].min[2], objects[node.obj_indices[3]].min[2],
                    objects[node.obj_indices[4]].min[2], objects[node.obj_indices[5]].min[2],
                    objects[node.obj_indices[6]].min[2], objects[node.obj_indices[7]].min[2]);
                __m256 max_z = _mm256_setr_ps(
                    objects[node.obj_indices[0]].max[2], objects[node.obj_indices[1]].max[2],
                    objects[node.obj_indices[2]].max[2], objects[node.obj_indices[3]].max[2],
                    objects[node.obj_indices[4]].max[2], objects[node.obj_indices[5]].max[2],
                    objects[node.obj_indices[6]].max[2], objects[node.obj_indices[7]].max[2]);
                
                __m256 r_orig_x = _mm256_set1_ps(r.orig[0]);
                __m256 r_orig_y = _mm256_set1_ps(r.orig[1]);
                __m256 r_orig_z = _mm256_set1_ps(r.orig[2]);
                __m256 r_inv_x = _mm256_set1_ps(r.inv_dir[0]);
                __m256 r_inv_y = _mm256_set1_ps(r.inv_dir[1]);
                __m256 r_inv_z = _mm256_set1_ps(r.inv_dir[2]);
                
                // Slab test all 8 at once
                __m256 t1_x = _mm256_mul_ps(_mm256_sub_ps(min_x, r_orig_x), r_inv_x);
                __m256 t2_x = _mm256_mul_ps(_mm256_sub_ps(max_x, r_orig_x), r_inv_x);
                __m256 tmin_x = _mm256_min_ps(t1_x, t2_x);
                __m256 tmax_x = _mm256_max_ps(t1_x, t2_x);
                
                __m256 t1_y = _mm256_mul_ps(_mm256_sub_ps(min_y, r_orig_y), r_inv_y);
                __m256 t2_y = _mm256_mul_ps(_mm256_sub_ps(max_y, r_orig_y), r_inv_y);
                __m256 tmin_y = _mm256_min_ps(t1_y, t2_y);
                __m256 tmax_y = _mm256_max_ps(t1_y, t2_y);
                
                __m256 t1_z = _mm256_mul_ps(_mm256_sub_ps(min_z, r_orig_z), r_inv_z);
                __m256 t2_z = _mm256_mul_ps(_mm256_sub_ps(max_z, r_orig_z), r_inv_z);
                __m256 tmin_z = _mm256_min_ps(t1_z, t2_z);
                __m256 tmax_z = _mm256_max_ps(t1_z, t2_z);
                
                __m256 tmin = _mm256_max_ps(_mm256_max_ps(tmin_x, tmin_y), tmin_z);
                __m256 tmax = _mm256_min_ps(_mm256_min_ps(tmax_x, tmax_y), tmax_z);
                
                __m256 tmax_ge_tmin = _mm256_cmp_ps(tmax, tmin, _CMP_GE_OQ);
                __m256 tmax_ge_zero = _mm256_cmp_ps(tmax, _mm256_setzero_ps(), _CMP_GE_OQ);
                __m256 hit_mask = _mm256_and_ps(tmax_ge_tmin, tmax_ge_zero);
                
                int mask = _mm256_movemask_ps(hit_mask);
                
                // Return first hit object (only check valid slots)
                for (int i = 0; i < node.n_objs; i++) {
                    if (mask & (1 << i)) {
                        return node.obj_indices[i];
                    }
                }
                continue;
#else
                scalar_leaf_count++;
                for (int i = 0; i < node.n_objs; i++) {
                    if (ray_aabb(r, objects[node.obj_indices[i]])) {
                        return node.obj_indices[i];
                    }
                }
                continue;
#endif
            }
            return node.obj_indices[0];
        }
        
        // Internal node: test all 8 children with AVX2
#ifdef __AVX2__
        avx2_internal_count++;
        __m256 orig_x = _mm256_set1_ps(r.orig[0]);
        __m256 orig_y = _mm256_set1_ps(r.orig[1]);
        __m256 orig_z = _mm256_set1_ps(r.orig[2]);
        __m256 inv_dx = _mm256_set1_ps(r.inv_dir[0]);
        __m256 inv_dy = _mm256_set1_ps(r.inv_dir[1]);
        __m256 inv_dz = _mm256_set1_ps(r.inv_dir[2]);

        __m256 nmin_x = _mm256_loadu_ps(node.min_x);
        __m256 nmax_x = _mm256_loadu_ps(node.max_x);
        __m256 nmin_y = _mm256_loadu_ps(node.min_y);
        __m256 nmax_y = _mm256_loadu_ps(node.max_y);
        __m256 nmin_z = _mm256_loadu_ps(node.min_z);
        __m256 nmax_z = _mm256_loadu_ps(node.max_z);

        // Slab test all 8:
        __m256 tx0 = _mm256_mul_ps(_mm256_sub_ps(nmin_x, orig_x), inv_dx);
        __m256 tx1 = _mm256_mul_ps(_mm256_sub_ps(nmax_x, orig_x), inv_dx);
        __m256 ty0 = _mm256_mul_ps(_mm256_sub_ps(nmin_y, orig_y), inv_dy);
        __m256 ty1 = _mm256_mul_ps(_mm256_sub_ps(nmax_y, orig_y), inv_dy);
        __m256 tz0 = _mm256_mul_ps(_mm256_sub_ps(nmin_z, orig_z), inv_dz);
        __m256 tz1 = _mm256_mul_ps(_mm256_sub_ps(nmax_z, orig_z), inv_dz);

        __m256 tmin_x = _mm256_min_ps(tx0, tx1);
        __m256 tmax_x = _mm256_max_ps(tx0, tx1);
        __m256 tmin_y = _mm256_min_ps(ty0, ty1);
        __m256 tmax_y = _mm256_max_ps(ty0, ty1);
        __m256 tmin_z = _mm256_min_ps(tz0, tz1);
        __m256 tmax_z = _mm256_max_ps(tz0, tz1);

        __m256 tmin = _mm256_max_ps(_mm256_max_ps(tmin_x, tmin_y), tmin_z);
        __m256 tmax = _mm256_min_ps(_mm256_min_ps(tmax_x, tmax_y), tmax_z);

        __m256 zero = _mm256_setzero_ps();
        __m256 tmax_ge_tmin = _mm256_cmp_ps(tmax, tmin, _CMP_GE_OQ);
        __m256 tmax_ge_zero = _mm256_cmp_ps(tmax, zero, _CMP_GE_OQ);
        int hit_mask = _mm256_movemask_ps(_mm256_and_ps(tmax_ge_tmin, tmax_ge_zero));

        // Push hit children in reverse order (so first is processed first)
        for (int i = 7; i >= 0; i--) {
            if (hit_mask & (1 << i) && node.children[i] != -1) {
                fixed_stack[stack_top++] = node.children[i];
            }
        }
#else
        scalar_internal_count++;
        // Fallback: scalar test for each child
        for (int i = 7; i >= 0; i--) {
            if (node.children[i] != -1) {
                AABB3 child_bounds = {{node.min_x[i], node.min_y[i], node.min_z[i]},
                                      {node.max_x[i], node.max_y[i], node.max_z[i]}};
                if (ray_aabb(r, child_bounds)) {
                    fixed_stack[stack_top++] = node.children[i];
                }
            }
        }
#endif
    }
    return -1;
}

// Compute BVH8 tree depth
int compute_depth_bvh8(const std::vector<BVH8Node>& bvh, int idx = 0) {
    if (idx < 0 || idx >= (int)bvh.size()) return 0;
    if (bvh[idx].is_leaf) return 1;
    int max_child_depth = 0;
    for (int i = 0; i < 8; i++) {
        if (bvh[idx].children[i] != -1) {
            max_child_depth = std::max(max_child_depth, compute_depth_bvh8(bvh, bvh[idx].children[i]));
        }
    }
    return 1 + max_child_depth;
}

// DDA grid traversal with per-cell BVH
int traverse_ns_grid(const GridCell grid[GX][GY][GZ], const std::vector<AABB3>& objects,
                      const Ray& r, int& grid_cells_visited, int& bvh_nodes_visited) {
    // Check if ray enters grid bounds
    AABB3 grid_bounds = {{0, 0, -10}, {2500, 2000, 10}};
    
    // Compute entry point into grid
    float t_entry = 0.0f;
    for (int i = 0; i < 3; i++) {
        float t1 = (grid_bounds.min[i] - r.orig[i]) * r.inv_dir[i];
        float t2 = (grid_bounds.max[i] - r.orig[i]) * r.inv_dir[i];
        float t_near = std::min(t1, t2);
        float t_far = std::max(t1, t2);
        if (t_near > t_entry) t_entry = t_near;
    }
    if (t_entry < 0) t_entry = 0.0f; // Ray starts inside
    
    // Entry point
    float entry_x = r.orig[0] + r.dir[0] * t_entry;
    float entry_y = r.orig[1] + r.dir[1] * t_entry;
    float entry_z = r.orig[2] + r.dir[2] * t_entry;
    
    // Initialize DDA from entry point
    int gx = std::clamp((int)(entry_x / CELL_W), 0, GX - 1);
    int gy = std::clamp((int)(entry_y / CELL_H), 0, GY - 1);
    int gz = 0;
    
    float t_delta_x = (r.dir[0] != 0) ? CELL_W / std::abs(r.dir[0]) : INFINITY;
    float t_delta_y = (r.dir[1] != 0) ? CELL_H / std::abs(r.dir[1]) : INFINITY;
    float t_delta_z = (r.dir[2] != 0) ? CELL_D / std::abs(r.dir[2]) : INFINITY;
    
    int step_x = (r.dir[0] >= 0) ? 1 : -1;
    int step_y = (r.dir[1] >= 0) ? 1 : -1;
    int step_z = (r.dir[2] >= 0) ? 1 : -1;
    
    float t_max_x = (r.dir[0] != 0) ? 
        ((step_x == 1) ? ((gx + 1) * CELL_W - entry_x) : (gx * CELL_W - entry_x)) / r.dir[0] : INFINITY;
    float t_max_y = (r.dir[1] != 0) ? 
        ((step_y == 1) ? ((gy + 1) * CELL_H - entry_y) : (gy * CELL_H - entry_y)) / r.dir[1] : INFINITY;
    float t_max_z = (r.dir[2] != 0) ? 
        ((step_z == 1) ? ((gz + 1) * CELL_D - entry_z) : (gz * CELL_D - entry_z)) / r.dir[2] : INFINITY;
    
    // March through grid
    while (gx >= 0 && gx < GX && gy >= 0 && gy < GY && gz >= 0 && gz < GZ) {
        grid_cells_visited++;
        
        if (grid[gx][gy][gz].occupied) {
            int visits = 0;
            int result = traverse_bvh8(grid[gx][gy][gz].local_bvh, r, 
                                      grid[gx][gy][gz].local_bvh_root, visits, objects);
            bvh_nodes_visited += visits;
            if (result != -1) return result;
        }
        
        // Step to next cell
        if (t_max_x < t_max_y && t_max_x < t_max_z) {
            gx += step_x;
            t_max_x += t_delta_x;
        } else if (t_max_y < t_max_z) {
            gy += step_y;
            t_max_y += t_delta_y;
        } else {
            gz += step_z;
            t_max_z += t_delta_z;
        }
    }
    
    return -1;
}

// ---------------------------------------------------------------------------
// Scene benchmark helper
// ---------------------------------------------------------------------------

struct SceneResult {
    const char* name;
    double ns_time_ms;
    double embree_time_ms;
    int ns_hits;
    int embree_hits;
    int total_objects;
    int occupied_cells;
};

struct AABBUserData {
    AABB3* objects;
    size_t n;
};

static SceneResult run_scene(const char* name, std::vector<AABB3>& objects,
                              const std::vector<Ray>& rays) {
    SceneResult r;
    r.name = name;
    r.total_objects = (int)objects.size();

    // --- Build NS grid + per-cell BVH8 ---
    GridCell grid[GX][GY][GZ];
    for (int x = 0; x < GX; x++)
        for (int y = 0; y < GY; y++)
            for (int z = 0; z < GZ; z++) {
                grid[x][y][z].occupied = false;
                grid[x][y][z].local_bvh_root = -1;
            }

    for (size_t i = 0; i < objects.size(); i++) {
        float cx = (objects[i].min[0] + objects[i].max[0]) * 0.5f;
        float cy = (objects[i].min[1] + objects[i].max[1]) * 0.5f;
        int gx = std::clamp((int)(cx / CELL_W), 0, GX - 1);
        int gy = std::clamp((int)(cy / CELL_H), 0, GY - 1);
        grid[gx][gy][0].obj_indices.push_back((int)i);
        grid[gx][gy][0].occupied = true;
    }

    for (int x = 0; x < GX; x++)
        for (int y = 0; y < GY; y++)
            for (int z = 0; z < GZ; z++)
                if (grid[x][y][z].occupied && !grid[x][y][z].obj_indices.empty())
                    grid[x][y][z].local_bvh_root = build_bvh8(
                        grid[x][y][z].local_bvh, objects,
                        grid[x][y][z].obj_indices);

    r.occupied_cells = 0;
    for (int x = 0; x < GX; x++)
        for (int y = 0; y < GY; y++)
            if (grid[x][y][0].occupied) r.occupied_cells++;

    // --- Run NS traversal ---
    r.ns_hits = 0;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (const auto& ray : rays) {
        int gv = 0, bv = 0;
        if (traverse_ns_grid(grid, objects, ray, gv, bv) != -1)
            r.ns_hits++;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    r.ns_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // --- Build Embree scene ---
    AABBUserData ud{objects.data(), objects.size()};
    RTCDevice device = rtcNewDevice(nullptr);
    RTCScene scene = rtcNewScene(device);
    RTCGeometry geom = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_USER);
    rtcSetGeometryUserPrimitiveCount(geom, objects.size());
    rtcSetGeometryUserData(geom, &ud);

    rtcSetGeometryBoundsFunction(geom,
        [](const RTCBoundsFunctionArguments* args) {
            auto* ud = (AABBUserData*)args->geometryUserPtr;
            const AABB3& obj = ud->objects[args->primID];
            args->bounds_o->lower_x = obj.min[0];
            args->bounds_o->lower_y = obj.min[1];
            args->bounds_o->lower_z = obj.min[2];
            args->bounds_o->upper_x = obj.max[0];
            args->bounds_o->upper_y = obj.max[1];
            args->bounds_o->upper_z = obj.max[2];
        }, nullptr);

    rtcSetGeometryIntersectFunction(geom,
        [](const RTCIntersectFunctionNArguments* args) {
            if (args->N != 1) return;
            auto* ud = (AABBUserData*)args->geometryUserPtr;
            const AABB3& obj = ud->objects[args->primID];
            RTCRayHit* rh = (RTCRayHit*)args->rayhit;
            float tmin = -INFINITY, tmax = INFINITY;
            float d[3] = {rh->ray.dir_x, rh->ray.dir_y, rh->ray.dir_z};
            float o[3] = {rh->ray.org_x, rh->ray.org_y, rh->ray.org_z};
            float mn[3] = {obj.min[0], obj.min[1], obj.min[2]};
            float mx[3] = {obj.max[0], obj.max[1], obj.max[2]};
            for (int i = 0; i < 3; i++) {
                if (std::abs(d[i]) > 1e-6f) {
                    float t1 = (mn[i] - o[i]) / d[i];
                    float t2 = (mx[i] - o[i]) / d[i];
                    tmin = std::max(tmin, std::min(t1, t2));
                    tmax = std::min(tmax, std::max(t1, t2));
                } else if (o[i] < mn[i] || o[i] > mx[i]) return;
            }
            if (tmax >= tmin && tmax >= 0 && tmin < rh->ray.tfar) {
                rh->ray.tfar = tmin;
                rh->hit.primID = args->primID;
                rh->hit.geomID = 0;
                rh->hit.u = 0.0f;
                rh->hit.v = 0.0f;
            }
        });

    rtcCommitGeometry(geom);
    rtcAttachGeometry(scene, geom);
    rtcReleaseGeometry(geom);
    rtcCommitScene(scene);

    // --- Run Embree ---
    r.embree_hits = 0;
    auto t2 = std::chrono::high_resolution_clock::now();
    for (const auto& ray : rays) {
        RTCRayHit rh;
        rh.ray.org_x = ray.orig[0]; rh.ray.org_y = ray.orig[1]; rh.ray.org_z = ray.orig[2];
        rh.ray.dir_x = ray.dir[0];  rh.ray.dir_y = ray.dir[1];  rh.ray.dir_z = ray.dir[2];
        rh.ray.tnear = 0.0f; rh.ray.tfar = INFINITY;
        rh.ray.mask = -1; rh.ray.flags = 0;
        rh.hit.geomID = RTC_INVALID_GEOMETRY_ID;
        rh.hit.primID = RTC_INVALID_GEOMETRY_ID;
        rtcIntersect1(scene, &rh);
        if (rh.hit.geomID != RTC_INVALID_GEOMETRY_ID) r.embree_hits++;
    }
    auto t3 = std::chrono::high_resolution_clock::now();
    r.embree_time_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

    rtcReleaseScene(scene);
    rtcReleaseDevice(device);

    return r;
}

static std::vector<Ray> generate_random_rays(std::mt19937& rng, int count) {
    std::vector<Ray> rays;
    std::uniform_real_distribution<float> rand_dir(-1.0f, 1.0f);
    std::uniform_real_distribution<float> rand_orig(0.0f, 2500.0f);
    for (int i = 0; i < count; i++) {
        Ray r;
        r.orig[0] = rand_orig(rng);
        r.orig[1] = rand_orig(rng) * 0.8f;
        r.orig[2] = -1000.0f;
        r.dir[0] = rand_dir(rng);
        r.dir[1] = rand_dir(rng);
        r.dir[2] = std::abs(rand_dir(rng)) + 0.1f;
        float len = std::sqrt(r.dir[0]*r.dir[0] + r.dir[1]*r.dir[1] + r.dir[2]*r.dir[2]);
        r.dir[0] /= len; r.dir[1] /= len; r.dir[2] /= len;
        r.inv_dir[0] = 1.0f / r.dir[0];
        r.inv_dir[1] = 1.0f / r.dir[1];
        r.inv_dir[2] = 1.0f / r.dir[2];
        rays.push_back(r);
    }
    return rays;
}

static std::vector<Ray> generate_targeted_rays(std::mt19937& rng, int count,
        const std::vector<std::array<float,3>>& cluster_centers) {
    std::vector<Ray> rays;
    std::uniform_int_distribution<int> rand_cluster(0, (int)cluster_centers.size() - 1);
    std::uniform_real_distribution<float> rand_dir(-1.0f, 1.0f);
    std::uniform_real_distribution<float> rand_orig(0.0f, 2500.0f);
    for (int i = 0; i < count; i++) {
        Ray r;
        if (i < count * 0.7) {
            int target = rand_cluster(rng);
            r.orig[0] = 5000.0f;
            r.orig[1] = 3750.0f;
            r.orig[2] = -1000.0f;
            float dx = cluster_centers[target][0] - r.orig[0];
            float dy = cluster_centers[target][1] - r.orig[1];
            float dz = cluster_centers[target][2] - r.orig[2];
            float len = std::sqrt(dx*dx + dy*dy + dz*dz);
            r.dir[0] = dx / len; r.dir[1] = dy / len; r.dir[2] = dz / len;
        } else {
            r.orig[0] = rand_orig(rng);
            r.orig[1] = rand_orig(rng) * 0.8f;
            r.orig[2] = -1000.0f;
            r.dir[0] = rand_dir(rng);
            r.dir[1] = rand_dir(rng);
            r.dir[2] = std::abs(rand_dir(rng)) + 0.1f;
            float len = std::sqrt(r.dir[0]*r.dir[0] + r.dir[1]*r.dir[1] + r.dir[2]*r.dir[2]);
            r.dir[0] /= len; r.dir[1] /= len; r.dir[2] /= len;
        }
        r.inv_dir[0] = 1.0f / r.dir[0];
        r.inv_dir[1] = 1.0f / r.dir[1];
        r.inv_dir[2] = 1.0f / r.dir[2];
        rays.push_back(r);
    }
    return rays;
}

static void print_result(const SceneResult& res, const char* ray_type) {
    double ratio = res.embree_time_ms / res.ns_time_ms;
    double ns_rps = 100000.0 / (res.ns_time_ms / 1000.0);
    double em_rps = 100000.0 / (res.embree_time_ms / 1000.0);
    std::cout << "=== " << res.name << " ===" << std::endl;
    std::cout << "  Objects: " << res.total_objects
              << " | Occupied cells: " << res.occupied_cells
              << "/" << (GX*GY*GZ)
              << " | Rays: 100000 (" << ray_type << ")" << std::endl;
    std::cout << "  NS BVH8:  " << std::fixed << std::setprecision(2) << res.ns_time_ms
              << " ms | " << std::setprecision(0) << ns_rps << " rays/sec | Hits: " << res.ns_hits << std::endl;
    std::cout << "  Embree:   " << std::fixed << std::setprecision(2) << res.embree_time_ms
              << " ms | " << std::setprecision(0) << em_rps << " rays/sec | Hits: " << res.embree_hits << std::endl;
    if (ratio > 1.0)
        std::cout << "  NS vs Embree: " << std::fixed << std::setprecision(2) << ratio << "x faster" << std::endl;
    else
        std::cout << "  NS vs Embree: " << std::fixed << std::setprecision(2) << (1.0/ratio) << "x slower" << std::endl;
    std::cout << std::endl;
}

int main() {
    std::mt19937 rng(42);
    const int NUM_RAYS = 100000;

    std::cout << "=== NS-BVH vs Embree4 — Three Scene Types ===" << std::endl;
    std::cout << "Grid: " << GX << "x" << GY << "x" << GZ
              << " cells (" << CELL_W << "x" << CELL_H << "x" << CELL_D << " units)" << std::endl;
    std::cout << "Rays: " << NUM_RAYS << " per scene" << std::endl;
    std::cout << std::endl;

    // --- Scene 1: Dense Uniform (random, no clustering) ---
    // Every grid cell occupied — NSBVH loss case (no empty space to skip)
    {
        std::vector<AABB3> objects;
        std::uniform_real_distribution<float> x_dist(10, 2490), y_dist(10, 1990), z_dist(-10, 10);
        for (int i = 0; i < 10000; i++) {
            float x = x_dist(rng), y = y_dist(rng), z = z_dist(rng);
            objects.push_back({{x-1, y-1, z-1}, {x+1, y+1, z+1}});
        }
        auto rays = generate_random_rays(rng, NUM_RAYS);
        auto res = run_scene("Dense Uniform (random, no clustering)", objects, rays);
        print_result(res, "random");
    }

    // --- Scene 2: Sparse Clustered (20 tight clusters at cell centers, empty space between) ---
    // 20/80 occupied cells — NSBVH win case (grid skips 75% of cells)
    // Targeted rays: 70% aimed at clusters from fixed origin, 30% random
    {
        std::vector<AABB3> objects;
        std::vector<std::array<float,3>> centers;
        for (int c = 0; c < 20; c++) {
            int cell_x = c % 10;
            int cell_y = c / 10;
            float cx = cell_x * CELL_W + CELL_W * 0.5f;
            float cy = cell_y * CELL_H + CELL_H * 0.5f;
            centers.push_back({cx, cy, 0.0f});
            for (int i = 0; i < 500; i++) {
                std::uniform_real_distribution<float> jitter(-5.0f, 5.0f);
                float x = cx + jitter(rng), y = cy + jitter(rng), z = jitter(rng);
                objects.push_back({{x-1, y-1, z-1}, {x+1, y+1, z+1}});
            }
        }
        auto rays = generate_targeted_rays(rng, NUM_RAYS, centers);
        auto res = run_scene("Sparse Clustered (20 clusters at cell centers)", objects, rays);
        print_result(res, "70% targeted + 30% random");
    }

    // --- Scene 3: Mixed (10 clusters at cell centers + 5000 random fill) ---
    {
        std::vector<AABB3> objects;
        std::vector<std::array<float,3>> centers;
        for (int c = 0; c < 10; c++) {
            int cell_x = c % 10;
            int cell_y = c / 10;
            float cx = cell_x * CELL_W + CELL_W * 0.5f;
            float cy = cell_y * CELL_H + CELL_H * 0.5f;
            centers.push_back({cx, cy, 0.0f});
            for (int i = 0; i < 500; i++) {
                std::uniform_real_distribution<float> jitter(-5.0f, 5.0f);
                float x = cx + jitter(rng), y = cy + jitter(rng), z = jitter(rng);
                objects.push_back({{x-1, y-1, z-1}, {x+1, y+1, z+1}});
            }
        }
        std::uniform_real_distribution<float> x_dist(10, 2490), y_dist(10, 1990), z_dist(-10, 10);
        for (int i = 0; i < 5000; i++) {
            float x = x_dist(rng), y = y_dist(rng), z = z_dist(rng);
            objects.push_back({{x-1, y-1, z-1}, {x+1, y+1, z+1}});
        }
        auto rays = generate_targeted_rays(rng, NUM_RAYS, centers);
        auto res = run_scene("Mixed (10 clusters + 5K random)", objects, rays);
        print_result(res, "70% targeted + 30% random");
    }

    std::cout << "AVX2 leaf: " << avx2_leaf_count << " | Scalar leaf: " << scalar_leaf_count << std::endl;
    std::cout << "AVX2 internal: " << avx2_internal_count << " | Scalar internal: " << scalar_internal_count << std::endl;
    std::cout << "AVX-512 path: " << (__builtin_cpu_supports("avx512f") ? "YES" : "NO") << std::endl;

    return 0;
}
