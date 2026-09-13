// Copyright (c) 2026 Orders of Magnitude LLC
// Licensed under the OOM Commercial License v1.0
// See LICENSE.md in the repository root or ofmagnitude.com

#include <iostream>
#include <vector>
#include <array>
#include <random>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <cfloat>
#include <cstdint>
#include <immintrin.h>
#include <embree4/rtcore.h>
#include "../NSHash/nshash.hpp"
#include "nsray.hpp"

const int N_OBJECTS = 10000;
const int N_CLUSTERS = 20;
const int N_RAYS = 100000;

struct AABB {
    float min[3], max[3];
};

struct Ray {
    float orig[3], dir[3], inv_dir[3];
};

// Standard slab intersection test
bool ray_aabb_intersect(const Ray& r, const AABB& box) {
    float tmin = -INFINITY, tmax = INFINITY;
    
    for (int i = 0; i < 3; i++) {
        if (std::abs(r.dir[i]) > 1e-6f) {
            float t1 = (box.min[i] - r.orig[i]) * r.inv_dir[i];
            float t2 = (box.max[i] - r.orig[i]) * r.inv_dir[i];
            tmin = std::max(tmin, std::min(t1, t2));
            tmax = std::min(tmax, std::max(t1, t2));
        } else if (r.orig[i] < box.min[i] || r.orig[i] > box.max[i]) {
            return false;
        }
    }
    
    return tmax >= tmin && tmax >= 0;
}

struct BVH16Internal {
    float min_x[16], max_x[16];
    float min_y[16], max_y[16];
    float min_z[16], max_z[16];
    int children[16];   // positive = internal index, negative = -(leaf_index+1)
    int child_count;
};

struct QBVH16Internal {
    // Parent bounds (for dequantization)
    float par_min[3], par_max[3];          // 24 bytes
    // Per-child quantized bounds (8-bit, 0-255 relative to parent)
    uint8_t qmin_x[16], qmax_x[16];       // 32 bytes
    uint8_t qmin_y[16], qmax_y[16];
    uint8_t qmin_z[16], qmax_z[16];       // 96 bytes total
    int children[16];                      // 64 bytes
    int child_count;                       // 4 bytes
};                                         // ~188 bytes vs 456: 2.4x smaller

struct BVH16Leaf {
    int obj_indices[4];
    int obj_count;
};

struct NSBVH {
    bool is_quantized;
    std::vector<BVH16Internal> internal;
    std::vector<QBVH16Internal> qinternal;
    std::vector<BVH16Leaf> leaves;
    std::vector<AABB> clipped_objects;
    int root;
};

// Helper: compute surface area of AABB
float surface_area(const AABB& box) {
    float dx = box.max[0] - box.min[0];
    float dy = box.max[1] - box.min[1];
    float dz = box.max[2] - box.min[2];
    return 2.0f * (dx * dy + dy * dz + dz * dx);
}

// Helper: compute bounds of a set of indices
AABB compute_bounds_indices(const std::vector<AABB>& objects, const std::vector<int>& indices) {
    if (indices.empty()) return {{0,0,0},{0,0,0}};
    AABB bounds = objects[indices[0]];
    for (size_t i = 1; i < indices.size(); i++) {
        for (int j = 0; j < 3; j++) {
            bounds.min[j] = std::min(bounds.min[j], objects[indices[i]].min[j]);
            bounds.max[j] = std::max(bounds.max[j], objects[indices[i]].max[j]);
        }
    }
    return bounds;
}

// Split result struct
struct SplitResult {
    std::vector<int> left_indices;
    std::vector<int> right_indices;
    bool is_spatial;
    int axis;
    float split_pos;
    std::vector<AABB> clipped_objects;
};

// Binned SAH split: 32 bins per axis, O(n) per node.
// Replaces O(n log n) sort-based SAH. No spatial splits.
SplitResult sah_binary_split(
    const std::vector<AABB>& objects, const std::vector<int>& indices,
    std::vector<AABB>& clipped_objects) {
    
    if (indices.size() <= 1) {
        return {indices, {}, false, -1, 0.0f, {}};
    }
    
    constexpr int B = 32;
    AABB parent_bounds = compute_bounds_indices(objects, indices);
    float parent_sa = surface_area(parent_bounds);
    float leaf_cost = (float)indices.size();
    
    float best_cost = leaf_cost;
    int best_axis = -1;
    int best_split_bin = -1;
    
    for (int axis = 0; axis < 3; axis++) {
        float centroid_min = INFINITY, centroid_max = -INFINITY;
        for (int idx : indices) {
            float c = (objects[idx].min[axis] + objects[idx].max[axis]) * 0.5f;
            centroid_min = std::min(centroid_min, c);
            centroid_max = std::max(centroid_max, c);
        }
        
        if (centroid_min == centroid_max) continue;
        
        float scale = (float)B / (centroid_max - centroid_min);
        
        int bin_count[B] = {};
        AABB bin_bounds[B];
        for (int i = 0; i < B; i++)
            bin_bounds[i] = {INFINITY, INFINITY, INFINITY, -INFINITY, -INFINITY, -INFINITY};
        
        for (int idx : indices) {
            float c = (objects[idx].min[axis] + objects[idx].max[axis]) * 0.5f;
            int bin = std::min((int)((c - centroid_min) * scale), B - 1);
            if (bin < 0) bin = 0;
            bin_count[bin]++;
            for (int j = 0; j < 3; j++) {
                bin_bounds[bin].min[j] = std::min(bin_bounds[bin].min[j], objects[idx].min[j]);
                bin_bounds[bin].max[j] = std::max(bin_bounds[bin].max[j], objects[idx].max[j]);
            }
        }
        
        AABB prefix_bounds[B];
        int prefix_count[B];
        prefix_count[0] = bin_count[0];
        prefix_bounds[0] = bin_bounds[0];
        for (int i = 1; i < B; i++) {
            prefix_count[i] = prefix_count[i-1] + bin_count[i];
            prefix_bounds[i] = prefix_bounds[i-1];
            for (int j = 0; j < 3; j++) {
                prefix_bounds[i].min[j] = std::min(prefix_bounds[i].min[j], bin_bounds[i].min[j]);
                prefix_bounds[i].max[j] = std::max(prefix_bounds[i].max[j], bin_bounds[i].max[j]);
            }
        }
        
        AABB suffix_bounds[B];
        int suffix_count[B];
        suffix_count[B-1] = bin_count[B-1];
        suffix_bounds[B-1] = bin_bounds[B-1];
        for (int i = B-2; i >= 0; i--) {
            suffix_count[i] = suffix_count[i+1] + bin_count[i];
            suffix_bounds[i] = suffix_bounds[i+1];
            for (int j = 0; j < 3; j++) {
                suffix_bounds[i].min[j] = std::min(suffix_bounds[i].min[j], bin_bounds[i].min[j]);
                suffix_bounds[i].max[j] = std::max(suffix_bounds[i].max[j], bin_bounds[i].max[j]);
            }
        }
        
        for (int i = 0; i < B - 1; i++) {
            if (prefix_count[i] == 0 || suffix_count[i+1] == 0) continue;
            float left_sa = surface_area(prefix_bounds[i]);
            float right_sa = surface_area(suffix_bounds[i+1]);
            float cost = (prefix_count[i] * left_sa + suffix_count[i+1] * right_sa) / parent_sa;
            if (cost < best_cost) {
                best_cost = cost;
                best_axis = axis;
                best_split_bin = i;
            }
        }
    }
    
    if (best_axis == -1) {
        return {indices, {}, false, -1, 0.0f, {}};
    }
    
    float centroid_min = INFINITY, centroid_max = -INFINITY;
    for (int idx : indices) {
        float c = (objects[idx].min[best_axis] + objects[idx].max[best_axis]) * 0.5f;
        centroid_min = std::min(centroid_min, c);
        centroid_max = std::max(centroid_max, c);
    }
    float scale = (float)B / (centroid_max - centroid_min);
    
    std::vector<int> left_indices, right_indices;
    for (int idx : indices) {
        float c = (objects[idx].min[best_axis] + objects[idx].max[best_axis]) * 0.5f;
        int bin = std::min((int)((c - centroid_min) * scale), B - 1);
        if (bin < 0) bin = 0;
        if (bin <= best_split_bin)
            left_indices.push_back(idx);
        else
            right_indices.push_back(idx);
    }
    
    return {left_indices, right_indices, false, best_axis, 0.0f, {}};
}

// Build BVH16 using SAH: split into 16 children by doing four binary SAH splits (unquantized)
int build_bvh16_unquantized(std::vector<BVH16Internal>& internal, std::vector<BVH16Leaf>& leaves,
                            const std::vector<AABB>& objects, const std::vector<AABB>& clipped_objects,
                            const std::vector<int>& indices, int depth = 0) {
    // If we have <= 4 objects, make a leaf
    if (indices.size() <= 4) {
        int leaf_idx = leaves.size();
        leaves.push_back({});
        BVH16Leaf& leaf = leaves[leaf_idx];
        leaf.obj_count = indices.size();
        for (size_t i = 0; i < indices.size(); i++) {
            leaf.obj_indices[i] = indices[i];
        }
        return -(leaf_idx + 1);  // Return negative to indicate leaf
    }
    
    // First SAH binary split
    std::vector<AABB> local_clipped;
    auto split1 = sah_binary_split(objects, indices, local_clipped);
    auto left_half = split1.left_indices;
    auto right_half = split1.right_indices;
    
    // If first split didn't divide, make leaf
    if (left_half.empty() || right_half.empty()) {
        int leaf_idx = leaves.size();
        leaves.push_back({});
        BVH16Leaf& leaf = leaves[leaf_idx];
        leaf.obj_count = std::min((int)indices.size(), 4);
        for (int i = 0; i < leaf.obj_count; i++) {
            leaf.obj_indices[i] = indices[i];
        }
        return -(leaf_idx + 1);
    }
    
    // Second SAH binary split on each half
    auto split2_left = sah_binary_split(objects, left_half, local_clipped);
    auto split2_right = sah_binary_split(objects, right_half, local_clipped);
    auto left_q1 = split2_left.left_indices;
    auto left_q2 = split2_left.right_indices;
    auto right_q3 = split2_right.left_indices;
    auto right_q4 = split2_right.right_indices;
    
    // Third SAH binary split on each quarter
    auto split3_q1 = sah_binary_split(objects, left_q1, local_clipped);
    auto split3_q2 = sah_binary_split(objects, left_q2, local_clipped);
    auto split3_q3 = sah_binary_split(objects, right_q3, local_clipped);
    auto split3_q4 = sah_binary_split(objects, right_q4, local_clipped);
    auto left_q1_a = split3_q1.left_indices;
    auto left_q1_b = split3_q1.right_indices;
    auto left_q2_a = split3_q2.left_indices;
    auto left_q2_b = split3_q2.right_indices;
    auto right_q3_a = split3_q3.left_indices;
    auto right_q3_b = split3_q3.right_indices;
    auto right_q4_a = split3_q4.left_indices;
    auto right_q4_b = split3_q4.right_indices;
    
    // Fourth SAH binary split on each eighth
    auto split4_q1a = sah_binary_split(objects, left_q1_a, local_clipped);
    auto split4_q1b = sah_binary_split(objects, left_q1_b, local_clipped);
    auto split4_q2a = sah_binary_split(objects, left_q2_a, local_clipped);
    auto split4_q2b = sah_binary_split(objects, left_q2_b, local_clipped);
    auto split4_q3a = sah_binary_split(objects, right_q3_a, local_clipped);
    auto split4_q3b = sah_binary_split(objects, right_q3_b, local_clipped);
    auto split4_q4a = sah_binary_split(objects, right_q4_a, local_clipped);
    auto split4_q4b = sah_binary_split(objects, right_q4_b, local_clipped);
    auto left_q1_a_1 = split4_q1a.left_indices;
    auto left_q1_a_2 = split4_q1a.right_indices;
    auto left_q1_b_1 = split4_q1b.left_indices;
    auto left_q1_b_2 = split4_q1b.right_indices;
    auto left_q2_a_1 = split4_q2a.left_indices;
    auto left_q2_a_2 = split4_q2a.right_indices;
    auto left_q2_b_1 = split4_q2b.left_indices;
    auto left_q2_b_2 = split4_q2b.right_indices;
    auto right_q3_a_1 = split4_q3a.left_indices;
    auto right_q3_a_2 = split4_q3a.right_indices;
    auto right_q3_b_1 = split4_q3b.left_indices;
    auto right_q3_b_2 = split4_q3b.right_indices;
    auto right_q4_a_1 = split4_q4a.left_indices;
    auto right_q4_a_2 = split4_q4a.right_indices;
    auto right_q4_b_1 = split4_q4b.left_indices;
    auto right_q4_b_2 = split4_q4b.right_indices;
    
    // Collect up to 16 non-empty groups
    std::vector<std::vector<int>> groups;
    if (!left_q1_a_1.empty()) groups.push_back(left_q1_a_1);
    if (!left_q1_a_2.empty()) groups.push_back(left_q1_a_2);
    if (!left_q1_b_1.empty()) groups.push_back(left_q1_b_1);
    if (!left_q1_b_2.empty()) groups.push_back(left_q1_b_2);
    if (!left_q2_a_1.empty()) groups.push_back(left_q2_a_1);
    if (!left_q2_a_2.empty()) groups.push_back(left_q2_a_2);
    if (!left_q2_b_1.empty()) groups.push_back(left_q2_b_1);
    if (!left_q2_b_2.empty()) groups.push_back(left_q2_b_2);
    if (!right_q3_a_1.empty()) groups.push_back(right_q3_a_1);
    if (!right_q3_a_2.empty()) groups.push_back(right_q3_a_2);
    if (!right_q3_b_1.empty()) groups.push_back(right_q3_b_1);
    if (!right_q3_b_2.empty()) groups.push_back(right_q3_b_2);
    if (!right_q4_a_1.empty()) groups.push_back(right_q4_a_1);
    if (!right_q4_a_2.empty()) groups.push_back(right_q4_a_2);
    if (!right_q4_b_1.empty()) groups.push_back(right_q4_b_1);
    if (!right_q4_b_2.empty()) groups.push_back(right_q4_b_2);
    
    // If we ended up with just 1 group, make leaf
    if (groups.size() == 1) {
        int leaf_idx = leaves.size();
        leaves.push_back({});
        BVH16Leaf& leaf = leaves[leaf_idx];
        leaf.obj_count = std::min((int)indices.size(), 4);
        for (int i = 0; i < leaf.obj_count; i++) {
            leaf.obj_indices[i] = indices[i];
        }
        return -(leaf_idx + 1);
    }
    
    // Reserve parent slot
    int node_idx = internal.size();
    internal.push_back({});
    
    // Build all children first, collecting info
    std::vector<int> child_indices;
    for (auto& group : groups) {
        child_indices.push_back(build_bvh16_unquantized(internal, leaves, objects, clipped_objects, group, depth + 1));
    }
    
    // Now fill parent node - get fresh references after all push_backs done
    {
        BVH16Internal& node = internal[node_idx];
        node.child_count = groups.size();
        int child_slot = 0;
        for (size_t g = 0; g < groups.size(); g++) {
            int child_idx = child_indices[g];
            
            // Compute bounds for this child
            float cmin[3] = {INFINITY, INFINITY, INFINITY};
            float cmax[3] = {-INFINITY, -INFINITY, -INFINITY};
            
            if (child_idx < 0) {
                // Leaf - compute bounds from objects (may be clipped)
                int leaf_idx = -(child_idx + 1);
                BVH16Leaf& leaf = leaves[leaf_idx];
                for (int i = 0; i < leaf.obj_count; i++) {
                    int obj_idx = leaf.obj_indices[i];
                    const AABB& obj = (obj_idx < (int)objects.size()) ? objects[obj_idx] : clipped_objects[obj_idx - objects.size()];
                    for (int j = 0; j < 3; j++) {
                        cmin[j] = std::min(cmin[j], obj.min[j]);
                        cmax[j] = std::max(cmax[j], obj.max[j]);
                    }
                }
            } else {
                // Internal node - compute bounds from child
                BVH16Internal& child = internal[child_idx];
                for (int c = 0; c < child.child_count; c++) {
                    cmin[0] = std::min(cmin[0], child.min_x[c]);
                    cmax[0] = std::max(cmax[0], child.max_x[c]);
                    cmin[1] = std::min(cmin[1], child.min_y[c]);
                    cmax[1] = std::max(cmax[1], child.max_y[c]);
                    cmin[2] = std::min(cmin[2], child.min_z[c]);
                    cmax[2] = std::max(cmax[2], child.max_z[c]);
                }
            }
            
            node.min_x[child_slot] = cmin[0]; node.max_x[child_slot] = cmax[0];
            node.min_y[child_slot] = cmin[1]; node.max_y[child_slot] = cmax[1];
            node.min_z[child_slot] = cmin[2]; node.max_z[child_slot] = cmax[2];
            node.children[child_slot] = child_idx;
            child_slot++;
        }
    }
    
    return node_idx;
}

// Build BVH16 using SAH: split into 16 children by doing four binary SAH splits (quantized)
int build_bvh16_quantized(std::vector<QBVH16Internal>& internal, std::vector<BVH16Leaf>& leaves,
                          const std::vector<AABB>& objects, const std::vector<AABB>& clipped_objects,
                          const std::vector<int>& indices, int depth = 0) {
    // If we have <= 4 objects, make a leaf
    if (indices.size() <= 4) {
        int leaf_idx = leaves.size();
        leaves.push_back({});
        BVH16Leaf& leaf = leaves[leaf_idx];
        leaf.obj_count = indices.size();
        for (size_t i = 0; i < indices.size(); i++) {
            leaf.obj_indices[i] = indices[i];
        }
        return -(leaf_idx + 1);  // Return negative to indicate leaf
    }
    
    // First SAH binary split
    std::vector<AABB> local_clipped;
    auto split1 = sah_binary_split(objects, indices, local_clipped);
    auto left_half = split1.left_indices;
    auto right_half = split1.right_indices;
    
    // If first split didn't divide, make leaf
    if (left_half.empty() || right_half.empty()) {
        int leaf_idx = leaves.size();
        leaves.push_back({});
        BVH16Leaf& leaf = leaves[leaf_idx];
        leaf.obj_count = std::min((int)indices.size(), 4);
        for (int i = 0; i < leaf.obj_count; i++) {
            leaf.obj_indices[i] = indices[i];
        }
        return -(leaf_idx + 1);
    }
    
    // Second SAH binary split on each half
    auto split2_left = sah_binary_split(objects, left_half, local_clipped);
    auto split2_right = sah_binary_split(objects, right_half, local_clipped);
    auto left_q1 = split2_left.left_indices;
    auto left_q2 = split2_left.right_indices;
    auto right_q3 = split2_right.left_indices;
    auto right_q4 = split2_right.right_indices;
    
    // Third SAH binary split on each quarter
    auto split3_q1 = sah_binary_split(objects, left_q1, local_clipped);
    auto split3_q2 = sah_binary_split(objects, left_q2, local_clipped);
    auto split3_q3 = sah_binary_split(objects, right_q3, local_clipped);
    auto split3_q4 = sah_binary_split(objects, right_q4, local_clipped);
    auto left_q1_a = split3_q1.left_indices;
    auto left_q1_b = split3_q1.right_indices;
    auto left_q2_a = split3_q2.left_indices;
    auto left_q2_b = split3_q2.right_indices;
    auto right_q3_a = split3_q3.left_indices;
    auto right_q3_b = split3_q3.right_indices;
    auto right_q4_a = split3_q4.left_indices;
    auto right_q4_b = split3_q4.right_indices;
    
    // Fourth SAH binary split on each eighth
    auto split4_q1a = sah_binary_split(objects, left_q1_a, local_clipped);
    auto split4_q1b = sah_binary_split(objects, left_q1_b, local_clipped);
    auto split4_q2a = sah_binary_split(objects, left_q2_a, local_clipped);
    auto split4_q2b = sah_binary_split(objects, left_q2_b, local_clipped);
    auto split4_q3a = sah_binary_split(objects, right_q3_a, local_clipped);
    auto split4_q3b = sah_binary_split(objects, right_q3_b, local_clipped);
    auto split4_q4a = sah_binary_split(objects, right_q4_a, local_clipped);
    auto split4_q4b = sah_binary_split(objects, right_q4_b, local_clipped);
    auto left_q1_a_1 = split4_q1a.left_indices;
    auto left_q1_a_2 = split4_q1a.right_indices;
    auto left_q1_b_1 = split4_q1b.left_indices;
    auto left_q1_b_2 = split4_q1b.right_indices;
    auto left_q2_a_1 = split4_q2a.left_indices;
    auto left_q2_a_2 = split4_q2a.right_indices;
    auto left_q2_b_1 = split4_q2b.left_indices;
    auto left_q2_b_2 = split4_q2b.right_indices;
    auto right_q3_a_1 = split4_q3a.left_indices;
    auto right_q3_a_2 = split4_q3a.right_indices;
    auto right_q3_b_1 = split4_q3b.left_indices;
    auto right_q3_b_2 = split4_q3b.right_indices;
    auto right_q4_a_1 = split4_q4a.left_indices;
    auto right_q4_a_2 = split4_q4a.right_indices;
    auto right_q4_b_1 = split4_q4b.left_indices;
    auto right_q4_b_2 = split4_q4b.right_indices;
    
    // Collect up to 16 non-empty groups
    std::vector<std::vector<int>> groups;
    if (!left_q1_a_1.empty()) groups.push_back(left_q1_a_1);
    if (!left_q1_a_2.empty()) groups.push_back(left_q1_a_2);
    if (!left_q1_b_1.empty()) groups.push_back(left_q1_b_1);
    if (!left_q1_b_2.empty()) groups.push_back(left_q1_b_2);
    if (!left_q2_a_1.empty()) groups.push_back(left_q2_a_1);
    if (!left_q2_a_2.empty()) groups.push_back(left_q2_a_2);
    if (!left_q2_b_1.empty()) groups.push_back(left_q2_b_1);
    if (!left_q2_b_2.empty()) groups.push_back(left_q2_b_2);
    if (!right_q3_a_1.empty()) groups.push_back(right_q3_a_1);
    if (!right_q3_a_2.empty()) groups.push_back(right_q3_a_2);
    if (!right_q3_b_1.empty()) groups.push_back(right_q3_b_1);
    if (!right_q3_b_2.empty()) groups.push_back(right_q3_b_2);
    if (!right_q4_a_1.empty()) groups.push_back(right_q4_a_1);
    if (!right_q4_a_2.empty()) groups.push_back(right_q4_a_2);
    if (!right_q4_b_1.empty()) groups.push_back(right_q4_b_1);
    if (!right_q4_b_2.empty()) groups.push_back(right_q4_b_2);
    
    // If we ended up with just 1 group, make leaf
    if (groups.size() == 1) {
        int leaf_idx = leaves.size();
        leaves.push_back({});
        BVH16Leaf& leaf = leaves[leaf_idx];
        leaf.obj_count = std::min((int)indices.size(), 4);
        for (int i = 0; i < leaf.obj_count; i++) {
            leaf.obj_indices[i] = indices[i];
        }
        return -(leaf_idx + 1);
    }
    
    // Reserve parent slot
    int node_idx = internal.size();
    internal.push_back({});
    
    // Build all children first, collecting info
    std::vector<int> child_indices;
    std::vector<std::array<float,3>> child_mins, child_maxs;
    for (auto& group : groups) {
        child_indices.push_back(build_bvh16_quantized(internal, leaves, objects, clipped_objects, group, depth + 1));
    }
    
    // Compute child bounds first
    for (size_t g = 0; g < groups.size(); g++) {
        int child_idx = child_indices[g];
        float cmin[3] = {INFINITY, INFINITY, INFINITY};
        float cmax[3] = {-INFINITY, -INFINITY, -INFINITY};
        
        if (child_idx < 0) {
            int leaf_idx = -(child_idx + 1);
            BVH16Leaf& leaf = leaves[leaf_idx];
            for (int i = 0; i < leaf.obj_count; i++) {
                int obj_idx = leaf.obj_indices[i];
                const AABB& obj = (obj_idx < (int)objects.size()) ? objects[obj_idx] : clipped_objects[obj_idx - objects.size()];
                for (int j = 0; j < 3; j++) {
                    cmin[j] = std::min(cmin[j], obj.min[j]);
                    cmax[j] = std::max(cmax[j], obj.max[j]);
                }
            }
        } else {
            QBVH16Internal& child = internal[child_idx];
            for (int c = 0; c < child.child_count; c++) {
                for (int j = 0; j < 3; j++) {
                    float scale = (child.par_max[j] - child.par_min[j]) / 255.0f;
                    cmin[j] = std::min(cmin[j], child.par_min[j] + child.qmin_x[c] * scale);
                    cmax[j] = std::max(cmax[j], child.par_min[j] + child.qmax_x[c] * scale);
                }
            }
        }
        child_mins.push_back({cmin[0], cmin[1], cmin[2]});
        child_maxs.push_back({cmax[0], cmax[1], cmax[2]});
    }
    
    // Compute parent bounds from all children
    float par_min[3] = {INFINITY, INFINITY, INFINITY};
    float par_max[3] = {-INFINITY, -INFINITY, -INFINITY};
    for (size_t g = 0; g < groups.size(); g++) {
        for (int j = 0; j < 3; j++) {
            par_min[j] = std::min(par_min[j], child_mins[g][j]);
            par_max[j] = std::max(par_max[j], child_maxs[g][j]);
        }
    }
    
    // Now fill parent node with quantized bounds
    {
        QBVH16Internal& node = internal[node_idx];
        node.child_count = groups.size();
        for (int j = 0; j < 3; j++) {
            node.par_min[j] = par_min[j];
            node.par_max[j] = par_max[j];
        }
        
        for (size_t g = 0; g < groups.size(); g++) {
            node.children[g] = child_indices[g];
            
            // Quantize child bounds
            for (int j = 0; j < 3; j++) {
                float range = par_max[j] - par_min[j];
                float scale = range > 0 ? 255.0f / range : 0.0f;
                
                int qmin = (int)std::floor((child_mins[g][j] - par_min[j]) * scale) - 1;
                int qmax = (int)std::ceil((child_maxs[g][j] - par_min[j]) * scale) + 1;
                
                if (j == 0) {
                    node.qmin_x[g] = (uint8_t)std::clamp(qmin, 0, 255);
                    node.qmax_x[g] = (uint8_t)std::clamp(qmax, 0, 255);
                } else if (j == 1) {
                    node.qmin_y[g] = (uint8_t)std::clamp(qmin, 0, 255);
                    node.qmax_y[g] = (uint8_t)std::clamp(qmax, 0, 255);
                } else {
                    node.qmin_z[g] = (uint8_t)std::clamp(qmin, 0, 255);
                    node.qmax_z[g] = (uint8_t)std::clamp(qmax, 0, 255);
                }
            }
        }
    }
    
    return node_idx;
}

// Wrapper build function that chooses quantized vs unquantized based on object count
NSBVH build_nsbvh(const std::vector<AABB>& objects) {
    NSBVH bvh;
    std::vector<int> all_indices(objects.size());
    for (size_t i = 0; i < objects.size(); i++) all_indices[i] = i;
    
    if (objects.size() < 50000) {
        bvh.is_quantized = false;
        bvh.root = build_bvh16_unquantized(bvh.internal, bvh.leaves, objects, bvh.clipped_objects, all_indices);
    } else {
        bvh.is_quantized = true;
        bvh.root = build_bvh16_quantized(bvh.qinternal, bvh.leaves, objects, bvh.clipped_objects, all_indices);
    }
    
    return bvh;
}

inline __attribute__((always_inline)) bool traverse_bvh16_unquantized(const std::vector<BVH16Internal>& internal, const std::vector<BVH16Leaf>& leaves,
                                const std::vector<AABB>& objects, const std::vector<AABB>& clipped_objects, int root, const Ray& r, int& visits, int& obj_tests) {
    int stack[64];
    int stack_top = 0;
    stack[stack_top++] = root;

    __m512 ox  = _mm512_set1_ps(r.orig[0]), oy  = _mm512_set1_ps(r.orig[1]), oz  = _mm512_set1_ps(r.orig[2]);
    __m512 idx = _mm512_set1_ps(r.inv_dir[0]), idy = _mm512_set1_ps(r.inv_dir[1]), idz = _mm512_set1_ps(r.inv_dir[2]);
    __m512 zero = _mm512_setzero_ps();

    while (stack_top > 0) {
        int node_idx = stack[--stack_top];
        visits++;
        const BVH16Internal& node = internal[node_idx];

        __m512 t1x = _mm512_mul_ps(_mm512_sub_ps(_mm512_loadu_ps(node.min_x), ox), idx);
        __m512 t2x = _mm512_mul_ps(_mm512_sub_ps(_mm512_loadu_ps(node.max_x), ox), idx);
        __m512 tmin = _mm512_min_ps(t1x, t2x), tmax = _mm512_max_ps(t1x, t2x);

        __m512 t1y = _mm512_mul_ps(_mm512_sub_ps(_mm512_loadu_ps(node.min_y), oy), idy);
        __m512 t2y = _mm512_mul_ps(_mm512_sub_ps(_mm512_loadu_ps(node.max_y), oy), idy);
        tmin = _mm512_max_ps(tmin, _mm512_min_ps(t1y, t2y));
        tmax = _mm512_min_ps(tmax, _mm512_max_ps(t1y, t2y));

        __m512 t1z = _mm512_mul_ps(_mm512_sub_ps(_mm512_loadu_ps(node.min_z), oz), idz);
        __m512 t2z = _mm512_mul_ps(_mm512_sub_ps(_mm512_loadu_ps(node.max_z), oz), idz);
        tmin = _mm512_max_ps(tmin, _mm512_min_ps(t1z, t2z));
        tmax = _mm512_min_ps(tmax, _mm512_max_ps(t1z, t2z));

        __mmask16 hit_mask = _mm512_cmp_ps_mask(tmax, _mm512_max_ps(tmin, zero), _CMP_GE_OQ)
                             & ((1 << node.child_count) - 1);
        if (!hit_mask) continue;

        for (int i = 0; i < node.child_count; i++) {
            if (!(hit_mask & (1 << i))) continue;
            int child_idx = node.children[i];
            if (child_idx < 0) {
                const BVH16Leaf& leaf = leaves[-(child_idx + 1)];
                for (int j = 0; j < leaf.obj_count; j++) {
                    obj_tests++;
                    int obj_idx = leaf.obj_indices[j];
                    const AABB& obj = (obj_idx < (int)objects.size()) ? objects[obj_idx] : clipped_objects[obj_idx - objects.size()];
                    if (ray_aabb_intersect(r, obj)) return true;
                }
            } else {
                stack[stack_top++] = child_idx;
            }
        }
    }
    return false;
}

#ifdef __AVX512F__
inline __attribute__((always_inline)) bool traverse_bvh16_quantized(const std::vector<QBVH16Internal>& internal, const std::vector<BVH16Leaf>& leaves,
                              const std::vector<AABB>& objects, const std::vector<AABB>& clipped_objects, int root, const Ray& r, int& visits, int& obj_tests) {
    int stack[64];
    int stack_top = 0;
    stack[stack_top++] = root;

    __m512 ox  = _mm512_set1_ps(r.orig[0]), oy  = _mm512_set1_ps(r.orig[1]), oz  = _mm512_set1_ps(r.orig[2]);
    __m512 idx = _mm512_set1_ps(r.inv_dir[0]), idy = _mm512_set1_ps(r.inv_dir[1]), idz = _mm512_set1_ps(r.inv_dir[2]);
    __m512 zero = _mm512_setzero_ps();

    while (stack_top > 0) {
        int node_idx = stack[--stack_top];
        visits++;
        const QBVH16Internal& node = internal[node_idx];

        // Dequantize bounds: convert 8-bit to float using parent bounds
        __m512 scale_x = _mm512_set1_ps((node.par_max[0] - node.par_min[0]) / 255.0f);
        __m512 scale_y = _mm512_set1_ps((node.par_max[1] - node.par_min[1]) / 255.0f);
        __m512 scale_z = _mm512_set1_ps((node.par_max[2] - node.par_min[2]) / 255.0f);
        __m512 par_min_x = _mm512_set1_ps(node.par_min[0]);
        __m512 par_min_y = _mm512_set1_ps(node.par_min[1]);
        __m512 par_min_z = _mm512_set1_ps(node.par_min[2]);

        // Load 16 uint8 values, convert to int32, then to float, scale and offset
        __m128i qmin_x_128 = _mm_loadu_si128((__m128i*)node.qmin_x);
        __m128i qmax_x_128 = _mm_loadu_si128((__m128i*)node.qmax_x);
        __m512i qmin_x_512i = _mm512_cvtepu8_epi32(qmin_x_128);
        __m512i qmax_x_512i = _mm512_cvtepu8_epi32(qmax_x_128);
        __m512 qmin_x_512 = _mm512_cvtepi32_ps(qmin_x_512i);
        __m512 qmax_x_512 = _mm512_cvtepi32_ps(qmax_x_512i);
        __m512 min_x = _mm512_fmadd_ps(qmin_x_512, scale_x, par_min_x);
        __m512 max_x = _mm512_fmadd_ps(qmax_x_512, scale_x, par_min_x);

        __m128i qmin_y_128 = _mm_loadu_si128((__m128i*)node.qmin_y);
        __m128i qmax_y_128 = _mm_loadu_si128((__m128i*)node.qmax_y);
        __m512i qmin_y_512i = _mm512_cvtepu8_epi32(qmin_y_128);
        __m512i qmax_y_512i = _mm512_cvtepu8_epi32(qmax_y_128);
        __m512 qmin_y_512 = _mm512_cvtepi32_ps(qmin_y_512i);
        __m512 qmax_y_512 = _mm512_cvtepi32_ps(qmax_y_512i);
        __m512 min_y = _mm512_fmadd_ps(qmin_y_512, scale_y, par_min_y);
        __m512 max_y = _mm512_fmadd_ps(qmax_y_512, scale_y, par_min_y);

        __m128i qmin_z_128 = _mm_loadu_si128((__m128i*)node.qmin_z);
        __m128i qmax_z_128 = _mm_loadu_si128((__m128i*)node.qmax_z);
        __m512i qmin_z_512i = _mm512_cvtepu8_epi32(qmin_z_128);
        __m512i qmax_z_512i = _mm512_cvtepu8_epi32(qmax_z_128);
        __m512 qmin_z_512 = _mm512_cvtepi32_ps(qmin_z_512i);
        __m512 qmax_z_512 = _mm512_cvtepi32_ps(qmax_z_512i);
        __m512 min_z = _mm512_fmadd_ps(qmin_z_512, scale_z, par_min_z);
        __m512 max_z = _mm512_fmadd_ps(qmax_z_512, scale_z, par_min_z);

        // Slab test
        __m512 t1x = _mm512_mul_ps(_mm512_sub_ps(min_x, ox), idx);
        __m512 t2x = _mm512_mul_ps(_mm512_sub_ps(max_x, ox), idx);
        __m512 tmin = _mm512_min_ps(t1x, t2x), tmax = _mm512_max_ps(t1x, t2x);

        __m512 t1y = _mm512_mul_ps(_mm512_sub_ps(min_y, oy), idy);
        __m512 t2y = _mm512_mul_ps(_mm512_sub_ps(max_y, oy), idy);
        tmin = _mm512_max_ps(tmin, _mm512_min_ps(t1y, t2y));
        tmax = _mm512_min_ps(tmax, _mm512_max_ps(t1y, t2y));

        __m512 t1z = _mm512_mul_ps(_mm512_sub_ps(min_z, oz), idz);
        __m512 t2z = _mm512_mul_ps(_mm512_sub_ps(max_z, oz), idz);
        tmin = _mm512_max_ps(tmin, _mm512_min_ps(t1z, t2z));
        tmax = _mm512_min_ps(tmax, _mm512_max_ps(t1z, t2z));

        __mmask16 hit_mask = _mm512_cmp_ps_mask(tmax, _mm512_max_ps(tmin, zero), _CMP_GE_OQ)
                             & ((1 << node.child_count) - 1);
        if (!hit_mask) continue;

        for (int i = 0; i < node.child_count; i++) {
            if (!(hit_mask & (1 << i))) continue;
            int child_idx = node.children[i];
            if (child_idx < 0) {
                const BVH16Leaf& leaf = leaves[-(child_idx + 1)];
                for (int j = 0; j < leaf.obj_count; j++) {
                    obj_tests++;
                    int obj_idx = leaf.obj_indices[j];
                    const AABB& obj = (obj_idx < (int)objects.size()) ? objects[obj_idx] : clipped_objects[obj_idx - objects.size()];
                    if (ray_aabb_intersect(r, obj)) return true;
                }
            } else {
                stack[stack_top++] = child_idx;
            }
        }
    }
    return false;
}
#endif

// Wrapper traverse function that picks the right path based on is_quantized flag
bool traverse_nsbvh(const NSBVH& bvh, const std::vector<AABB>& objects, const std::vector<AABB>& clipped_objects, const Ray& r, int& visits, int& obj_tests) {
    if (bvh.is_quantized) {
#ifdef __AVX512F__
        return traverse_bvh16_quantized(bvh.qinternal, bvh.leaves, objects, clipped_objects, bvh.root, r, visits, obj_tests);
#else
        return traverse_bvh16_quantized_scalar(bvh.qinternal, bvh.leaves, objects, clipped_objects, bvh.root, r, visits, obj_tests);
#endif
    } else {
#ifdef __AVX512F__
        return traverse_bvh16_unquantized(bvh.internal, bvh.leaves, objects, clipped_objects, bvh.root, r, visits, obj_tests);
#else
        return traverse_bvh16_unquantized_scalar(bvh.internal, bvh.leaves, objects, clipped_objects, bvh.root, r, visits, obj_tests);
#endif
    }
}

// Helper function to run benchmarks for a given scene
void run_benchmark(const std::vector<AABB>& objects, const std::vector<Ray>& rays,
                   const std::string& scene_name, bool print_stats = false) {
    // Compute grid bounds from object AABBs with padding
    float scene_min[3] = {FLT_MAX, FLT_MAX, FLT_MAX};
    float scene_max[3] = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
    
    for (const auto& obj : objects) {
        for (int j = 0; j < 3; j++) {
            scene_min[j] = std::min(scene_min[j], obj.min[j]);
            scene_max[j] = std::max(scene_max[j], obj.max[j]);
        }
    }
    
    float scene_max_extent = std::max({scene_max[0]-scene_min[0], 
                                        scene_max[1]-scene_min[1], 
                                        scene_max[2]-scene_min[2]});
    float CELL_SIZE = std::max(50.0f, scene_max_extent / 200.0f);
    
    // Add padding
    for (int j = 0; j < 3; j++) {
        scene_min[j] -= CELL_SIZE;
        scene_max[j] += CELL_SIZE;
    }
    
    // Compute grid dimensions
    float GRID_OFFSET = scene_min[0];
    int GX = (int)std::ceil((scene_max[0] - scene_min[0]) / CELL_SIZE);
    int GY = (int)std::ceil((scene_max[1] - scene_min[1]) / CELL_SIZE);
    int GZ = (int)std::ceil((scene_max[2] - scene_min[2]) / CELL_SIZE);
    
    // COMPETITOR 2: NS-BVH (quantized or unquantized based on object count)
    NSBVH bvh = build_nsbvh(objects);
    
    // Flush CPU cache before BVH16 timed section
    static std::vector<char> flush_buf(32 * 1024 * 1024, 1); // 32MB
    volatile char sink = 0;
    for (char c : flush_buf) sink += c;
    
    auto t2 = std::chrono::high_resolution_clock::now();
    int bvh_hits = 0, bvh_visits = 0, bvh_obj_tests = 0;
    for (const auto& r : rays) {
        if (traverse_nsbvh(bvh, objects, bvh.clipped_objects, r, bvh_visits, bvh_obj_tests)) {
            bvh_hits++;
        }
    }
    auto t3 = std::chrono::high_resolution_clock::now();
    double bvh_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
    
    // COMPETITOR 2.5: Embree 4 user geometry
    struct AABBUserData {
        const AABB* objects;
        size_t n;
    };
    
    AABBUserData* ud = new AABBUserData{objects.data(), objects.size()};
    
    RTCDevice device = rtcNewDevice(nullptr);
    RTCScene scene_e = rtcNewScene(device);
    
    RTCGeometry geom = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_USER);
    rtcSetGeometryUserPrimitiveCount(geom, objects.size());
    rtcSetGeometryUserData(geom, ud);
    
    // Bounds callback
    rtcSetGeometryBoundsFunction(geom,
        [](const RTCBoundsFunctionArguments* args) {
            auto* ud = (AABBUserData*)args->geometryUserPtr;
            const AABB& obj = ud->objects[args->primID];
            args->bounds_o->lower_x = obj.min[0];
            args->bounds_o->lower_y = obj.min[1];
            args->bounds_o->lower_z = obj.min[2];
            args->bounds_o->upper_x = obj.max[0];
            args->bounds_o->upper_y = obj.max[1];
            args->bounds_o->upper_z = obj.max[2];
        },
        nullptr
    );
    
    // Intersect callback - slab test
    rtcSetGeometryIntersectFunction(geom,
        [](const RTCIntersectFunctionNArguments* args) {
            if (args->N != 1) return;
            auto* ud = (AABBUserData*)args->geometryUserPtr;
            const AABB& obj = ud->objects[args->primID];
            
            RTCRayHit* rayhit = (RTCRayHit*)args->rayhit;
            float org_x = rayhit->ray.org_x;
            float org_y = rayhit->ray.org_y;
            float org_z = rayhit->ray.org_z;
            float dir_x = rayhit->ray.dir_x;
            float dir_y = rayhit->ray.dir_y;
            float dir_z = rayhit->ray.dir_z;
            float tfar = rayhit->ray.tfar;
            
            float tmin = -INFINITY, tmax = INFINITY;
            
            for (int i = 0; i < 3; i++) {
                float d = (i == 0) ? dir_x : (i == 1) ? dir_y : dir_z;
                float o = (i == 0) ? org_x : (i == 1) ? org_y : org_z;
                float omin = obj.min[i];
                float omax = obj.max[i];
                
                if (std::abs(d) > 1e-6f) {
                    float t1 = (omin - o) / d;
                    float t2 = (omax - o) / d;
                    tmin = std::max(tmin, std::min(t1, t2));
                    tmax = std::min(tmax, std::max(t1, t2));
                } else if (o < omin || o > omax) {
                    return;
                }
            }
            
            if (tmax >= tmin && tmax >= 0 && tmin < tfar) {
                rayhit->ray.tfar = tmin;
                rayhit->hit.primID = args->primID;
                rayhit->hit.geomID = 0;
                rayhit->hit.u = 0.0f;
                rayhit->hit.v = 0.0f;
            }
        }
    );
    
    rtcCommitGeometry(geom);
    rtcAttachGeometry(scene_e, geom);
    rtcReleaseGeometry(geom);
    rtcCommitScene(scene_e);
    
    // Flush CPU cache before Embree timed section
    for (char c : flush_buf) sink += c;
    
    // Benchmark Embree
    int embree_hits = 0;
    auto t_embree_start = std::chrono::high_resolution_clock::now();
    for (const auto& r : rays) {
        RTCRayHit rayhit;
        rayhit.ray.org_x = r.orig[0];
        rayhit.ray.org_y = r.orig[1];
        rayhit.ray.org_z = r.orig[2];
        rayhit.ray.dir_x = r.dir[0];
        rayhit.ray.dir_y = r.dir[1];
        rayhit.ray.dir_z = r.dir[2];
        rayhit.ray.tnear = 0.0f;
        rayhit.ray.tfar = INFINITY;
        rayhit.ray.mask = -1;
        rayhit.ray.flags = 0;
        rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
        rayhit.hit.primID = RTC_INVALID_GEOMETRY_ID;
        
        rtcIntersect1(scene_e, &rayhit);
        if (rayhit.hit.geomID != RTC_INVALID_GEOMETRY_ID) {
            embree_hits++;
        }
    }
    auto t_embree_end = std::chrono::high_resolution_clock::now();
    double embree_ms = std::chrono::duration<double, std::milli>(t_embree_end - t_embree_start).count();
    
    rtcReleaseScene(scene_e);
    rtcReleaseDevice(device);
    delete ud;
    
    // Output
    std::cout << "=== " << scene_name << " ===" << std::endl;
    std::cout << "  Embree: " << std::fixed << std::setprecision(2) << embree_ms << " ms | Hits: " << embree_hits << std::endl;
    std::cout << "  BVH16:  " << std::fixed << std::setprecision(2) << bvh_ms << " ms | Hits: " << bvh_hits << std::endl;
    
    double speedup_bvh16 = embree_ms / bvh_ms;
    if (speedup_bvh16 > 1.0) {
        std::cout << "  BVH16 vs Embree: " << std::fixed << std::setprecision(2) << speedup_bvh16 << "x faster" << std::endl;
    } else {
        std::cout << "  BVH16 vs Embree: " << std::fixed << std::setprecision(2) << (1.0/speedup_bvh16) << "x slower" << std::endl;
    }
    
    if (print_stats) {
        std::cout << "  BVH16 Traversal Stats:" << std::endl;
        std::cout << "    Avg node visits/ray: " << std::fixed << std::setprecision(2) << (float)bvh_visits / rays.size() << std::endl;
        std::cout << "    Avg obj tests/ray: " << std::fixed << std::setprecision(2) << (float)bvh_obj_tests / rays.size() << std::endl;
    }
    std::cout << std::endl;
}

void generate_scene(std::mt19937& rng, int n_clusters, int n_objects, float spacing,
                    float miss_offset, std::vector<AABB>& objects,
                    std::vector<std::array<float,3>>& cluster_centers,
                    std::vector<Ray>& rays) {
    int grid_w = (int)std::ceil(std::sqrt(n_clusters));
    int grid_h = (int)std::ceil((float)n_clusters / grid_w);
    int objects_per_cluster = n_objects / n_clusters;
    
    std::uniform_real_distribution<float> jitter(-10.0f, 10.0f);
    std::uniform_real_distribution<float> rand_dir(-1.0f, 1.0f);
    std::uniform_int_distribution<int> rand_cluster(0, n_clusters - 1);
    
    for (int c = 0; c < n_clusters; c++) {
        float cx = (c % grid_w) * spacing;
        float cy = ((c / grid_w) % grid_h) * spacing;
        float cz = 0.0f;
        cluster_centers.push_back({cx, cy, cz});
        
        for (int i = 0; i < objects_per_cluster; i++) {
            float x = cx + jitter(rng);
            float y = cy + jitter(rng);
            float z = cz + jitter(rng);
            objects.push_back({{x-1, y-1, z-1}, {x+1, y+1, z+1}});
        }
    }
    
    for (int i = 0; i < N_RAYS; i++) {
        Ray r;
        if (i < N_RAYS / 2) {
            int target_cluster = rand_cluster(rng);
            r.orig[0] = -100.0f;
            r.orig[1] = cluster_centers[target_cluster][1];
            r.orig[2] = 0.0f;
            
            float dx = cluster_centers[target_cluster][0] - r.orig[0];
            float dy = cluster_centers[target_cluster][1] - r.orig[1];
            float dz = cluster_centers[target_cluster][2] - r.orig[2];
            float len = std::sqrt(dx*dx + dy*dy + dz*dz);
            r.dir[0] = dx / len;
            r.dir[1] = dy / len;
            r.dir[2] = dz / len;
        } else {
            int cluster = rand_cluster(rng);
            float cx = cluster_centers[cluster][0] + miss_offset;
            float cy = cluster_centers[cluster][1] + miss_offset;
            r.orig[0] = cx;
            r.orig[1] = cy;
            r.orig[2] = 0.0f;
            
            r.dir[0] = rand_dir(rng);
            r.dir[1] = rand_dir(rng);
            r.dir[2] = rand_dir(rng);
            float len = std::sqrt(r.dir[0]*r.dir[0] + r.dir[1]*r.dir[1] + r.dir[2]*r.dir[2]);
            r.dir[0] /= len;
            r.dir[1] /= len;
            r.dir[2] /= len;
        }
        
        r.inv_dir[0] = 1.0f / r.dir[0];
        r.inv_dir[1] = 1.0f / r.dir[1];
        r.inv_dir[2] = 1.0f / r.dir[2];
        
        rays.push_back(r);
    }
}


// Secondary-ray benchmark: unsorted vs joint-key-sorted (10M rays)
void run_secondary_ray_benchmark(const std::vector<AABB>& objects,
                                  const std::string& scene_name) {
    NSBVH bvh = build_nsbvh(objects);

    constexpr int N_SEC = 10000000;
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dir_dist(-1.0f, 1.0f);

    float smin[3] = {FLT_MAX, FLT_MAX, FLT_MAX};
    float smax[3] = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
    for (const auto& obj : objects) {
        for (int j = 0; j < 3; j++) {
            smin[j] = std::min(smin[j], obj.min[j]);
            smax[j] = std::max(smax[j], obj.max[j]);
        }
    }

    std::uniform_int_distribution<int> obj_dist(0, (int)objects.size() - 1);
    std::vector<Ray> rays;
    rays.reserve(N_SEC);
    for (int i = 0; i < N_SEC; i++) {
        Ray r;
        const AABB& src = objects[obj_dist(rng)];
        r.orig[0] = (src.min[0] + src.max[0]) * 0.5f;
        r.orig[1] = (src.min[1] + src.max[1]) * 0.5f;
        r.orig[2] = (src.min[2] + src.max[2]) * 0.5f;
        r.dir[0] = dir_dist(rng);
        r.dir[1] = dir_dist(rng);
        r.dir[2] = dir_dist(rng);
        float len = std::sqrt(r.dir[0]*r.dir[0] + r.dir[1]*r.dir[1] + r.dir[2]*r.dir[2]);
        r.dir[0] /= len;
        r.dir[1] /= len;
        r.dir[2] /= len;
        r.inv_dir[0] = 1.0f / r.dir[0];
        r.inv_dir[1] = 1.0f / r.dir[1];
        r.inv_dir[2] = 1.0f / r.dir[2];
        rays.push_back(r);
    }

    std::vector<Ray> rays_sorted = rays;

    // Compute keys on Ray array directly
    auto tk0 = std::chrono::high_resolution_clock::now();
    uint32_t* keys = new uint32_t[N_SEC];
    {
        float inv_ext[3] = {
            8.0f / (smax[0] - smin[0]),
            8.0f / (smax[1] - smin[1]),
            8.0f / (smax[2] - smin[2])
        };
        static constexpr uint32_t s3[8] = {0,1,8,9,64,65,72,73};
        static constexpr uint32_t s4[16] = {0,1,8,9,64,65,72,73,512,513,520,521,576,577,584,585};
        for (int i = 0; i < N_SEC; i++) {
            const Ray& r = rays_sorted[i];
            uint32_t cx = std::min(7u, (uint32_t)((r.orig[0] - smin[0]) * inv_ext[0]));
            uint32_t cy = std::min(7u, (uint32_t)((r.orig[1] - smin[1]) * inv_ext[1]));
            uint32_t cz = std::min(7u, (uint32_t)((r.orig[2] - smin[2]) * inv_ext[2]));
            uint32_t cell = s3[cx] | (s3[cy] << 1) | (s3[cz] << 2);
            float inv_len = 1.0f / std::sqrt(r.dir[0]*r.dir[0]+r.dir[1]*r.dir[1]+r.dir[2]*r.dir[2]);
            float nx = r.dir[0]*inv_len, ny = r.dir[1]*inv_len, nz = r.dir[2]*inv_len;
            uint32_t dx = std::min(15u, (uint32_t)((nx*0.5f+0.5f)*15.0f));
            uint32_t dy = std::min(15u, (uint32_t)((ny*0.5f+0.5f)*15.0f));
            uint32_t dz = std::min(15u, (uint32_t)((nz*0.5f+0.5f)*15.0f));
            uint32_t dir = s4[dx] | (s4[dy] << 1) | (s4[dz] << 2);
            keys[i] = (cell << 12) | dir;
        }
    }
    auto tk1 = std::chrono::high_resolution_clock::now();
    double ms_keys = std::chrono::duration<double, std::milli>(tk1 - tk0).count();

    // Convert to NSRay for radix sort
    auto tc0 = std::chrono::high_resolution_clock::now();
    NSRay* nsrays = new NSRay[N_SEC];
    for (int i = 0; i < N_SEC; i++) {
        nsrays[i] = {rays_sorted[i].orig[0], rays_sorted[i].orig[1], rays_sorted[i].orig[2],
                     rays_sorted[i].dir[0], rays_sorted[i].dir[1], rays_sorted[i].dir[2]};
    }
    auto tc1 = std::chrono::high_resolution_clock::now();
    double ms_conv = std::chrono::duration<double, std::milli>(tc1 - tc0).count();

    // Radix sort only (keys pre-computed)
    auto ts0 = std::chrono::high_resolution_clock::now();
    nsray::radix_sort_rays(nsrays, keys, N_SEC);
    auto ts1 = std::chrono::high_resolution_clock::now();
    double ms_radix = std::chrono::duration<double, std::milli>(ts1 - ts0).count();

    // Convert back to Ray + recompute inv_dir
    auto tc2 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N_SEC; i++) {
        rays_sorted[i].orig[0] = nsrays[i].ox;
        rays_sorted[i].orig[1] = nsrays[i].oy;
        rays_sorted[i].orig[2] = nsrays[i].oz;
        rays_sorted[i].dir[0] = nsrays[i].dx;
        rays_sorted[i].dir[1] = nsrays[i].dy;
        rays_sorted[i].dir[2] = nsrays[i].dz;
        rays_sorted[i].inv_dir[0] = 1.0f / rays_sorted[i].dir[0];
        rays_sorted[i].inv_dir[1] = 1.0f / rays_sorted[i].dir[1];
        rays_sorted[i].inv_dir[2] = 1.0f / rays_sorted[i].dir[2];
    }
    auto tc3 = std::chrono::high_resolution_clock::now();
    double ms_back = std::chrono::duration<double, std::milli>(tc3 - tc2).count();

    delete[] nsrays;
    delete[] keys;

    double ms_sort = ms_keys + ms_conv + ms_radix + ms_back;

    // Flush cache
    static std::vector<char> flush_buf(32 * 1024 * 1024, 1);
    volatile char sink = 0;
    for (char c : flush_buf) sink += c;

    // Unsorted traversal
    auto t1 = std::chrono::high_resolution_clock::now();
    int hits_unsorted = 0;
    for (const auto& r : rays) {
        int v = 0, ot = 0;
        if (traverse_nsbvh(bvh, objects, bvh.clipped_objects, r, v, ot))
            hits_unsorted++;
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    double ms_unsorted = std::chrono::duration<double, std::milli>(t2 - t1).count();

    // Flush cache again
    for (char c : flush_buf) sink += c;

    // Sorted traversal
    auto t3 = std::chrono::high_resolution_clock::now();
    int hits_sorted = 0;
    for (const auto& r : rays_sorted) {
        int v = 0, ot = 0;
        if (traverse_nsbvh(bvh, objects, bvh.clipped_objects, r, v, ot))
            hits_sorted++;
    }
    auto t4 = std::chrono::high_resolution_clock::now();
    double ms_sorted = std::chrono::duration<double, std::milli>(t4 - t3).count();

    double rps_unsorted = N_SEC / (ms_unsorted / 1000.0);
    double rps_sorted = N_SEC / (ms_sorted / 1000.0);

    double total_unsorted = ms_unsorted;
    double total_sorted = ms_sort + ms_sorted;

    std::cout << "=== " << scene_name << " — Secondary Rays (10M, Radix Sort) ===" << std::endl;
    std::cout << "  Rays: " << N_SEC << " (random directions, object-center origins)" << std::endl;
    std::cout << "  Key computation:  " << std::fixed << std::setprecision(2) << ms_keys << " ms" << std::endl;
    std::cout << "  NSRay conversion:  " << std::fixed << std::setprecision(2) << ms_conv << " ms" << std::endl;
    std::cout << "  Radix sort only:   " << std::fixed << std::setprecision(2) << ms_radix << " ms" << std::endl;
    std::cout << "  Back-conversion:   " << std::fixed << std::setprecision(2) << ms_back << " ms" << std::endl;
    std::cout << "  Sort total:        " << std::fixed << std::setprecision(2) << ms_sort << " ms" << std::endl;
    std::cout << "  Traversal unsorted: " << std::fixed << std::setprecision(2) << ms_unsorted << " ms | "
              << std::setprecision(2) << rps_unsorted / 1e6 << " Mrays/s | Hits: " << hits_unsorted << std::endl;
    std::cout << "  Traversal sorted:   " << std::fixed << std::setprecision(2) << ms_sorted << " ms | "
              << std::setprecision(2) << rps_sorted / 1e6 << " Mrays/s | Hits: " << hits_sorted << std::endl;
    std::cout << "  Traversal speedup:   " << std::fixed << std::setprecision(2) << ms_unsorted / ms_sorted << "x" << std::endl;
    std::cout << "  Total unsorted: " << std::fixed << std::setprecision(2) << total_unsorted << " ms" << std::endl;
    std::cout << "  Total sorted:   " << std::fixed << std::setprecision(2) << total_sorted << " ms" << std::endl;
    std::cout << "  Total speedup:  " << std::fixed << std::setprecision(2) << total_unsorted / total_sorted << "x" << std::endl;
    std::cout << "  Hit match: " << (hits_unsorted == hits_sorted ? "YES" : "NO") << std::endl;
    std::cout << std::endl;
}

int main() {
#ifdef __AVX512F__
    std::cout << "AVX-512: YES" << std::endl;
#else
    std::cout << "AVX-512: NO (scalar)" << std::endl;
#endif
    std::mt19937 rng(42);
    
    std::cout << "=== NS-BVH Simple Kill Test ===" << std::endl;
    std::cout << "Rays: " << N_RAYS << std::endl;
    std::cout << "BVH16 vs Embree: N/A (per-scene comparison below)" << std::endl;
    std::cout << std::endl;
    
    std::vector<AABB> objects_dense, objects_sparse, objects_200, objects_1000, objects_5000;
    std::vector<std::array<float,3>> centers_dense, centers_sparse, centers_200, centers_1000, centers_5000;
    std::vector<Ray> rays_dense, rays_sparse, rays_200, rays_1000, rays_5000;
    
    generate_scene(rng, 20, 10000, 2000.0f, 800.0f, objects_dense, centers_dense, rays_dense);
    run_benchmark(objects_dense, rays_dense, "DENSE SCENE (20 clusters, 2000-unit spacing)");
    
    generate_scene(rng, 20, 10000, 10000.0f, 4000.0f, objects_sparse, centers_sparse, rays_sparse);
    run_benchmark(objects_sparse, rays_sparse, "SPARSE SCENE (20 clusters, 10000-unit spacing)");

    run_secondary_ray_benchmark(objects_dense, "DENSE SCENE (20 clusters, 2000-unit spacing)");
    
    generate_scene(rng, 200, 100000, 10000.0f, 4000.0f, objects_200, centers_200, rays_200);
    run_benchmark(objects_200, rays_200, "SPARSE SCENE (200 clusters, 10000-unit spacing)");
    
    generate_scene(rng, 1000, 500000, 10000.0f, 4000.0f, objects_1000, centers_1000, rays_1000);
    run_benchmark(objects_1000, rays_1000, "SPARSE SCENE (1000 clusters, 10000-unit spacing)");
    
    generate_scene(rng, 5000, 2500000, 10000.0f, 4000.0f, objects_5000, centers_5000, rays_5000);
    run_benchmark(objects_5000, rays_5000, "SPARSE SCENE (5000 clusters, 10000-unit spacing)");
    
    return 0;
}
