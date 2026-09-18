// Orders of Magnitude LLC: OOM Commercial License v1.0
// https://ofmagnitude.com/license
// Copyright 2026 Orders of Magnitude LLC. All rights reserved.

#pragma once

#include <cstdint>
#include <cmath>
#include <algorithm>
#include <vector>
#include <cstring>
#include <hwy/contrib/sort/vqsort.h>

struct NSRay {
    float ox, oy, oz;
    float dx, dy, dz;
};

namespace nsray {

static constexpr uint32_t spread3_table[8] = {
    0, 1, 8, 9, 64, 65, 72, 73
};

static constexpr uint32_t spread4_table[16] = {
    0, 1, 8, 9, 64, 65, 72, 73,
    512, 513, 520, 521, 576, 577, 584, 585
};

inline uint32_t cell_morton3(uint32_t cx, uint32_t cy, uint32_t cz) {
    return spread3_table[cx & 7] | (spread3_table[cy & 7] << 1) | (spread3_table[cz & 7] << 2);
}

inline uint32_t dir_morton3(uint32_t dx, uint32_t dy, uint32_t dz) {
    return spread4_table[dx & 15] | (spread4_table[dy & 15] << 1) | (spread4_table[dz & 15] << 2);
}

inline uint32_t joint_key(const NSRay& r, const float scene_min[3], const float scene_max[3]) {
    float inv_ext[3] = {
        8.0f / (scene_max[0] - scene_min[0]),
        8.0f / (scene_max[1] - scene_min[1]),
        8.0f / (scene_max[2] - scene_min[2])
    };
    uint32_t cx = std::min(7u, (uint32_t)((r.ox - scene_min[0]) * inv_ext[0]));
    uint32_t cy = std::min(7u, (uint32_t)((r.oy - scene_min[1]) * inv_ext[1]));
    uint32_t cz = std::min(7u, (uint32_t)((r.oz - scene_min[2]) * inv_ext[2]));
    uint32_t cell = cell_morton3(cx, cy, cz);

    float inv_len = 1.0f / std::sqrt(r.dx * r.dx + r.dy * r.dy + r.dz * r.dz);
    float nx = r.dx * inv_len, ny = r.dy * inv_len, nz = r.dz * inv_len;
    uint32_t dx = std::min(15u, (uint32_t)((nx * 0.5f + 0.5f) * 15.0f));
    uint32_t dy = std::min(15u, (uint32_t)((ny * 0.5f + 0.5f) * 15.0f));
    uint32_t dz = std::min(15u, (uint32_t)((nz * 0.5f + 0.5f) * 15.0f));
    uint32_t dir = dir_morton3(dx, dy, dz);

    return (cell << 12) | dir;
}

// Sort rays by pre-computed keys using Highway VQSort.
// Packs (key, index) into uint64_t pairs, VQSorts, then applies permutation.
inline void radix_sort_rays(NSRay* rays, uint32_t* keys, int n) {
    if (n <= 1) return;

    uint64_t* pairs = new uint64_t[n];
    for (int i = 0; i < n; i++)
        pairs[i] = ((uint64_t)keys[i] << 32) | (uint32_t)i;

    hwy::VQSort(pairs, (size_t)n, hwy::SortAscending());

    NSRay* sorted_rays = new NSRay[n];
    for (int i = 0; i < n; i++)
        sorted_rays[i] = rays[(uint32_t)pairs[i]];
    std::memcpy(rays, sorted_rays, (size_t)n * sizeof(NSRay));

    delete[] sorted_rays;
    delete[] pairs;
}

inline void sort_rays(NSRay* rays, int n, float scene_min[3], float scene_max[3]) {
    uint32_t* keys = new uint32_t[n];
    for (int i = 0; i < n; i++)
        keys[i] = joint_key(rays[i], scene_min, scene_max);
    radix_sort_rays(rays, keys, n);
    delete[] keys;
}

inline void sort_rays(std::vector<NSRay>& rays, float scene_min[3], float scene_max[3]) {
    sort_rays(rays.data(), (int)rays.size(), scene_min, scene_max);
}

}
