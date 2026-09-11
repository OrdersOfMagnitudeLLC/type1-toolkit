// NSMace — OOM Proprietary, All Rights Reserved, Orders of Magnitude LLC
#include "NSMace.h"
#include <vector>
#include <cmath>
#include <unordered_map>
#include <cstdint>
#include <algorithm>

namespace NSMace {

void NeighborList::build(const std::vector<Atom>& atoms, Real cutoff_A) {
    build(atoms, cutoff_A, {{0,0,0}});
}

void NeighborList::build(const std::vector<Atom>& atoms, Real cutoff_A, const std::array<Real,3>& box) {
    n_atoms = (int)atoms.size();
    cutoff  = cutoff_A;
    box_lengths = box;
    neighbors.assign(n_atoms, {});
    if (n_atoms == 0) return;
    rebuild_count++;
    update_last_positions(atoms);

    // Bounding box
    Real xmin = atoms[0].x, ymin = atoms[0].y, zmin = atoms[0].z;
    for (auto& a : atoms) {
        xmin = std::min(xmin, a.x);
        ymin = std::min(ymin, a.y);
        zmin = std::min(zmin, a.z);
    }

    Real rc2 = cutoff_A * cutoff_A;
    Real inv_cut = 1.0f / cutoff_A;

    // Cell key: pack (ix,iy,iz) into uint64 — stride chosen so no collisions
    // up to 100K atoms in a ~200A box (cell count <1000 per axis)
    auto make_key = [](int64_t ix, int64_t iy, int64_t iz) -> uint64_t {
        return (uint64_t)(ix + 1000) +
               (uint64_t)(iy + 1000) * 100000ULL +
               (uint64_t)(iz + 1000) * 10000000000ULL;
    };

    // Build grid: cell key -> atom indices
    std::unordered_map<uint64_t, std::vector<int>> grid;
    grid.reserve(n_atoms * 2);
    for (int i = 0; i < n_atoms; i++) {
        int64_t ix = (int64_t)((atoms[i].x - xmin) * inv_cut);
        int64_t iy = (int64_t)((atoms[i].y - ymin) * inv_cut);
        int64_t iz = (int64_t)((atoms[i].z - zmin) * inv_cut);
        grid[make_key(ix, iy, iz)].push_back(i);
    }

    // Each atom checks only 27 neighboring cells
    for (int i = 0; i < n_atoms; i++) {
        Real ax = atoms[i].x, ay = atoms[i].y, az = atoms[i].z;
        int64_t cx = (int64_t)((ax - xmin) * inv_cut);
        int64_t cy = (int64_t)((ay - ymin) * inv_cut);
        int64_t cz = (int64_t)((az - zmin) * inv_cut);
        for (int ddx = -1; ddx <= 1; ddx++)
        for (int ddy = -1; ddy <= 1; ddy++)
        for (int ddz = -1; ddz <= 1; ddz++) {
            auto it = grid.find(make_key(cx+ddx, cy+ddy, cz+ddz));
            if (it == grid.end()) continue;
            for (int j : it->second) {
                if (i == j) continue;
                Real dx = atoms[j].x - ax;
                Real dy = atoms[j].y - ay;
                Real dz = atoms[j].z - az;
                
                // Apply minimum image convention for PBC
                if (box[0] > 0) dx -= box[0] * std::round(dx / box[0]);
                if (box[1] > 0) dy -= box[1] * std::round(dy / box[1]);
                if (box[2] > 0) dz -= box[2] * std::round(dz / box[2]);
                
                Real r2 = dx*dx + dy*dy + dz*dz;
                if (r2 < rc2) {
                    neighbors[i].push_back({j, dx, dy, dz, std::sqrt(r2)});
                }
            }
        }
    }
}

void NeighborList::recompute(const std::vector<Atom>& atoms) {
    for (int i = 0; i < n_atoms; i++) {
        Real ax = atoms[i].x, ay = atoms[i].y, az = atoms[i].z;
        for (auto& nb : neighbors[i]) {
            Real dx = atoms[nb.j].x - ax;
            Real dy = atoms[nb.j].y - ay;
            Real dz = atoms[nb.j].z - az;
            nb.dx = dx;
            nb.dy = dy;
            nb.dz = dz;
            nb.r  = std::sqrt(dx*dx + dy*dy + dz*dz);
        }
    }
}

void NeighborhoodHash::compute(const std::vector<Atom>& atoms,
                                const NeighborList& nl) {
    hashes.resize(atoms.size());
    for (int i = 0; i < (int)atoms.size(); i++) {
        // FNV-1a hash over quantized neighbor distances + types
        uint64_t h = 14695981039346656037ULL;
        for (auto& nb : nl.neighbors[i]) {
            // Quantize to 0.01 Angstrom grid
            int32_t qr  = (int32_t)(nb.r  / 0.01f);
            int32_t qdx = (int32_t)(nb.dx / 0.01f);
            int32_t qdy = (int32_t)(nb.dy / 0.01f);
            int32_t qdz = (int32_t)(nb.dz / 0.01f);
            auto mix = [&](uint32_t v) {
                h ^= v; h *= 1099511628211ULL;
            };
            mix((uint32_t)qr);
            mix((uint32_t)qdx);
            mix((uint32_t)qdy);
            mix((uint32_t)qdz);
            mix((uint32_t)atoms[nb.j].type);
        }
        hashes[i] = h;
    }
}

bool NeighborList::needs_rebuild(const std::vector<Atom>& atoms) const {
    if (last_positions.size() != atoms.size()) return true;
    
    Real skin_half = skin_distance * Real(0.5);
    Real max_disp = Real(0.0);
    
    for (size_t i = 0; i < atoms.size(); i++) {
        Real dx = atoms[i].x - last_positions[i][0];
        Real dy = atoms[i].y - last_positions[i][1];
        Real dz = atoms[i].z - last_positions[i][2];
        Real disp = std::sqrt(dx*dx + dy*dy + dz*dz);
        max_disp = std::max(max_disp, disp);
        if (max_disp > skin_half) return true;
    }
    return false;
}

void NeighborList::update_last_positions(const std::vector<Atom>& atoms) {
    last_positions.resize(atoms.size());
    for (size_t i = 0; i < atoms.size(); i++) {
        last_positions[i] = {atoms[i].x, atoms[i].y, atoms[i].z};
    }
}

std::vector<bool> NeighborhoodHash::changed(const NeighborhoodHash& prev) const {
    std::vector<bool> mask(hashes.size(), true);
    if (prev.hashes.size() != hashes.size()) return mask;
    for (size_t i = 0; i < hashes.size(); i++)
        mask[i] = (hashes[i] != prev.hashes[i]);
    return mask;
}

} // namespace NSMace
