// NSMace: OOM Proprietary, All Rights Reserved, Orders of Magnitude LLC
// ML force field engine: C++ rewrite of MACE-MP-0 architecture
// Target: 10,000x MACE-Python on CPU via AVX-512 + structural gap exploitation

#pragma once
#include <vector>
#include <array>
#include <cmath>
#include <cstdint>

namespace NSMace {

#ifdef NSMACE_FP64
using Real = double;
#else
using Real = float;
#endif

// ── Atom ──────────────────────────────────────────────────────────
struct Atom {
    Real x, y, z;          // position (Angstrom)
    int  type;             // element index
    int  id;               // global atom index
};

// ── Neighbor entry ────────────────────────────────────────────────
struct Neighbor {
    int  j;                // neighbor atom index
    Real dx, dy, dz;       // displacement vector (Angstrom)
    Real r;                // distance
};

// ── Per-atom neighbor list ────────────────────────────────────────
struct NeighborList {
    std::vector<std::vector<Neighbor>> neighbors;  // [atom_i] → list
    int   n_atoms;
    Real  cutoff;
    Real  skin_distance = 0.5;  // Skin distance for MD caching (default 0.5A)
    std::vector<std::array<Real,3>> last_positions;  // Positions at last build
    int rebuild_count = 0;  // Track number of rebuilds
    std::array<Real,3> box_lengths = {{0,0,0}};  // PBC box lengths (0 = no PBC)

    void build(const std::vector<Atom>& atoms, Real cutoff_A);
    void build(const std::vector<Atom>& atoms, Real cutoff_A, const std::array<Real,3>& box);
    // Recompute r, dx, dy, dz for the existing neighbor pairs.
    void recompute(const std::vector<Atom>& atoms);
    // Check if rebuild is needed based on max displacement since last build
    bool needs_rebuild(const std::vector<Atom>& atoms) const;
    // Update last positions after build
    void update_last_positions(const std::vector<Atom>& atoms);
};

// ── Gap 2: neighborhood hash for skip-recompute ───────────────────
struct NeighborhoodHash {
    std::vector<uint64_t> hashes;   // [atom_i] → hash of neighbor config
    float delta_threshold = 0.01f;  // Angstrom: skip if max displacement < this

    void compute(const std::vector<Atom>& atoms, const NeighborList& nl);
    // Returns bitmask of atoms whose neighborhoods changed since last call
    std::vector<bool> changed(const NeighborhoodHash& prev) const;
};

// ── Result ────────────────────────────────────────────────────────
struct Result {
    Real energy_eV;
    std::vector<std::array<Real,3>> forces_eV_per_A;
    bool converged;
};

// ── Space group symmetry operation ───────────────────────────────────
struct SymmetryOperation {
    std::array<std::array<Real,3>,3> rotation;  // 3x3 rotation matrix
    std::array<Real,3> translation;             // translation vector (fractional coords)
};

// ── Crystal symmetry tiling ─────────────────────────────────────────
struct CrystalSymmetry {
    int Z;  // Number of unique atoms in unit cell
    int N_supercell;  // Total atoms in supercell
    std::vector<int> atom_mapping;  // Maps supercell atom i -> unit cell atom j
    
    // Initialize with unit cell size
    void init(int unit_cell_size) {
        Z = unit_cell_size;
    }
    
    // Map supercell atoms to unit cell equivalents based on periodic tiling
    void build_mapping(const std::vector<Atom>& supercell, const std::vector<Atom>& unit_cell) {
        N_supercell = (int)supercell.size();
        atom_mapping.resize(supercell.size());
        
        // For periodic tiling, atoms repeat in order
        for (size_t i = 0; i < supercell.size(); i++) {
            atom_mapping[i] = i % Z;
        }
    }
    
    // Apply force tiling: direct copy for identical molecules
    void tile_forces(const std::vector<Real>& unit_forces, std::vector<Real>& super_forces) const {
        if (N_supercell <= 0 || atom_mapping.size() != (size_t)N_supercell) {
            super_forces.clear();
            return;
        }
        super_forces.resize((size_t)N_supercell * 3);
        for (int i = 0; i < N_supercell; i++) {
            int j = atom_mapping[i];
            if (j >= 0 && j < Z) {
                super_forces[i * 3 + 0] = unit_forces[j * 3 + 0];
                super_forces[i * 3 + 1] = unit_forces[j * 3 + 1];
                super_forces[i * 3 + 2] = unit_forces[j * 3 + 2];
            }
        }
    }
};

} // namespace NSMace
