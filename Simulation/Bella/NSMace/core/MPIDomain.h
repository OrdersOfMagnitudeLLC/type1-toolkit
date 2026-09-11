// Copyright (c) 2026 Orders of Magnitude LLC
// Licensed under the OOM Commercial License v1.0
// See LICENSE.md in the repository root or ofmagnitude.com

#pragma once

#include <vector>
#include <array>
#include <memory>

namespace NSMace {

// Forward declarations
struct Atom;
struct Vec3;

/**
 * MPIDomain - Domain decomposition structure for billion-atom scale simulations
 * 
 * Architecture sketch for MPI parallelization:
 * - Split simulation box into NxNxN spatial domains (one per MPI rank)
 * - Each rank owns atoms in its local domain
 * - Ghost atoms within cutoff distance from domain boundaries are exchanged
 * - Ghost positions are updated via neighbor communication
 * - Forces on ghost atoms are accumulated back to owning ranks
 */
struct MPIDomain {
    // Domain geometry
    int rank;                           // MPI rank of this domain
    int num_ranks;                      // Total number of MPI ranks
    std::array<int, 3> grid_size;       // NxNxN domain grid dimensions
    std::array<int, 3> grid_coords;     // (ix, iy, iz) coordinates of this domain in grid
    
    // Domain boundaries (in Cartesian coordinates)
    Vec3 domain_min;                    // Minimum corner of local domain
    Vec3 domain_max;                    // Maximum corner of local domain
    Vec3 box_min;                       // Global simulation box minimum
    Vec3 box_max;                       // Global simulation box maximum
    
    // Cutoff for ghost atom exchange
    double cutoff;                      // Interaction cutoff distance
    double ghost_skin;                  // Extra skin distance for ghost atoms
    
    // Local atoms owned by this rank
    std::vector<Atom> local_atoms;      // Atoms whose positions are in local domain
    std::vector<int> local_atom_ids;    // Global atom IDs for local atoms
    
    // Ghost atoms from neighboring domains
    std::vector<Atom> ghost_atoms;      // Ghost atoms from neighbor domains
    std::vector<int> ghost_atom_ids;    // Global atom IDs for ghost atoms
    std::vector<int> ghost_owner_ranks; // Owning rank for each ghost atom
    
    // Neighbor domain information
    struct NeighborInfo {
        int rank;                       // MPI rank of neighbor
        std::array<int, 3> grid_coords; // Grid coordinates of neighbor
        Vec3 domain_min;                // Neighbor domain minimum
        Vec3 domain_max;                // Neighbor domain maximum
        std::vector<int> send_indices;  // Indices of local atoms to send to this neighbor
        std::vector<int> recv_indices;  // Indices of ghost atoms received from this neighbor
    };
    std::vector<NeighborInfo> neighbors; // All 26 neighboring domains (3D)
    
    // Communication buffers
    struct CommBuffers {
        std::vector<double> send_positions;  // Buffer for sending positions
        std::vector<double> recv_positions;  // Buffer for receiving positions
        std::vector<double> send_forces;     // Buffer for sending forces
        std::vector<double> recv_forces;     // Buffer for receiving forces
        std::vector<int> send_ids;           // Atom IDs to send
        std::vector<int> recv_ids;           // Atom IDs received
    };
    CommBuffers comm_buffers;
    
    // Force accumulation for ghost atoms
    std::vector<Vec3> ghost_forces;     // Forces on ghost atoms (to be sent back)
    
    /**
     * Initialize domain decomposition
     * @param box_min Global simulation box minimum
     * @param box_max Global simulation box maximum
     * @param cutoff Interaction cutoff distance
     * @param ghost_skin Extra skin for ghost atoms
     */
    void initialize(const Vec3& box_min, const Vec3& box_max, 
                   double cutoff, double ghost_skin = 0.0);
    
    /**
     * Compute domain grid dimensions from number of ranks
     * @param num_ranks Total number of MPI ranks
     * @return Grid dimensions (Nx, Ny, Nz)
     */
    std::array<int, 3> compute_grid_size(int num_ranks);
    
    /**
     * Assign atoms to domains based on their positions
     * @param all_atoms All atoms in the simulation
     * @return Vector of atom counts per rank
     */
    std::vector<int> assign_atoms_to_domains(const std::vector<Atom>& all_atoms);
    
    /**
     * Build neighbor list for this domain
     * Identifies all 26 neighboring domains in 3D grid
     */
    void build_neighbor_list();
    
    /**
     * Identify ghost atoms needed from neighbors
     * Ghost atoms are those within cutoff + skin of domain boundary
     */
    void identify_ghost_atoms();
    
    /**
     * Exchange ghost atom positions with neighbors
     * Sends local atom positions to neighbors, receives ghost positions
     */
    void exchange_ghost_positions();
    
    /**
     * Accumulate forces on ghost atoms back to owning ranks
     * Forces computed on ghost atoms are sent to their owning ranks
     */
    void reduce_ghost_forces();
    
    /**
     * Update local atom positions after force reduction
     * Combines local forces with forces accumulated from ghost atoms
     */
    void update_local_forces();
    
    /**
     * Check if a point is within the local domain
     * @param point Point to check
     * @return True if point is in local domain (inclusive)
     */
    bool is_in_local_domain(const Vec3& point) const;
    
    /**
     * Check if a point is within ghost region of local domain
     * @param point Point to check
     * @return True if point is within cutoff + skin of domain boundary
     */
    bool is_in_ghost_region(const Vec3& point) const;
    
    /**
     * Get the MPI rank that owns a given position
     * @param point Position in global coordinates
     * @return MPI rank that owns this position
     */
    int get_owner_rank(const Vec3& point) const;
    
    /**
     * Get grid coordinates from MPI rank
     * @param rank MPI rank
     * @return Grid coordinates (ix, iy, iz)
     */
    std::array<int, 3> rank_to_grid_coords(int rank) const;
    
    /**
     * Get MPI rank from grid coordinates
     * @param coords Grid coordinates (ix, iy, iz)
     * @return MPI rank
     */
    int grid_coords_to_rank(const std::array<int, 3>& coords) const;
    
    /**
     * Compute domain boundaries for a given grid position
     * @param grid_coords Grid coordinates (ix, iy, iz)
     * @return Pair of (domain_min, domain_max)
     */
    std::pair<Vec3, Vec3> compute_domain_bounds(
        const std::array<int, 3>& grid_coords) const;
    
    /**
     * Clean up and deallocate resources
     */
    void finalize();
};

/**
 * MPI communicator wrapper for domain decomposition
 * Provides interface to MPI communication primitives
 */
struct MPICommunicator {
    int rank;
    int size;
    
    /**
     * Initialize MPI communicator
     */
    void initialize();
    
    /**
     * Finalize MPI communicator
     */
    void finalize();
    
    /**
     * Send data to a specific rank
     * @param dest_rank Destination rank
     * @param data Data buffer to send
     * @param count Number of elements
     * @param tag Message tag
     */
    void send(int dest_rank, const double* data, int count, int tag);
    
    /**
     * Receive data from a specific rank
     * @param src_rank Source rank
     * @param data Data buffer to receive into
     * @param count Number of elements
     * @param tag Message tag
     */
    void recv(int src_rank, double* data, int count, int tag);
    
    /**
     * Non-blocking send
     * @param dest_rank Destination rank
     * @param data Data buffer to send
     * @param count Number of elements
     * @param tag Message tag
     * @return Request handle
     */
    void* isend(int dest_rank, const double* data, int count, int tag);
    
    /**
     * Non-blocking receive
     * @param src_rank Source rank
     * @param data Data buffer to receive into
     * @param count Number of elements
     * @param tag Message tag
     * @return Request handle
     */
    void* irecv(int src_rank, double* data, int count, int tag);
    
    /**
     * Wait for communication to complete
     * @param request Request handle
     */
    void wait(void* request);
    
    /**
     * Barrier synchronization across all ranks
     */
    void barrier();
    
    /**
     * All-reduce operation across all ranks
     * @param send_buf Send buffer
     * @param recv_buf Receive buffer
     * @param count Number of elements
     * @param op Reduction operation (sum, max, min, etc.)
     */
    void all_reduce(const double* send_buf, double* recv_buf, int count, int op);
};

} // namespace NSMace
