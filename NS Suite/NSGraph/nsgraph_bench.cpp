// Copyright (c) 2026 Orders of Magnitude LLC
// Licensed under the OOM Commercial License v1.0
// See LICENSE.md in the repository root or ofmagnitude.com

// Compile command:
// g++ -O3 -march=native -std=c++17 -o nsgraph_bench nsgraph_bench.cpp

#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>
#include <set>
#include <cstdlib>
#include "nsgraph.hpp"

using namespace std;
using namespace std::chrono;

uint32_t num_nodes = 1000000;
uint32_t num_communities = 100;
uint32_t community_size = 10000;

// Global ID maps for scrambling/reordering
vector<uint32_t> id_map;
vector<uint32_t> inverse_map;

// Graph A: Community graph (NS home turf) - SCRAMBLED
// 1,000,000 nodes, 10,000 communities of 100 nodes each
// Intra-community edges: each node connects to all others in its community (99 edges)
// Inter-community edges: each node connects to 2 random nodes outside community
// Node IDs are shuffled to scatter community members across random IDs
vector<pair<uint32_t,uint32_t>> generate_community_graph() {
    // Create ID shuffle map
    id_map.resize(num_nodes);
    for (uint32_t i = 0; i < num_nodes; i++) {
        id_map[i] = i;
    }
    mt19937 rng(42);
    shuffle(id_map.begin(), id_map.end(), rng);
    
    // Create inverse map for lookups
    inverse_map.resize(num_nodes);
    for (uint32_t i = 0; i < num_nodes; i++) {
        inverse_map[id_map[i]] = i;
    }
    
    vector<pair<uint32_t,uint32_t>> edges;
    
    for (uint32_t comm = 0; comm < num_communities; comm++) {
        uint32_t start = comm * community_size;
        
        // Intra-community edges: 10 random edges per node
        uniform_int_distribution<uint32_t> intra_dist(0, community_size - 1);
        for (uint32_t i = 0; i < community_size; i++) {
            uint32_t u = id_map[start + i];
            set<uint32_t> connected;
            for (int k = 0; k < 10; k++) {
                uint32_t j = intra_dist(rng);
                if (j != i && connected.insert(j).second) {
                    uint32_t v = id_map[start + j];
                    edges.push_back({u, v});
                    edges.push_back({v, u});
                }
            }
        }
        
        // Inter-community edges: 2 bridge edges per community
        uniform_int_distribution<uint32_t> outside_dist(0, num_nodes - 1);
        uint32_t bridge_u = id_map[start];
        for (int k = 0; k < 2; k++) {
            uint32_t v;
            do {
                v = outside_dist(rng);
            } while (inverse_map[v] / community_size == comm);
            edges.push_back({bridge_u, v});
            edges.push_back({v, bridge_u});
        }
    }
    
    return edges;
}

// Graph B: Random graph (NS away turf)
// 1,000,000 nodes, each node connects to 20 random nodes
vector<pair<uint32_t,uint32_t>> generate_random_graph() {
    vector<pair<uint32_t,uint32_t>> edges;
    mt19937 rng(42);
    uniform_int_distribution<uint32_t> node_dist(0, num_nodes - 1);
    
    for (uint32_t u = 0; u < num_nodes; u++) {
        for (int k = 0; k < 20; k++) {
            uint32_t v;
            do {
                v = node_dist(rng);
            } while (v == u);
            edges.push_back({u, v});
            edges.push_back({v, u});
        }
    }
    
    return edges;
}

double benchmark_bfs_standard(NSGraph& graph, const string& graph_name, uint32_t source, ofstream& out) {
    // Warm-up run
    graph.bfs_standard(source);
    
    // Benchmark bfs_standard (best of 3)
    double best_time = 1e9;
    vector<uint32_t> result;
    for (int run = 0; run < 3; run++) {
        auto start = high_resolution_clock::now();
        result = graph.bfs_standard(source);
        auto end = high_resolution_clock::now();
        double time_ms = duration_cast<microseconds>(end - start).count() / 1000.0;
        best_time = min(best_time, time_ms);
    }
    
    out << "=== " << graph_name << " ===" << endl;
    out << "Nodes visited: " << result.size() << endl;
    out << "BFS Standard: " << best_time << " ms" << endl;
    out << endl;
    
    cout << "=== " << graph_name << " ===" << endl;
    cout << "Nodes visited: " << result.size() << endl;
    cout << "BFS Standard: " << best_time << " ms" << endl;
    cout << endl;
    
    return best_time;
}

int main() {
    const char* home_raw = std::getenv("HOME");
    string home = home_raw ? home_raw : "";
    ofstream out(home + "/NS/Testing/NSGraph/results.txt");
    if (!out.is_open()) {
        cerr << "ERROR: cannot open results file: " << home << "/NS/Testing/NSGraph/results.txt" << endl;
        cerr << "       Directory does not exist. Create it first: mkdir -p " << home << "/NS/Testing/NSGraph" << endl;
        return 1;
    }
    
    out << "NS Graph Traversal Benchmark Results" << endl;
    out << "=====================================" << endl;
    out << "Compile: g++ -O3 -march=native -std=c++17 -o nsgraph_bench nsgraph_bench.cpp" << endl;
    out << endl;
    
    // Graph A: scrambled community graph
    cout << "Generating Graph A (Community graph, scrambled IDs)..." << endl;
    auto edges_a = generate_community_graph();
    cout << "Graph A edges: " << edges_a.size() << endl;
    
    NSGraph graph_a;
    graph_a.build(num_nodes, edges_a);
    
    // Benchmark Graph A: standard BFS only, BEFORE any reorder
    cout << "Benchmarking Graph A (scrambled)..." << endl;
    double time_a = benchmark_bfs_standard(graph_a, "Graph A - Scrambled Community", 0, out);
    
    // Graph C: copy of Graph A with reorder() applied
    cout << "Building Graph C (Graph A + reorder())..." << endl;
    NSGraph graph_c = graph_a;  // copy
    graph_c.reorder(community_size);
    
    // Benchmark Graph C: standard BFS on reordered graph
    cout << "Benchmarking Graph C (reordered)..." << endl;
    double time_c = benchmark_bfs_standard(graph_c, "Graph C - Reordered Community", 0, out);
    
    // Report reordering speedup
    double speedup = time_a / time_c;
    
    out << "=== Reordering Speedup ===" << endl;
    out << "Graph A (scrambled): " << time_a << " ms" << endl;
    out << "Graph C (reordered): " << time_c << " ms" << endl;
    out << "Speedup: " << speedup << "x" << endl;
    out << endl;
    
    cout << "=== Reordering Speedup ===" << endl;
    cout << "Graph A (scrambled): " << time_a << " ms" << endl;
    cout << "Graph C (reordered): " << time_c << " ms" << endl;
    cout << "Speedup: " << speedup << "x" << endl;
    cout << endl;
    
    // Graph B: random graph (NS away turf)
    cout << "Generating Graph B (Random graph)..." << endl;
    auto edges_b = generate_random_graph();
    cout << "Graph B edges: " << edges_b.size() << endl;
    
    NSGraph graph_b;
    graph_b.build(num_nodes, edges_b);
    
    cout << "Benchmarking Graph B..." << endl;
    benchmark_bfs_standard(graph_b, "Graph B - Random", 0, out);
    
    out.close();
    cout << "Results saved to " + home + "/NS/Testing/NSGraph/results.txt" << endl;
    
    return 0;
}
