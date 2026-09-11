// Copyright 2026 Orders of Magnitude LLC
// Free for non-commercial and academic use.
// Commercial use requires a license from ofmagnitude.com

#include "bob.hpp"

int main() {
    std::cout << "BOB Kill Test - Materials Search Benchmark\n";
    std::cout << "===========================================\n";
    std::cout << "N = " << N << " materials\n";
    std::cout << "Properties = " << NUM_PROPERTIES << " (formation_energy, bandgap, density, melting_point, bulk_modulus)\n\n";
    
    generate_materials();
    std::cout << "Done.\n\n";
    
    std::cout << "Building BOB indices...\n";
    double index_build_time = build_bob_indices();
    std::cout << "Index build time: " << std::fixed << std::setprecision(4) << index_build_time << "s\n\n";
    
    run_benchmark("Wide Query (~25% pass)", generate_wide_query());
    run_benchmark("Tight Query (~0.1% pass)", generate_tight_query());
    run_batch_benchmark(1000);
    
    std::cout << "\n=== Summary ===\n";
    std::cout << "BOB index build time (one-time): " << std::fixed << std::setprecision(4) << index_build_time << "s\n";
    
    return 0;
}
